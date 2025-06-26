
import torch
import torch.distributed as dist
import os
import random
import logging
from megatron.core import mpu

# Setup per-rank logger
_rank = dist.get_rank() if dist.is_initialized() else -1
logger = logging.getLogger(f"sim-allreduce-rank{_rank}")
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)

# Save original function
_original_all_reduce = dist.all_reduce

# Track last good shard per stage (RS and AG)
_last_good_shard_rs = {}
_last_good_shard_ag = {}

# Flag to restrict to gradient synchronization
SIMULATE_GRADIENT_SYNC_ONLY = os.getenv("SIMULATE_GRADIENT_SYNC_ONLY", "1") == "1"

def simulated_all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):

    if not hasattr(simulated_all_reduce, "_warned"):
        logger.info(f"[RANK {_rank}] Using simulated_all_reduce override")
        simulated_all_reduce._warned = True

    if group != mpu.get_data_parallel_group():
        return _original_all_reduce(tensor, op=op, group=group, async_op=async_op)

    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)

    if SIMULATE_GRADIENT_SYNC_ONLY:
        # Skip loss/stat tensors (e.g. size < world_size * 4)
        if tensor.numel() < world_size * 4:
            return _original_all_reduce(tensor, op=op, group=group, async_op=async_op)

    # Reduce-Scatter simulation
    chunks = tensor.chunk(world_size)
    if len(chunks) != world_size:
        return _original_all_reduce(tensor, op=op, group=group, async_op=async_op)

    recv_chunk = torch.zeros_like(chunks[rank])
    loss_prob_rs = float(os.getenv("PACKET_LOSS_PROB_RS", 0.0))
    dropped_rs = random.random() < loss_prob_rs

    if dropped_rs and (group, rank) in _last_good_shard_rs:
        recv_chunk.copy_(_last_good_shard_rs[(group, rank)])
        logger.info(f"[RANK {rank}] RS: using previous good shard due to drop")
    elif not dropped_rs:
        dist.reduce_scatter(recv_chunk, list(chunks), group=group, op=op)
        _last_good_shard_rs[(group, rank)] = recv_chunk.clone()
    else:
        recv_chunk.zero_()
        logger.info(f"[RANK {rank}] RS: drop occurred with no fallback, using zero")

    # All-Gather simulation
    gather_input = recv_chunk.clone()
    gathered_chunks = [torch.zeros_like(gather_input) for _ in range(world_size)]
    loss_prob_ag = float(os.getenv("PACKET_LOSS_PROB_AG", 0.0))
    dropped_ag = random.random() < loss_prob_ag

    if dropped_ag and (group, rank) in _last_good_shard_ag:
        gather_input.copy_(_last_good_shard_ag[(group, rank)])
        logger.info(f"[RANK {rank}] AG: using previous good shard due to drop")
    elif not dropped_ag:
        _last_good_shard_ag[(group, rank)] = gather_input.clone()
    else:
        gather_input.zero_()
        logger.info(f"[RANK {rank}] AG: drop occurred with no fallback, using zero")

    dist.all_gather(gathered_chunks, gather_input, group=group)
    tensor.copy_(torch.cat(gathered_chunks))

    return dist.Work() if async_op else None

# Apply patch globally
dist.all_reduce = simulated_all_reduce
