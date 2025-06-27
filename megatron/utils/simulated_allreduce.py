"""
simulated_allreduce.py

Override torch.distributed primitives to simulate packet loss on
reduce-scatter and all-gather in Megatron-LM’s data-parallel collectives.

Usage:
    Set environment variables:
        SIM_RS_DROP_PROB: drop probability for reduce-scatter [0.0,1.0]
        SIM_AG_DROP_PROB: drop probability for all-gather [0.0,1.0]
    Import this module before initializing distributed training:
        import simulated_allreduce
    Ensure this import occurs before any call to mpu.initialize() or
torch.distributed.init_process_group().
"""
import os
import random
import logging

import torch
import torch.distributed as dist
from megatron.core import mpu

# === ADD THIS BEFORE ANY logger calls ===
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s",
                    level=logging.INFO)

print(f"[simulated_allreduce] dropping RS at p={os.getenv('PACKET_LOSS_PROB_RS')} "
      f"AG at p={os.getenv('PACKET_LOSS_PROB_AG')}")

logger = logging.getLogger(__name__)
# (you can also explicitly set the module logger level, though basicConfig is usually enough)
logger.setLevel(logging.INFO)

# Keep originals
_orig_reduce_scatter = dist.reduce_scatter
# Prefer the new API, fallback to private function
if hasattr(dist, 'reduce_scatter_tensor'):
    _orig_rsb = dist.reduce_scatter_tensor
else:
    _orig_rsb = dist._reduce_scatter_base
_orig_all_gather = dist.all_gather

# Packet-loss probabilities
_rs_drop = float(os.getenv("PACKET_LOSS_PROB_RS", "0.0"))
_ag_drop = float(os.getenv("PACKET_LOSS_PROB_AG", "0.0"))
logger.info(f"simulated_allreduce: rs_drop={_rs_drop}, ag_drop={_ag_drop}")

# Caches for last successful shards
_last_good_rs_list = {}
_last_good_rsb     = {}
_last_good_ag      = {}


def _get_dp_group():
    """
    Lazily retrieve the data-parallel group once initialized.
    Raises if groups are not yet set up.
    """
    dp = mpu.get_data_parallel_group()
    if dp is None:
        raise RuntimeError("Data parallel group is not initialized yet")
    return dp


def simulated_reduce_scatter(output, input_list, group=None, op=dist.ReduceOp.SUM, async_op=False):
    # Determine DP group at call time
    try:
        dp_group = _get_dp_group()
    except RuntimeError:
        # fallback to original behavior if not initialized
        return _orig_reduce_scatter(output, input_list, group=group, op=op, async_op=async_op)
    group_norm = dp_group if group is None else group
    if group_norm != dp_group:
        return _orig_reduce_scatter(output, input_list, group=group, op=op, async_op=async_op)
    rank = dist.get_rank(dp_group)
    new_input = []
    for idx, tensor in enumerate(input_list):
        if idx == rank and random.random() < _rs_drop:
            fallback = _last_good_rs_list.get(idx)
            if fallback is None:
                fallback = tensor.clone()
            logger.debug(f"RS: dropping chunk {idx}, using fallback")
            new_input.append(fallback.clone())
        else:
            if idx == rank:
                _last_good_rs_list[idx] = tensor.clone()
                logger.debug(f"RS: sending fresh chunk {idx}")
            new_input.append(tensor)
    return _orig_reduce_scatter(output, new_input, group=dp_group, op=op, async_op=async_op)


def simulated_reduce_scatter_base(output, input_tensor, group=None, async_op=False):
    """
    Handles both `reduce_scatter_tensor` and `_reduce_scatter_base` signatures.
    """
    try:
        dp_group = _get_dp_group()
    except RuntimeError:
        return _orig_rsb(output, input_tensor, group=group, async_op=async_op)
    group_norm = dp_group if group is None else group
    if group_norm != dp_group:
        return _orig_rsb(output, input_tensor, group=group, async_op=async_op)
    world_size = dist.get_world_size(dp_group)
    rank       = dist.get_rank(dp_group)
    chunks = list(input_tensor.chunk(world_size))
    new_chunks = []
    for idx, chunk in enumerate(chunks):
        if idx == rank and random.random() < _rs_drop:
            last = _last_good_rsb.get(idx)
            if last is None:
                last = chunk.clone()
            logger.debug(f"RSB: dropping chunk {idx}, using fallback")
            new_chunks.append(last.clone())
        else:
            if idx == rank:
                _last_good_rsb[idx] = chunk.clone()
                logger.debug(f"RSB: sending fresh chunk {idx}")
            new_chunks.append(chunk)
    new_input = torch.cat(new_chunks, dim=0)
    return _orig_rsb(output, new_input, group=dp_group, async_op=async_op)


def simulated_all_gather(output_list, input_tensor, group=None, async_op=False):
    # Determine DP group at call time
    try:
        dp_group = _get_dp_group()
    except RuntimeError:
        return _orig_all_gather(output_list, input_tensor, group=group, async_op=async_op)
    group_norm = dp_group if group is None else group
    if group_norm != dp_group:
        return _orig_all_gather(output_list, input_tensor, group=group, async_op=async_op)
    # Perform the real all_gather
    res = _orig_all_gather(output_list, input_tensor, group=dp_group, async_op=async_op)
    world_size = dist.get_world_size(dp_group)
    for idx in range(world_size):
        if random.random() < _ag_drop:
            fallback = _last_good_ag.get(idx)
            if fallback is not None:
                logger.debug(f"AG: dropping chunk {idx}, using fallback")
                output_list[idx] = fallback.clone()
        else:
            _last_good_ag[idx] = output_list[idx].clone()
            logger.debug(f"AG: received fresh chunk {idx}")
    return res

# Apply patches
# Collective list-based RS
dist.reduce_scatter = simulated_reduce_scatter
# Tensor-based RS APIs
if hasattr(dist, 'reduce_scatter_tensor'):
    dist.reduce_scatter_tensor = simulated_reduce_scatter_base
if hasattr(dist, '_reduce_scatter_base'):
    dist._reduce_scatter_base = simulated_reduce_scatter_base
# All-gather
dist.all_gather = simulated_all_gather

logger.info("simulated_allreduce: patches applied")
