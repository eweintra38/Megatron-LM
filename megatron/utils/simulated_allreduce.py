"""
simulated_allreduce.py

Override torch.distributed tensor-based primitives to simulate packet loss on
reduce-scatter and all-gather in Megatron-LM’s ZeRO-1 distributed-optimizer path.

Usage:
    Set environment variables:
        PACKET_LOSS_PROB_RS: drop probability for reduce-scatter [0.0,1.0]
        PACKET_LOSS_PROB_AG: drop probability for all-gather [0.0,1.0]
    Import this module before initializing distributed training:
        import simulated_allreduce
    Ensure this import occurs before any call to mpu.initialize() or
    torch.distributed.init_process_group().
"""
import os
import random
import logging
import torch
from torch import no_grad

torch_dist = torch.distributed
from megatron.core import mpu

# Configure logging
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
print(f"[simulated_allreduce] loaded, RS p={os.getenv('PACKET_LOSS_PROB_RS')} AG p={os.getenv('PACKET_LOSS_PROB_AG')}")
logger = logging.getLogger(__name__)

# Store original tensor-based collectives
torch_rsb = getattr(torch_dist, 'reduce_scatter_tensor', None)
torch_rsb_base = getattr(torch_dist, '_reduce_scatter_base', None)
torch_agt = getattr(torch_dist, 'all_gather_into_tensor', None)
torch_agb = getattr(torch_dist, '_all_gather_base', None)

# Packet-loss probabilities
_rs_drop = float(os.getenv('PACKET_LOSS_PROB_RS', '0.0'))
_ag_drop = float(os.getenv('PACKET_LOSS_PROB_AG', '0.0'))
logger.info(f"simulated_allreduce: rs_drop={_rs_drop}, ag_drop={_ag_drop}")

# Caches for last successful shards
_last_good_rs = {}
_last_good_ag = {}

def _get_dp_group():
    dp = mpu.get_data_parallel_group()
    if dp is None:
        raise RuntimeError("DP group not initialized")
    return dp

def _normalize_group(group):
    try:
        dp = _get_dp_group()
    except RuntimeError:
        return None
    return dp if group is None else group


# simulated_reduce_scatter_tensor removed to avoid autograd double-backward errors
# Gradient sync will use default ZeRO-1 reduce-scatter without drop simulation



def simulated_all_gather_into_tensor(output_tensor, input_tensor, group=None, async_op=False):
    dp_group = _normalize_group(group)
    # if not our DP group, call original
    if dp_group is None or dp_group != _get_dp_group():
        if torch_agt:
            return torch_agt(output_tensor, input_tensor, group=group, async_op=async_op)
        return torch_agb(output_tensor, input_tensor, group=group, async_op=async_op)

    # perform actual all_gather_into_tensor
    if torch_agt:
        res = torch_agt(output_tensor, input_tensor, group=dp_group, async_op=async_op)
    else:
        res = torch_agb(output_tensor, input_tensor, group=dp_group, async_op=async_op)

    # simulate packet loss on the reconstructed buffer
    world = torch_dist.get_world_size(dp_group)
    flat = output_tensor.view(-1)
    size = flat.numel() // world
    with no_grad():
        for idx in range(world):
            seg = flat[idx*size:(idx+1)*size]
            if random.random() < _ag_drop:
                fb = _last_good_ag.get(idx)
                if fb is not None:
                    logger.info(f"dropping AG shard {idx}")
                    seg.copy_(fb)
            else:
                logger.info(f"caching AG shard {idx}")
                _last_good_ag[idx] = seg.clone()
    return res

# Apply patches on tensor-based collectives
# NOTE: We only override all_gather into tensor to avoid double-backward errors
if torch_agt:
    torch_dist.all_gather_into_tensor = simulated_all_gather_into_tensor
if torch_agb:
    torch_dist._all_gather_base = simulated_all_gather_into_tensor

logger.info("[simulated_allreduce] all_gather patches applied successfully; reduce_scatter left unmodified")

logger.info("[simulated_allreduce] tensor-based patches applied successfully")
