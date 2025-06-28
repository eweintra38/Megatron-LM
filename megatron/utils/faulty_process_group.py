# faulty_process_group.py
#
# A PyTorch wrapper to simulate packet loss in reduce_scatter and all_gather
# for Megatron-LM’s ZeRO-1 path, without breaking autograd hooks.
#
# Usage:
#   1. Place this file on PYTHONPATH before training.
#   2. In your launcher (before dist.init_process_group):
#        from megatron.utils.faulty_process_group import install_faulty_pg
#        install_faulty_pg()
#   3. Launch Megatron as usual with --use-distributed-optimizer.

import os
import random
import torch
from torch import no_grad
from megatron.core import mpu

# Packet-loss probabilities (from env vars)
_RS_DROP = float(os.getenv('PACKET_LOSS_PROB_RS', '0.0'))

# Fallback buffers (if packet is dropped)
_last_good_rs = {}

def _is_dp(group):
    """Check if the given group is the data parallel group."""
    dp = mpu.get_data_parallel_group()
    return group is None or group == dp

def install_faulty_pg():
    import torch.distributed as dist  # Ensure dist is local in scope

    original_init = dist.init_process_group

    def wrapped_init(*args, **kwargs):
        original_init(*args, **kwargs)
        _patch_collectives()

    dist.init_process_group = wrapped_init

def _patch_collectives():
    import torch.distributed as dist

    original_reduce_scatter = dist.reduce_scatter_tensor

    def faulty_reduce_scatter_tensor(output, input, group=None, async_op=False):
        ret = original_reduce_scatter(output, input, group, async_op)
        if _is_dp(group):
            rank = dist.get_rank(group)
            with no_grad():
                if random.random() < _RS_DROP:
                    fb = _last_good_rs.get(rank)
                    if fb is not None:
                        output.copy_(fb)
                else:
                    _last_good_rs[rank] = output.clone()
        return ret

    def faulty_all_gather_into_tensor(output, input, group=None, async_op=False):
        return dist.all_gather_into_tensor(output, input, group, async_op)

    # dist.reduce_scatter_tensor = faulty_reduce_scatter_tensor
    dist.all_gather_into_tensor = faulty_all_gather_into_tensor
