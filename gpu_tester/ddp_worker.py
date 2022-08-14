"""ddp worker"""

import torch
import torch.distributed as dist
from torch import nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import socket
import time
from .world_info_from_env import world_info_from_env

torch.manual_seed(0)


def main():
    """example"""
    local_rank, global_rank, world_size = world_info_from_env()
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["RANK"] = str(global_rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    hostname = socket.gethostname()
    try:
        # create default process group
        dist.init_process_group("nccl", rank=global_rank, world_size=world_size)
        # create local model
        model = nn.Linear(1000, 1000).to(local_rank)
        # construct DDP model
        ddp_model = DDP(model, device_ids=[local_rank])
        # define loss function and optimizer
        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

        inputs = torch.randn(200, 1000).to(local_rank)
        labels = torch.randn(200, 1000).to(local_rank)
        # warmup
        for _ in range(10):
            outputs = ddp_model(inputs)
            loss_fn(outputs, labels).backward()
            optimizer.step()
        # measure
        t = time.time()
        for _ in range(1000):
            outputs = ddp_model(inputs)
            loss_fn(outputs, labels).backward()
            optimizer.step()
        d = time.time() - t
        print("result", hostname, local_rank, outputs.detach().cpu().numpy()[0][0], d)
    except RuntimeError as err:
        print("gpu_error", hostname, local_rank)
        print(err)


if __name__ == "__main__":
    main()
