"""worker running in each gpu"""

import torch
import os
import socket
import time

torch.manual_seed(0)


def main():
    local_rank = 0
    for v in ("LOCAL_RANK", "MPI_LOCALRANKID", "SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK"):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break

    hostname = socket.gethostname()
    try:
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

        vector1 = torch.rand(1, 100000, device=device)
        vector2 = torch.rand(1, 100000, device=device)
        t = time.time()
        dot = (vector1 @ vector2.T).cpu().numpy()
        d = time.time() - t

        print("result", hostname, local_rank, dot[0][0], d)
    except RuntimeError as _:
        print("gpu_error", hostname, local_rank)


if __name__ == "__main__":
    main()
