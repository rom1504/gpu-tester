"""world info from env"""

import os


def world_info_from_env():
    """world info from env"""
    local_rank = 0
    for v in ("LOCAL_RANK", "MPI_LOCALRANKID", "SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK"):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ("RANK", "PMI_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ("WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size
