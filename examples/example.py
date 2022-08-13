from gpu_tester import gpu_tester

# this command will start a test on these gpu
gpu_tester(
    cluster="slurm",
    gpu_nodes="compute-od-gpu-st-p4d-24xlarge-[10-20]",
    partition="compute-od-gpu",
    gpu_per_node=8,
    nodes=11,
    output_folder=None,
)
