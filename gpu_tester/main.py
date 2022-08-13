"""gpu tester"""

import os
import fire
import subprocess
import time


def is_job_finished(job_id):
    status = subprocess.check_output(["squeue", "-j", job_id]).decode("utf8")
    print(f"job status is {status}")
    return status == "slurm_load_jobs error: Invalid job id specified" or len(status.split("\n")) == 2


def wait_for_job_to_finish(job_id, timeout=30):
    t = time.time()
    while 1:
        if time.time() - t > timeout:
            return False
        time.sleep(1)
        if is_job_finished(job_id):
            return True


def start_job(sbatch_file, sbatch_args):
    sbatch_output = subprocess.check_output(
        ["sbatch"] + ([sbatch_args] if sbatch_args is not None else []) + [sbatch_file]
    ).decode("utf8")
    parsed_sbatch = sbatch_output.split(" ")
    if parsed_sbatch[0] != "Submitted":
        raise ValueError(f"slurm sbatch failed: {sbatch_output}")
    job_id = parsed_sbatch[3].strip()
    return job_id


def gpu_tester(
    cluster="slurm",
    job_name="gpu_tester",
    partition="compute-od-gpu",
    gpu_per_node=8,
    nodes=1,
    output_folder=None,
    job_timeout=300,
    sbatch_args=None,
):
    """gpu tester main function"""
    if cluster != "slurm":
        raise ValueError("only slurm is supported currently")
    if output_folder is None:
        output_folder = os.getcwd() + "/results"
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    tmp_file = output_folder + "/sbatch_output"
    sbatch_content = generate_sbatch(job_name, partition, nodes, gpu_per_node, tmp_file, sbatch_args)
    sbatch_file = output_folder + "/sbatch_file"
    with open(sbatch_file, "w", encoding="utf8") as f:
        f.write(sbatch_content)

    print("starting job")
    job_id = start_job(sbatch_file, sbatch_args)
    print(f"waiting for job {job_id}")
    status = wait_for_job_to_finish(job_id, job_timeout)
    if not status:
        print(f"canceling {job_id}")
        subprocess.check_output(["scancel", job_id]).decode("utf8")
        status = wait_for_job_to_finish(job_id)
        raise ValueError("job cancelled")
    print("job succeeded")

    with open(tmp_file, "r", encoding="utf8") as f:
        result_output = f.read()

    results = result_output.split("\n")

    error_gpu = [r for r in results if "gpu_error" in r]
    error_gpus = [r.split(" ") for r in error_gpu]

    real_results = [r for r in results if "result" in r]
    if len(real_results) == 0:
        raise ValueError(f"failed, output is {result_output}")

    parsed_results = [r.split(" ") for r in real_results]
    failed_wrong_results = [r for r in parsed_results if r[3] != "26.186691"]

    failed_count = len(failed_wrong_results)
    success_count = len(parsed_results) - failed_count

    print(f"{failed_count} have incorrect results, {len(error_gpus)} have gpu errors and {success_count} succeeded")

    print("incorrect results:")
    print(failed_wrong_results)

    print("gpu errors:")
    print(error_gpus)


def get_boilerplate():
    return """
module load intelmpi
source /opt/intel/mpi/latest/env/vars.sh
export LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib:/opt/amazon/efa/lib64:/usr/local/cuda-11.0/efa/lib:/usr/local/cuda-11.0/lib:/usr/local/cuda-11.0/lib64:/usr/local/cuda-11.0:/opt/nccl/build/lib:/opt/aws-ofi-nccl-install/lib:/opt/aws-ofi-nccl/lib:$LD_LIBRARY_PATH
export NCCL_PROTO=simple
export PATH=/opt/amazon/efa/bin:$PATH
export LD_PRELOAD="/opt/nccl/build/lib/libnccl.so"

export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn

export NCCL_DEBUG=info

export PYTHONFAULTHANDLER=1

export CUDA_LAUNCH_BLOCKING=0
export OMPI_MCA_mtl_base_verbose=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_TREE_THRESHOLD=0
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO

# sent to sub script
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
echo $HOSTNAMES
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

echo go $COUNT_NODE
echo $HOSTNAMES
"""


def generate_sbatch(job_name, partition, nodes, gpu_per_node, output_file, sbatch_args):
    ntasks_per_node = gpu_per_node
    gres = f"gpu:{gpu_per_node}"
    constant_boilerplate = get_boilerplate()
    venv = os.environ["VIRTUAL_ENV"]

    return f"""#!/bin/bash
#SBATCH --partition={partition}
#SBATCH --job-name={job_name}
#SBATCH --nodes {nodes}
#SBATCH --ntasks-per-node {ntasks_per_node}
#SBATCH --gres={gres}
#SBATCH --output={output_file}
#SBATCH --exclusive

{constant_boilerplate}

source {venv}/bin/activate


srun {sbatch_args if sbatch_args is not None else ""} --cpu_bind=v --accel-bind=gn python -m gpu_tester.worker
"""


def main():
    fire.Fire(gpu_tester)


if __name__ == "__main__":
    main()
