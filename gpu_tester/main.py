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


def start_job(sbatch_file):
    """start job"""
    args = ["sbatch"]
    args.append(sbatch_file)
    sbatch_output = subprocess.check_output(args).decode("utf8")
    lines = sbatch_output.split("\n")

    lines = [line for line in lines if "Submitted" in line]
    if len(lines) == 0:
        raise ValueError(f"slurm sbatch failed: {sbatch_output}")

    parsed_sbatch = lines[0].split(" ")
    job_id = parsed_sbatch[3].strip()
    return job_id


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
echo hosts $HOSTNAMES
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

echo go $COUNT_NODE
"""


def generate_sbatch(job_name, partition, nodes, gpu_per_node, output_file, job_comment, test_kind):
    ntasks_per_node = gpu_per_node
    gres = f"gpu:{gpu_per_node}"
    constant_boilerplate = get_boilerplate()
    venv = os.environ["VIRTUAL_ENV"]
    scomment = ("--comment " + job_comment) if job_comment is not None else ""
    sbatch_scomment = ("#SBATCH --comment " + job_comment) if job_comment is not None else ""
    worker = test_kind + "_worker"

    return f"""#!/bin/bash
#SBATCH --partition={partition}
#SBATCH --job-name={job_name}
#SBATCH --nodes {nodes}
#SBATCH --ntasks-per-node {ntasks_per_node}
#SBATCH --gres={gres}
#SBATCH --output={output_file}
#SBATCH --exclusive
{sbatch_scomment}

{constant_boilerplate}

source {venv}/bin/activate


srun {scomment} --cpu_bind=v --accel-bind=gn python -m gpu_tester.{worker}
"""


def gpu_tester(
    cluster="slurm",
    job_name="gpu_tester",
    partition="compute-od-gpu",
    gpu_per_node=8,
    nodes=1,
    output_folder=None,
    job_timeout=150,
    job_comment=None,
    test_kind="simple_forward",
):
    """gpu tester main function"""
    if cluster != "slurm":
        raise ValueError("only slurm is supported currently")
    if output_folder is None:
        output_folder = os.getcwd() + "/results"
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    tmp_file = output_folder + "/sbatch_output"
    sbatch_content = generate_sbatch(job_name, partition, nodes, gpu_per_node, tmp_file, job_comment, test_kind)
    sbatch_file = output_folder + "/sbatch_file"
    with open(sbatch_file, "w", encoding="utf8") as f:
        f.write(sbatch_content)

    print("starting job")
    job_id = start_job(sbatch_file)
    print(f"waiting for job {job_id}")
    status = wait_for_job_to_finish(job_id, job_timeout)
    if not status:
        print(f"canceling {job_id}")
        subprocess.check_output(["scancel", job_id]).decode("utf8")
        status = wait_for_job_to_finish(job_id)
        print("job cancelled")
    else:
        print("job succeeded")

    with open(tmp_file, "r", encoding="utf8") as f:
        result_output = f.read()

    results = result_output.split("\n")
    hosts = [h for h in results if "hosts" in h]
    if len(hosts) == 0:
        raise ValueError("failed" + result_output)
    hosts = hosts[0].split(" ")[1:]
    hosts_gpus = [h + " " + str(gpu) for gpu in range(8) for h in hosts]

    error_gpu = [r for r in results if "gpu_error" in r]
    error_gpus = [r.split(" ") for r in error_gpu]

    real_results = [r for r in results if "result" in r]
    if len(real_results) == 0:
        raise ValueError(f"failed, output is {result_output}")

    parsed_results = [r.split(" ") for r in real_results]

    if test_kind == "simple_forward":
        expected_value = "24954.1"
        expected_delay = 5
    elif test_kind == "ddp":
        expected_value = "1.3285987"
        expected_delay = 5
    failed_wrong_results = [r for r in parsed_results if abs(float(r[3]) - float(expected_value)) > 0.01]
    slow_results = [r for r in parsed_results if float(r[4]) > expected_delay]

    failed_count = len(failed_wrong_results)

    answered_host_gpu = set([e[1] + " " + e[2] for e in error_gpu] + [e[1] + " " + e[2] for e in parsed_results])

    no_answer = list(set(hosts_gpus) - answered_host_gpu)
    success_count = len(parsed_results) - failed_count - len(no_answer) - len(slow_results)

    print(
        f"""
        * {failed_count} have incorrect results
        * {len(slow_results)} have slow results
        * {len(error_gpus)} have gpu errors
        * {len(no_answer)} did not answer
        * {success_count} succeeded
            """
    )

    print("slow results:")
    print(slow_results)

    print("incorrect results:")
    print(failed_wrong_results)

    print("gpu errors:")
    print(error_gpus)

    print("no_answer:")
    print(no_answer)


def main():
    fire.Fire(gpu_tester)


if __name__ == "__main__":
    main()
