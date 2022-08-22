# gpu_tester
[![pypi](https://img.shields.io/pypi/v/gpu_tester.svg)](https://pypi.python.org/pypi/gpu_tester)

Gpu tester finds all your bad gpus.

Works on slurm.

Features:
* does a forward on each gpu
* check for gpu returning incorrect results
* check for gpu failing due to ECC errors

Roadmap:
* sanity check forward speed
* sanity check broadcast speed

## Install

pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu116

then 

pip install gpu_tester

## Python examples

Checkout these examples to call this as a lib:
* [example.py](examples/example.py)

## Output

Output looks like this:

```
job succeeded
0 have incorrect results, 1 have gpu errors and 319 succeeded
incorrect results:
[]
gpu errors:
[['gpu_error', 'compute-od-gpu-st-p4d-24xlarge-156', '3']]
```

## Recommended testing strategy

### Pair based strategy

The easiest way to quickly spot broken node is to do the pair-based strategy.
It will run many jobs in parallel and find which node can talk together
Here is one example
```
gpu_tester --nodes 2 --parallel-tests 50 --job_comment laion --partition "gpu" --test_kind "ddp" --job_timeout 45 --exclude 'gpu-st-p4d-24xlarge-[66]'
```

### All at once strategy

Once you validated this works, you may want to try the DDP strategy over all nodes, eg:
```
gpu_tester --nodes 100 --parallel-tests 1 --job_comment laion --partition "gpu" --test_kind "ddp" --job_timeout 300 --exclude 'gpu-st-p4d-24xlarge-[66]'
```

### Simple forward

If you want to only validate the forward functionality of gpus and not the communication, you may use:

```
gpu_tester --nodes 100 --parallel-tests 1 --job_comment laion --partition "gpu" --test_kind "simple_forward" --job_timeout 50 --exclude 'gpu-st-p4d-24xlarge-[66]'
```


## API

This module exposes a single function `gpu_tester` which takes the same arguments as the command line tool:

* **cluster** the cluster. (default *slurm*)
* **job_name** slurm job name. (default *gpu_tester*)
* **partition** slurm partition. (default *compute-od-gpu*)
* **gpu_per_node** numbe of gpu per node. (default *8*)
* **nodes** number of gpu nodes. (default *1*)
* **output_folder** the output folder. (default *None* which means current folder / results)
* **job_timeout** job timeout (default *150* seconds)
* **job_comment** optional comment arg given to slurm (default *None*)
* **test_kind** simple_forward or ddp. simple_forward is quick forward test. DDP uses pytorch ddp to check gpu interconnect (default *simple_forward*)
* **parallel_tests** number of tests to run in parallel. Recommended to use that with nodes == 2 to test pair by pair (default *1*)
* **nodelist** node whitelist, example 'gpu-st-p4d-24xlarge-[66-67]' (default *None*)
* **exclude** node blacklist, example 'gpu-st-p4d-24xlarge-[66-67]' (default *None*)

## For development

Either locally, or in [gitpod](https://gitpod.io/#https://github.com/rom1504/gpu_tester) (do `export PIP_USER=false` there)

Setup a virtualenv:

```
python3 -m venv .env
source .env/bin/activate
pip install -e .
```

to run tests:
```
pip install -r requirements-test.txt
```
then 
```
make lint
make test
```

You can use `make black` to reformat the code

`python -m pytest -x -s -v tests -k "dummy"` to run a specific test
