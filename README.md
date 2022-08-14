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
