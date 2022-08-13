# gpu_tester
[![pypi](https://img.shields.io/pypi/v/gpu_tester.svg)](https://pypi.python.org/pypi/gpu_tester)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rom1504/gpu_tester/blob/master/notebook/gpu_tester_getting_started.ipynb)
[![Try it on gitpod](https://img.shields.io/badge/try-on%20gitpod-brightgreen.svg)](https://gitpod.io/#https://github.com/rom1504/gpu_tester)

Gpu tester finds all your bad gpus.

Works on slurm.

## Install

pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu116

then 

pip install gpu_tester

## Python examples

Checkout these examples to call this as a lib:
* [example.py](examples/example.py)

## API

This module exposes a single function `gpu_tester` which takes the same arguments as the command line tool:

* **cluster** the cluster. (default *slurm*)
* **job_name** slurm job name. (default *gpu_tester*)
* **partition** slurm partition. (default *compute-od-gpu*)
* **gpu_per_node** numbe of gpu per node. (default *8*)
* **nodes** number of gpu nodes. (default *1*)
* **output_folder** the output folder. (default *None* which means current folder / results)
* **job_timeout** job timeout (default *300* seconds)

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
