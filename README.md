# Generating Question Relevant Captions for VQA

The repository contains training code for generation question relevant captions for VQA task.
Approach based on [Generating Question Relevant Captions to Aid Visual Question Answering](https://www.aclweb.org/anthology/P19-1348.pdf) paper and [repo](https://github.com/jialinwu17/caption_vqa)

## Prerequisites
* Python3.5+
* CUDA 10.1

## Installation
Python setuptools and python package manager (pip) install packages into system directory by default.  The training code tested only via [virtual environment](https://docs.python.org/3/tutorial/venv.html).

In order to use virtual environment you should install it first:

```bash
python3 -m pip install virtualenv
python3 -m virtualenv -p `which python3` <directory_for_environment>
```

Before starting to work inside virtual environment, it should be activated:

```bash
source <directory_for_environment>/bin/activate
```

Virtual environment can be deactivated using command

```bash
deactivate
```
Install dependencies with python package meneger:
```bash
pip install -r requirements.txt
```

## Data preparation
See [data preparation instruction](./data/README.md)

## Training
You are able to run training using following command
```bash
python main.py --config configs/caption_vqa.yml --device_ids <gpu_ids>
```
## Test on VQA2.0
TBD
