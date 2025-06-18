
# R2C: Mapping Room to Chessboard to Unlock LLM As Low-Level Action Planner

This repo contains source code for our CVPR 2025 paper:

[R2C: Mapping Room to Chessboard to Unlock LLM As Low-Level Action Planner](https://vipl-vsu.github.io/Room2Chessboard/)

[Ziyi Bai](https://scholar.google.com/citations?&user=jRe11usAAAAJ), [Hanxuan Li](https://scholar.google.com/citations?user=oiv8QCoAAAAJ), [Bin Fu](https://scholar.google.com/citations?&user=OljzYgQAAAAJ), [Chuyan Xiong](https://scholar.google.com/citations?&user=PzKTeZYAAAAJ), [Ruiping Wang](https://scholar.google.com/citations?&user=duIUwpwAAAAJ), [Xilin Chen](https://scholar.google.com/citations?user=vVx2v20AAAAJ)

## Prerequisites

1. Clone the repository:

```bash
$ git clone https://github.com/ByZ0e/R2C
$ export ROOT=$(pwd)/R2C
$ export LOGS=$ROOT/logs
$ export DATA=$ROOT/data
$ export PYTHONPATH=$PYTHONPATH:$ROOT
```

2. Install the required dependencies:

```bash
$ conda create -n r2c_env python=3.10
$ conda activate r2c_env

$ cd $ROOT
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

3. Download the ALFRED dataset:

```bash
$ cd $DATA
$ sh download_data.sh json_feat
```

4. Render PNG images:

```bash
$ python -m alfred.gen.render_trajs
```

## Evaluation

With your environment set up and data ready, you can start evaluating the model:

```bash
$ cd eval_mp
$ python eval_llm_demo.py
```

