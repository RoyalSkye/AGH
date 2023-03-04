<h1 align="center">Neural Airport Ground Handling</h1>

<p align="center">
    <a href=""><img src="https://img.shields.io/badge/Download-PDF-green" alt="Paper"></a>
    <a href=""><img alt="License" src="https://img.shields.io/static/v1?label=Pub&message=IEEE TITS&color=red"></a>
    <a href="https://github.com/RoyalSkye/AGH/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue" alt="Paper"></a>
</p>

The implementation of *IEEE Transactions on Intelligent Transportation Systems (TITS) - "Neural Airport Ground Handling"* by [Yaoxin Wu*](https://wxy1427.github.io), [Jianan Zhou*](https://royalskye.github.io), Yunwen Xia, Xianli Zhang, [Zhiguang Cao](https://zhiguangcaosg.github.io), [Jie Zhang](https://personal.ntu.edu.sg/zhangj).

## Overview

We propose a learning-based construction framework to solve *Airport Ground Handling (AGH)* problems in an end-to-end manner. The studied problem is modeled as a multiple-fleet vehicle routing problem (VRP) with miscellaneous constraints, including precedence, time windows, and capacity. It is much more complicated than the simple VRPs (e.g., TSP/CVRP) studied in the major ML conferences. The proposed method could also serve as a simple learning-based baseline for further research on complicated VRPs (e.g., CVRPTW).

<p align="center">
  <img src="../imgs/Overview.png" width=90% alt="framework"/>
</p>

## Dependencies

See environment.yml for more details.

* Python >= 3.8
* Pytorch >= 1.7
* NumPy
* tensorboard_logger

## How to Run

### Train

See [args.json](https://github.com/RoyalSkye/attention-agh/blob/main/data/20/args.json) for the detailed setting.

```shell
python run.py --problem "agh" --graph_size 20 --baseline "rollout"
```

### Evaluation

```shell
# eval with greedy (am-greedy)
python eval.py --eval_batch_size 1000 --decode_strategy greedy --datasets ./data/agh/agh20_validation_seed4321.pkl --model data/20 --val_size 1000

# eval with sampling (am-sample-1000)
python eval.py --eval_batch_size 1 --width 1000 --decode_strategy sample --datasets ./data/agh/agh20_validation_seed4321.pkl --model data/20 --val_size 1000
```

### Baseline

```shell
# ['nearest_insert', 'farthest_insert', 'random_insert', 'nearest_neighbor', 'sa', 'cws']
# --multiprocess is supported for three insertion methods.
python agh_baseline.py --filename ./data/agh/agh20_validation_seed4321.pkl --graph_size 20 --val_method "cws" --val_size 1000

# ['cplex', 'lns', 'lns_sa']
python cplex_lns.py --filename ./data/agh/agh20_validation_seed4321.pkl --graph_size 20 --val_method "cplex" --val_size 1000
```

## Reference

* [ICLR 2019] - [Attention, Learn to Solve Routing Problems!](https://openreview.net/forum?id=ByxBFsRqYm)

## Acknowledgments

Thank the following repositories, which are baselines of our code:

* https://github.com/wouterkool/attention-learn-to-route

## Citation

```bibtex
@article{wu2023neural,
title       = {Neural Airport Ground Handling},
author      = {Yaoxin Wu and Jianan Zhou and Yunwen Xia and Xianli Zhang and Zhiguang Cao and Jie Zhang},
journal     = {IEEE Transactions on Intelligent Transportation Systems},
year        = {2023},
publisher   = {IEEE}
}
```

