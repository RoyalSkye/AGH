<h2 align="center">Learning Large Neighborhood Search for Vehicle Routing in Airport Ground Handling</h2>

<p align="center">
    <a href="https://arxiv.org/abs/2302.13797"><img src="https://img.shields.io/badge/Download-PDF-green" alt="Paper"></a>
    <a href="https://ieeexplore.ieee.org/document/10054476"><img alt="License" src="https://img.shields.io/static/v1?label=Pub&message=IEEE TKDE&color=red"></a>
    <a href="https://github.com/RoyalSkye/AGH/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue" alt="Paper"></a>
</p>

The implementation of *IEEE Transactions on Knowledge and Data Engineering (TKDE) - "Learning Large Neighborhood Search for Vehicle Routing in Airport Ground Handling"* by [Jianan Zhou](https://royalskye.github.io), [Yaoxin Wu](https://wxy1427.github.io), [Zhiguang Cao](https://zhiguangcaosg.github.io), [Wen Song](https://songwenas12.github.io), [Jie Zhang](https://personal.ntu.edu.sg/zhangj), [Zhenghua Chen](https://zhenghuantu.github.io).

## Overview

We propose a learning-based improvement framework to solve large-scale *Airport Ground Handling (AGH)* instances. Specifically, we leverage the Large Neighborhood Search (LNS) framework, which consists of a pair of *destroy* and *repair* operators, to decompose the global (intractable) optimization problem and re-optimize each sub-problem. The operation scheduling in AGH is formulated as a mixed integer linear programming (MILP) model. To mitigate the need of domain expertise, 1) our proposed framework directly operates on the decision variables of the MILP model; 2) we employ an off-the-shelf solver (e.g., CPLEX) as the repair operator to conduct re-optimization. Our method could efficiently solve large-scale AGH instances with hundreds of flights, while CPLEX would simply stuck, even when searching for a feasible solution.

<p align="center">
  <img src="../imgs/Overview_LNS.png" width=90% alt="framework"/>
</p>

## Dependencies

See environment.yml for more details.

* Python >= 3.7
* Pytorch >= 1.5
* docplex = 2.12.182
* Cplex
* scikit-learn
* NumPy

## File Structure

The implementations of our method include `CPLEX` and `OR-Tools` versions.

One baseline implementation is `decomposition_cplex`.

```shell
./cplex
├── alns_cvrptw.py                   # The Implementations of MILP modeling and LNS methods
├── data                             # Data example
│   ├── 20
│   │   ├── SHARED_DATA_1.json
│   │   ├── distance_1.pickle
│   │   └── schedule_1.xlsx
│   └── 20_Test.zip
├── datasets.py
├── eval.py                          # Evaluation
├── eval_generate_json.py
├── forward_training.py              # The Implementation of forward training algorithm
├── generate_data.py
├── models.py                        # GCN model
├── projectSolver.py                 # Start from here, get an initial sol -> run LNS
├── test_200304                      # Files of data and env settings
│   ├── Changi_Background.png
│   ├── airport.net.xml
│   ├── operation_W.xlsx
│   ├── schedule_100.xlsx
│   ├── schedule_20.xlsx
│   ├── schedule_200.xlsx
│   ├── schedule_300.xlsx
│   ├── schedule_50.xlsx
│   └── testPrj.cfg
├── train.py                         # Start training
└── utils.py
```

## Reference

* [CP 1998] - [Using constraint programming and local search methods to solve vehicle routing problems](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=8ecbdd4b849cb2b023d4fc0da3e328d451e6f978)
* [Computers & Operations Research, 2016] - [A bi-objective approach for scheduling ground-handling vehicles in airports](https://hal-enac.archives-ouvertes.fr/hal-01819078/file/Padron_et_al_COR%5B1%5D.pdf)
* [NeurIPS 2020] - [A General Large Neighborhood Search Framework for Solving Integer Linear Programs](https://arxiv.org/pdf/2004.00422.pdf)
* [NeurIPS 2021] - [Learning Large Neighborhood Search Policy for Integer Programming](https://openreview.net/forum?id=IaM7U4J-w3c)

## Citation

```bibtex
@article{zhou2023learning,
title       = {Learning Large Neighborhood Search for Vehicle Routing in Airport Ground Handling},
author      = {Jianan Zhou and Yaoxin Wu and Zhiguang Cao and Wen Song and Jie Zhang and Zhenghua Chen},
journal     = {IEEE Transactions on Knowledge and Data Engineering},
year        = {2023},
publisher   = {IEEE}
}
```
