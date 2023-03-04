<h1 align="center">Airport Ground Handling</h1>

This repository contains the implementations of our papers studying *Airport Ground Handling (AGH)* problems:

* **Learning Large Neighborhood Search for Vehicle Routing in Airport Ground Handling**

  <p align="center">
  	<a href="https://royalskye.github.io">Jianan Zhou</a>, <a href="https://wxy1427.github.io">Yaoxin Wu</a>, <a href="https://zhiguangcaosg.github.io">Zhiguang Cao</a>, <a href="https://songwenas12.github.io">Wen Song</a>, <a href="https://personal.ntu.edu.sg/zhangj">Jie Zhang</a>, <a href="https://zhenghuantu.github.io">Zhenghua Chen</a></p>

  <p align="center">
      <a href="https://arxiv.org/abs/2302.13797"><img src="https://img.shields.io/badge/Download-PDF-green" alt="Paper"></a>
      <a href="https://ieeexplore.ieee.org/document/10054476"><img alt="License" src="https://img.shields.io/static/v1?label=Pub&message=IEEE TKDE&color=red"></a>
    	<a href="https://github.com/RoyalSkye/AGH/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue" alt="Paper"></a>
  </p>

* **Neural Airport Ground Handling**

  <p align="center">
  	<a href="https://wxy1427.github.io">Yaoxin Wu*</a>, <a href="https://royalskye.github.io">Jianan Zhou*</a>, Yunwen Xia, Xianli Zhang, <a href="https://zhiguangcaosg.github.io">Zhiguang Cao</a>, <a href="https://personal.ntu.edu.sg/zhangj">Jie Zhang</a></p>
  
  <p align="center">
      <a href=""><img src="https://img.shields.io/badge/Download-PDF-green" alt="Paper"></a>
      <a href=""><img alt="License" src="https://img.shields.io/static/v1?label=Pub&message=IEEE TITS&color=red"></a>
    	<a href="https://github.com/RoyalSkye/AGH/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue" alt="Paper"></a>
  </p>

*Note: These works were done in 2021 and 2022, respectively. Based on our experiments, we recommend building upon the code of the second work for further research.*

## Overview

### Learning Large Neighborhood Search for Vehicle Routing in Airport Ground Handling

We propose a learning-based <u>improvement framework</u> to solve large-scale *Airport Ground Handling (AGH)* instances. Specifically, we leverage the Large Neighborhood Search (LNS) framework, which consists of a pair of *destroy* and *repair* operators, to decompose the global (intractable) optimization problem and re-optimize each sub-problem. The operation scheduling in AGH is formulated as a mixed integer linear programming (MILP) model. To mitigate the need of domain expertise, 1) our proposed framework directly operates on the decision variables of the MILP model; 2) we employ an off-the-shelf solver (e.g., CPLEX) as the repair operator to conduct re-optimization. Our method could efficiently solve large-scale AGH instances with hundreds of flights, while CPLEX would simply stuck, even when searching for a feasible solution.

<p align="center">
  <img src="./imgs/Overview_LNS.png" width=90% alt="framework"/>
</p>

### Neural Airport Ground Handling

We propose a learning-based <u>construction framework</u> to solve *Airport Ground Handling (AGH)* problems in an end-to-end manner. The studied problem is modeled as a multiple-fleet vehicle routing problem (VRP) with miscellaneous constraints, including precedence, time windows, and capacity. It is much more complicated than the simple VRPs (e.g., TSP/CVRP) studied in the major ML conferences. The proposed method could also serve as a simple learning-based baseline for further research on complicated VRPs (e.g., CVRPTW).

<p align="center">
  <img src="./imgs/Overview.png" width=90% alt="framework"/>
</p>

## File Structure

```shell
./
├── Construction_based               # The implementation of TITS paper
├── Improvement_based                # The implementation of TKDE paper
├── LICENSE
├── README.md
└── imgs
```

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

```bibtex
@article{wu2023neural,
title       = {Neural Airport Ground Handling},
author      = {Yaoxin Wu and Jianan Zhou and Yunwen Xia and Xianli Zhang and Zhiguang Cao and Jie Zhang},
journal     = {IEEE Transactions on Intelligent Transportation Systems},
year        = {2023},
publisher   = {IEEE}
}
```