# attention-agh
Attention based model for learning to solve Airport Ground Handling (AGH) problems.

## Dependencies

* Python >= 3.8
* Pytorch >= 1.7
* NumPy
* SciPy
* tqdm
* tensorboard_logger
* Matplotlib

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

## Acknowledgements

Thanks the following repositories, which are baselines of our code:

* https://github.com/wouterkool/attention-learn-to-route

