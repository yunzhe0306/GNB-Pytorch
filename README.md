# GNB-Pytorch

This is an implementation of the paper "Graph Neural Bandits" in KDD 2023.

For the datasets, please download "dataset.zip" from [here](https://drive.google.com/file/d/19fEwrCBaoB_-vaMivRODWRbmERnu7ZXK/view?usp=sharing), and place "data/" and "processed_data/" folders to the root directory.

We use ```Run_benchmark_algos_multi_RUNS.py``` to run the experiments with baselines, and use ```User_GNN_Run.py``` to run the experiments for GNB. ```Parameters_Profile.py``` refers to the hyper-parameters of the GNB experiments.

```
@inproceedings{qi2023graph,
  title={Graph Neural Bandits},
  author={Qi, Yunzhe and Ban, Yikun and He, Jingrui},
  booktitle={Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={1920--1931},
  year={2023}
}
```




