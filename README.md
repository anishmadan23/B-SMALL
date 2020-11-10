# B-SMALL: A BAYESIAN NEURAL NETWORK APPROACH TO SPARSE MAML
This repository contains code for B-SMALL [arxiv link coming soon...]
It includes PyTorch code to train and evaluate both MAML[1] and B-SMALL for few-shot classification tasks on CIFAR-FS and MiniImageNet. The models were trained on a single 2080Ti but use less memory than its full capacity.

### Dependencies
This code requires the following key dependencies:
- Pytorch 1.4
- Python 3.6
- Tensorboard

### Setup
Download MiniImagenet dataset and put the images in data/mini-imagenet/images/. The splits according to [1] are already provided in the repo. For CIFAR-FS, see data/get_cifarfs.py script to setup.

### Usage
#### Training 5-way 5-shot MAML from a checkpoint
```
python train.py --n_way 5 --k_spt 5 --restore_model <Path_to_model>
```
#### Training 5-way 5-shot B-SMALL 
```
python train.py --n_way 5 --k_spt 5 --svdo 
```
There are a number of parameters given as flags in the script which can be easily changed. 
### Key Results
We show some results obtained on the MiniImagenet dataset after training for 60,000 iterations and with the same hyperparameter settings as MAML[1]. Detailed results on CIFAR-FS and inferences in paper.
| Model/Experiment | 5-way 1-shot  | 5-way 5-shot |
|--------|----------------|----------|
| MAML[1] | 48.70 ± 1.84% | 63.11 ± 0.92% | 
| CAVIA[2] | 47.24 ± 0.65% | 61.87 ± 0.93% | 
MAML(Ours)| 46.30 ± 0.29%    |66.3 ± 0.21% |
**B-SMALL**| **49.12 ± 0.30%**  | **66.97 ± 0.3%** |
**Sparsity** | **76%** | **44%** |

### To do
- [ ] Sinusoid Regression Tasks

### Acknowledgement
Special thanks to Jackie Loong's implementation, of which some parts are directly taken for quick prototyping : ```https://github.com/dragen1860/MAML-Pytorch```

### References

[1] Finn, Chelsea, Pieter Abbeel, and Sergey Levine. "Model-agnostic meta-learning for fast adaptation of deep networks." *arXiv preprint arXiv:1703.03400* (2017).
[2] Zintgraf, Luisa, et al. "Fast context adaptation via meta-learning." International Conference on Machine Learning. PMLR, 2019.