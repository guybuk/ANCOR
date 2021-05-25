





# Fine-grained Angular Contrastive Learning with Coarse Labels (CVPR 2021 Oral)
<a href="https://arxiv.org/abs/2012.03515"><img src="https://img.shields.io/badge/arXiv-2012.03515-b31b1b.svg"></a>
<a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache--2.0-yellow"></a>

> Few-shot learning methods offer pre-training techniques optimized for easier later adaptation of the model to new classes (unseen during training) using one or a few examples. This adaptivity to unseen classes is especially important for many practical applications where the pre-trained label space cannot remain fixed for effective use and the model needs to be "specialized" to support new categories on the fly. One particularly interesting scenario, essentially overlooked by the few-shot literature, is Coarse-to-Fine Few-Shot (C2FS), where the training classes (e.g. animals) are of much `coarser granularity' than the target (test) classes (e.g. breeds). A very practical example of C2FS is when the target classes are sub-classes of the training classes. Intuitively, it is especially challenging as (both regular and few-shot) supervised pre-training tends to learn to ignore intra-class variability which is essential for separating sub-classes. In this paper, we introduce a novel 'Angular normalization' module that allows to effectively combine supervised and self-supervised contrastive pre-training to approach the proposed C2FS task, demonstrating significant gains in a broad study over multiple baselines and datasets. We hope that this work will help to pave the way for future research on this new, challenging, and very practical topic of C2FS classification.


### Description

Official PyTorch implementation of "Fine-grained Angular Contrastive Learning with Coarse Labels"
### Installation
* Clone this repo:
```
git clone https://github.com/guybuk/ANCOR.git
cd ancor
```

* Install required packages:
```
pip install -r requirements.txt
```

### Data Preparation
#### BREEDS
1. Download the **ImageNet** dataset
2. Following the [official BREEDS repo](https://github.com/MadryLab/BREEDS-Benchmarks/blob/master/Constructing%20BREEDS%20datasets.ipynb), run:
```python
import os
from robustness.tools.breeds_helpers import setup_breeds
info_dir= "[your_imagenet_path]/ILSVRC/BREEDS"
if not (os.path.exists(info_dir) and len(os.listdir(info_dir))):
    print("Downloading class hierarchy information into `info_dir`")
    setup_breeds(info_dir)
```
3. Final folder structure should be:
```
└── ILSVRC
    ├── Annotations
    │   └── CLS-LOC
    ├── BREEDS
    │   ├── class_hierarchy.txt
    │   ├── dataset_class_info.json
    │   └── node_names.txt
    ├── Data
    │   └── CLS-LOC
    ├── ImageSets
    │   └── CLS-LOC
    └── Meta
        ├── imagenet_class_index.json
        ├── test.json
        ├── wordnet.is_a.txt
        └── words.txt
```
#### tieredImageNet
[[Google Drive](https://drive.google.com/open?id=1g1aIDy2Ar_MViF2gDXFYDBTR-HYecV07)]
* Taken from [Meta-Learning for Semi-Supervised Few-Shot Classification](https://github.com/renmengye/few-shot-ssl-public)
#### CIFAR-100
[[Direct Download](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)]

* Taken from [[the official webpage](https://www.cs.toronto.edu/~kriz/cifar.html)].

### Pretrained Models

| dataset           | model     | Epochs | 5-way acc.   | all-way acc. | download |
|----------------|:-----------:|:--------:|:--------------:|:--------------:|:----------:|
| LIVING-17      | ResNet-50 | 200    | 89.23 ± 0.55 | 45.14 ± 0.12 | [link](https://drive.google.com/file/d/1H_TOSsUVFW54oHyGd6qyT5hHiMdUbWnJ/view?usp=sharing)     |
| LIVING-17      | ResNet-50 | 800    | 92.59 ± 0.47 | 58.15 ± 0.16 | [link](https://drive.google.com/file/d/1DmXOLKp9xHO9fda6VTBWajyOKbuYisQ8/view?usp=sharing)     |
| NONLIVING-26   | ResNet-50 | 200    | 86.23 ± 0.54 | 43.10 ± 0.11 | [link](https://drive.google.com/file/d/1zSuVUE0oM8COYCsDzJMcs-xlpMy3upz8/view?usp=sharing)     |
| NONLIVING-26   | ResNet-50 | 800    | 88.25 ± 0.52 | 49.38 ± 0.13 | [link](https://drive.google.com/file/d/1loj3nren0sVm-RQfRDxStRUhR-UMBIJ0/view?usp=sharing)     |
| ENTITY-13      | ResNet-50 | 200    | 90.58 ± 0.54 | 42.29 ± 0.08 | [link](https://drive.google.com/file/d/10M8VqlCoCI7xBR9TOooAqAR-_haGR0wf/view?usp=sharing)     |
| ENTITY-13      | ResNet-50 | 800    | 92.04 ± 0.44 | 50.72 ± 0.09 | [link](https://drive.google.com/file/d/1ccuSrYf2XFBNJUhHHtMshy9ZvHMbUn9y/view?usp=sharing)     |
| ENTITY-30      | ResNet-50 | 200    | 88.12 ± 0.54 | 41.79 ± 0.08 | [link](https://drive.google.com/file/d/1yseiuwZVDoowuG5WqNiS0iVP991wc9HY/view?usp=sharing)     |
| ENTITY-30      | ResNet-50 | 800    | 92.13 ± 0.44 | 50.85 ± 0.09 | [link](https://drive.google.com/file/d/1kSK9Py802W7ykA3LPY49Z16eBTE2QEqP/view?usp=sharing)     |
| tieredImageNet | ResNet-12 | 200    | 63.54 ± 0.70 | 11.97 ± 0.06 | [link](https://drive.google.com/file/d/1H_TOSsUVFW54oHyGd6qyT5hHiMdUbWnJ/view?usp=sharing)     |
| CIFAR-100      | ResNet-12 | 200    | 74.56 ± 0.70 | 29.84 ± 0.11 | [link](https://drive.google.com/file/d/1x0Gv4U8-TAG8W-g72YmnH0GiSvJ_inlZ/view?usp=sharing)     |

### Training

This implementation only supports **multi-gpu**, **DistributedDataParallel** training. Single-gpu training runs without failure, but
was only used for debugging and we do not vouch for the outputs.

To train ANCOR on living17 using ResNet50 like the paper run:

```
python train.py \
--batch-size 256 -j 24 \
--dist-url tcp://localhost:2375 --world-size 1 --rank 0 --multiprocessing-distributed \
-p 1 -s 5 \
--cos --mlp \
--dataset living17 --mode coarse \
--data /nfs/datasets/classification/imagenet/ILSVRC
```

**NOTES**:
* Default parameters are set for BREEDS training
* Parameters tuned for 4 Tesla V-100 GPUs

### Evaluation

Given a pre-trained model, run few-shot evaluation (5-way followed by all-way) using the following:
```
python eval.py \
--model resnet50 \
--num_workers 10 \
--model_path [path to trained model] \
--dataset living17 \
--data_root [path to data] \
--mode fine \
--head seq \
--partition validation \
--only-base
```
**NOTES:** 
* Add `--fg` flag if you want the second evaluation will be intra-class rather than all-way.
* Pretrained CIFAR-100 and tieredImageNet models require the argument: `--model resnet12`

#### Ensemble 

Given pre-trained models, run a few-shot evaluation that averages the 
probabilities of two models by executing the following:
```
python ensemble.py \
--model resnet50 resnet50 \
--num_workers 20 \
--model_path [list of paths to models...] \
--dataset living17 \
--data_root [path to data] \
--mode fine \
--head cls \
--partition validation \
--only-base
```
**NOTE:** add `--fg` flag if you want the second evaluation will be intra-class rather than all-way.

#### Feature-concatenation Model
Given pre-trained models, run a few-shot evaluation that concatenates the features
produces by the two models by executing the following:

```
python concat.py \
--model resnet50 resnet50 \
--num_workers 20 \
--model_path [list of paths to models...] \
--dataset living17 \
--data_root [path to data] \
--mode fine \
--head cls \
--partition validation \
--only-base
```
**NOTE:** add `--fg` flag if you want the second evaluation will be intra-class rather than all-way.

#### Cascade Model 
Given two pre-trained models, run a few-shot evaluation that gives coarse scores
using the first model, and then does intra-class fine classification inside the max scoring coarse class.
```
python cascade.py \
--model resnet50 resnet50 \
--num_workers 20 \
--model_path [list of paths to models...] \
--dataset living17 \
--data_root [path to data] \
--mode fine \
--head cls \
--partition validation \
--only-base
```

### T-SNE Visualization
To create T-SNE visualizations for coarse and fine classes, run the following:
```
python tsne.py \
--model resnet50 \
--num_workers 0 \
--model_path [path to model] \
--dataset living17 \
--data_root [path to data] \
--mode fine \
--head seq \
--partition validation \
--n_ways 1000
```

### License
Copyright 2019 IBM Corp. This repository is released under the Apachi-2.0 license (see the LICENSE file for details)

### References
[1] Guy Bukchin, Eli Schwartz, Kate Saenko, Ori Shahar, Rogerio Feris, Raja Giryes, Leonid Karlinsky, Fine-grained Angular Contrastive Learning with Coarse Labels
. Accepted to CVPR 2021 (Oral). https://arxiv.org/abs/2012.03515