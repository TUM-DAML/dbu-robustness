# Evaluating Robustness of Predictive Uncertainty Estimation: Are Dirichlet-based Models Reliable ?

This repository presents the experiments of the paper:

[Evaluating Robustness of Predictive Uncertainty Estimation: Are Dirichlet-based Models Reliable ?](https://arxiv.org/pdf/2010.14986.pdf)<br>
Anna-Kathrin Kopetzki*, Bertrand Charpentier*, Daniel Zügner, Sandhya Giri, Stephan Günnemann<br>
International Conference on Machine Learning (ICML), 2021.

[[Paper](https://arxiv.org/pdf/2010.14986.pdf)]

![Diagram](uncertainty-diagram.png?raw=true "Diagram")

## Requirements

To install requirements:

```setup
conda env create -f environment.yaml
conda activate dbu-robustness
conda env list

python src/setup.py develop
python setup.py develop
```

Not that our code is based on  the following papers and repositories:
* **Foolbox**: [Foolbox Native: Fast adversarial attacks to benchmark the robustness of machine learning models in PyTorch, TensorFlow, and JAX](https://github.com/bethgelab/foolbox)
* **Median Smoothing**: [Detection as Regression: Certified Object Detection by Median Smoothing](https://arxiv.org/abs/2007.03730)
* **PostNet**: [Posterior Network: Uncertainty Estimation without OOD Samples via Density-Based Pseudo-Counts](https://arxiv.org/abs/2006.09239)

## Datasets

**MNIST** and **CIFAR10** are handled with torchvision. You can download preprocessed **segment** and **sensorless-drive** datasets at the follwing links:
- [Segment (No sky, no window)](https://ln4.sync.com/dl/b11f03d60#4xvfj3iu-bbn47996-6kew5cs4-ypsnx3mc), [Segment (Sky only)](https://ln4.sync.com/dl/19d2faf60#pxg8uctx-4ne438bp-sygfijqp-vj5ddb3f), [Segment (Window only)](https://ln4.sync.com/dl/1b11ce520#npynj7em-fkvkup45-rstwcfkx-tuubuzs9)
- [SensorlessDrive (No 9, 10, 11)](https://ln4.sync.com/dl/a1e84d690#4hpe866n-ffpdv5uq-t7qmvjs3-usznuiid), [SensorlessDrive (9 only)](https://ln4.sync.com/dl/2c1741440#cerfgh9j-phzc6kav-jwx9yici-qxnfw64q), [SensorlessDrive (10, 11 only)](https://ln4.sync.com/dl/df42edea0#mpiqq4wv-scd56vif-3kec8si3-9uchrttd)

## Training models

Note that our code implements the following **Dirichlet-based Uncertainty (DBU)** models:
* **PostNet**: [Posterior Network: Uncertainty Estimation without OOD Samples via Density-Based Pseudo-Counts](https://arxiv.org/abs/2006.09239)
* **PriorNet**: [Predictive Uncertainty Estimation via Prior Networks](https://arxiv.org/abs/1802.10501) and [Reverse KL-Divergence Training of Prior Networks: Improved Uncertainty and Adversarial Robustness](https://arxiv.org/abs/1905.13472)
* **EvNet**: [Evidential Deep Learning to Quantify Classification Uncertainty](https://arxiv.org/abs/1806.01768)
* **DDnet**: [Ensemble Distribution Distillation](https://arxiv.org/abs/1905.00076)

To train the models in the paper, run one jupyter notebook in the folder `notebooks/models-training`. Further, you can find pre-trained models with standard training at [this link](https://ln4.sync.com/dl/d8ca79940#4du2jdvf-48nw6s55-9u8a8xdt-zprm4vwj) which could be placed in the folder `notebooks/saved_models`. This include segment, sensorless-drive, MNIST and CIFAR10 datasets.

## Evaluating models

To evaluate the model(s) in the paper, run one jupyter notebook in the folder `notebooks/models-robustness-evaluation`. In particular you can find notebooks to run **label attacks** and **uncertainty attacks**. All parameter are described.

## Cite
Please cite our paper if you use the models or this code in your own work:

```
@incollection{dbu-robustness,
title = {Evaluating Robustness of Predictive Uncertainty Estimation: Are Dirichlet-based Models Reliable ?},
author = {Anna-Kathrin Kopetzki and Bertrand Charpentier and Daniel Zügner and Sandhya Giri and Stephan Günnemann},
booktitle = {International Conference on Machine Learning},
year = {2021}
}
```
