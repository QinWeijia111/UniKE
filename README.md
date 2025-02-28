<h1 align = "center">
Towards Unified Multimodal Editing with Enhanced Knowledge Collaboration
</h1>

<div align="center">
Kaihang Pan<sup>1</sup>*, Zhaoyu Fan<sup>1</sup>*, Juncheng Li<sup>1&dagger;</sup>, Qifan Yu<sup>1</sup>, Hao Fei<sup>2</sup>, Siliang Tang<sup>1</sup>, Richang Hong<sup>3</sup>, Hanwang Zhang<sup>4</sup>, Qianru Sun<sup>5</sup>

<sup>1</sup>Zhejiang University, <sup>2</sup>National University of Singapore, <sup>3</sup>Hefei University of Technology, <sup>4</sup>Nanyang Technological University, <sup>5</sup>Singapore Management University

<sup>*</sup>Equal Contribution, <sup>&dagger;</sup>Corresponding Author

<div align="left">

This repo contains the PyTorch implementation of [Towards Unified Multimodal Editing with Enhanced Knowledge Collaboration](https://openreview.net/forum?id=kf80ZS3fVy), which is accepted by **NeurIPS2024 (Spotlight)**.

### Note

Given our previous busy schedule, we only uploaded a basic version of the code and forgot to check its accuracy. Our code is based on EasyEdit and Tpatcher. During the process of modifying the code files with Tpatcher, we only altered the internal implementation and did not perform  renaming of methods, functions, or classes for UniKE (still with the name of Tpatcher). We apologize that this non-standard naming has caused minsunderstanding. Additionally, the current code is a version used for later ablation experiments and may differ from the main method in some details. We will promptly reorganize the code in a standardized manner to avoid any unnecessary misunderstandings.

## Acknowledgment

Our project is developed based on the following repositories:

* [EasyEdit](https://github.com/zjunlp/EasyEdit): An Easy-to-use Knowledge Editing Framework for LLMs.

* [Transformer-Patcher](https://github.com/ZeroYuHuang/Transformer-Patcher): One mistake worth one neuron.

## Citation
If you found this work useful, please consider  citing our paper as follows:
```
@article{pan2024towards,
  title={Towards unified multimodal editing with enhanced knowledge collaboration},
  author={Pan, Kaihang and Fan, Zhaoyu and Li, Juncheng and Yu, Qifan and Fei, Hao and Tang, Siliang and Hong, Richang and Zhang, Hanwang and Sun, Qianru},
  journal={arXiv preprint arXiv:2409.19872},
  year={2024}
}
```
