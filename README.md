<h1 align = "center">
Towards Unified Multimodal Editing with Enhanced Knowledge Collaboration
</h1>

<div align="center">
Kaihang Pan<sup>1</sup>*, Zhaoyu Fan<sup>1</sup>*, Juncheng Li<sup>1&dagger;</sup>, Qifan Yu<sup>1</sup>, Hao Fei<sup>2</sup>, Siliang Tang<sup>1</sup>, Richang Hong<sup>3</sup>, Hanwang Zhang<sup>4</sup>, Qianru Sun<sup>5</sup>

<sup>1</sup>Zhejiang University, <sup>2</sup>National University of Singapore, <sup>3</sup>Hefei University of Technology, <sup>4</sup>Nanyang Technological University, <sup>5</sup>Singapore Management University

<sup>*</sup>Equal Contribution, <sup>&dagger;</sup>Corresponding Author

<div align="left">

This repo contains the PyTorch implementation of [Towards Unified Multimodal Editing with Enhanced Knowledge Collaboration](https://openreview.net/forum?id=kf80ZS3fVy), which is accepted by **NeurIPS2024 (Spotlight)**.




## Note

<!-- Given our previous busy schedule, we only uploaded a basic version of the code and forgot to check its accuracy. Our code is based on EasyEdit and Tpatcher. During the process of modifying the code files with Tpatcher, we only altered the internal implementation and did not perform  renaming of methods, functions, or classes for UniKE (still with the name of Tpatcher). We apologize that this non-standard naming has caused minsunderstanding. Additionally, the current code is a version used for later ablation experiments and may differ from the main method in some details. We will promptly reorganize the code in a standardized manner to avoid any unnecessary misunderstandings. -->
(**Chinese Version**) 在上一版本的代码中，我们错误地上传了一个消融实验的代码版本（其中包含了UNIKE的核心实现，只是main函数easyeditor/models/unike/unike_main.py中由于消融的setting导致一些实现代码不会被调用到）；同时，我们的代码基于T-Patcher实现，存在一些命名不规范的情况：直接修改了T-Patcher的一些内部实现逻辑，但没有修改对应的函数或方法名称（名称上依然是T-Patcher）。这两方面引发了一些误解，误以为我们只是调用T-Patcher的原实现进行编辑。对此，我们为自己的疏忽诚挚道歉，对代码进行了紧急规范化的调整，在此给出一些核心实现的代码位置。


(**English Version**) In the previous version of the code, we mistakenly uploaded a version intended for ablation experiments (however, it did include the core implementation of UNIKE;). Additionally, our code is based on T-Patcher and contains some non-standard naming conventions: we directly modified some internal implementation logic of T-Patcher but did not change the corresponding function or method names (which still bear the names of T-Patcher). These two aspects have caused some misunderstandings, leading to the assumption that we merely invoked the original implementation of T-Patcher for editing. For this oversight, we sincerely apologize, have urgently standardized the code, and are now providing the locations of some core implementations.

Intrinsic Knowledge Editing: We follow the implementation of T-Patcher.

External Knowledge Resorting: Please refer to [AdapterLayer](easyeditor/models/unike/src/models/patch.py#105).

Knowledge Collaboration: Please refer to [easyeditor/models/unike/src/models/patch.py#389](easyeditor/models/unike/src/models/patch.py#389) and [easyeditor/models/unike/src/models/patch.py#136](easyeditor/models/unike/src/models/patch.py#136).


## Run the code
First setup the python environment in the following way:
```bash
pip install -r requirements.txt
```

Then download some necessary data and model checkpoints.
Download MiniGPT4 and Blip2-OPT into the `hugging_cache` dir. Download MMEdit data into the `data` dir, with images saved in `images` dir, please refer to [EasyEdit](https://github.com/zjunlp/EasyEdit) for more details 【To be continued...】

To run the multimodal editing:
```bash
python run_edit.py --hparams_dir hparams/minigpt4.yaml --task vqa
```

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
