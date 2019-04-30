# AnyNet: PYTORCH 1 - CUDA 10 - WANDB-LOGS (no spn)

# Results and visual [here](https://app.wandb.ai/pier/cloned_anynet_main)
## [init channels analysis (sceneflow)](https://app.wandb.ai/pier/cloned_anynet_main/reports?view=pier%2Finit_channels%20sceneflow%20analysis) 
## [init channels analysis (KITTI 2015)](https://app.wandb.ai/pier/cloned_anynet_finetune/reports?view=pier%2Finit_channels%20KITTI%20analysis) 

___________________________________________________________________________________________________

This repository contains the code (in PyTorch) for AnyNet introduced in the following paper

[Anytime Stereo Image Depth Estimation on Mobile Devices](https://arxiv.org/abs/1810.11408)

by [Yan Wang∗](https://www.cs.cornell.edu/~yanwang/), Zihang Lai∗, [Gao Huang](http://www.gaohuang.net/), [Brian Wang](https://campbell.mae.cornell.edu/research-group/brian-wang), [Laurens van der Maaten](https://lvdmaaten.github.io/), [Mark Campbell](https://campbell.mae.cornell.edu/) and [Kilian Q. Weinberger](http://kilian.cs.cornell.edu/).

It has been accepted by International Conference on Robotics and Automation (ICRA) 2019.

![Figure](figures/network.png)

### Citation
```
@article{wang2018anytime,
  title={Anytime Stereo Image Depth Estimation on Mobile Devices},
  author={Wang, Yan and Lai, Zihang and Huang, Gao and Wang, Brian H. and Van Der Maaten, Laurens and Campbell, Mark and Weinberger, Kilian Q},
  journal={arXiv preprint arXiv:1810.11408},
  year={2018}
}
```

## Usage
0. Install dependencies
1. Generate the soft-links for the SceneFlow Dataset. You need to modify the `scenflow_data_path` to the actual SceneFlow path in `create_dataset.sh` file. 
    ```shell2html
     sh ./create_dataset.sh
    ```
2. Compile SPNet if SPN refinement is needed. (change NVCC path in make.sh when necessary)
    ```
    cd model/spn
    sh make.sh
    ```
### Dependencies

- [Python3.7](https://www.python.org/downloads/)
- [PyTorch(1.0)](http://pytorch.org)
- CUDA 10.0
- [KITTI Stereo](http://www.cvlibs.net/datasets/kitti/eval_stereo.php)
- [Scene Flow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

### Train
Firstly, we use the following command to pretrained AnyNet on Scene Flow

```
python main.py --maxdisp 192 --with_spn
```

Secondly, we use the following command to finetune AnyNet on KITTI 2015 

```
python finetune.py --maxdisp 192 --with_spn --datapath path-to-kitti2015/training/
```



## Results

![Figure KITTI2012 Results](figures/results.png) 
