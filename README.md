# CAST

This is a PyTorch implementation of the paper:

"Unsupervised Cross-Domain Facial Expression Recognition via Class Adaptive self-Training"
## Environment
Ubuntu 16.04 LTS, python 3.8, pytorch 1.8.1
## Datasets
[RAFDB](http://www.whdeng.cn/raf/model1.html),
[AFE](https://github.com/HCPLab-SYSU/CD-FER-Benchmark),
[EXPW](http://mmlab.ie.cuhk.edu.hk/projects/socialrelation/index.html),
[SFEW](https://paperswithcode.com/dataset/sfew),
[FER2013](https://paperswithcode.com/dataset/fer2013)


## MobileNetV2 dual-view EMA branch

See [implementation changes and training commands](IMPROVEMENTS.md) and the behavioral tests in `tests/test_cast.py`.

[Code-audited architecture and figure downloads](docs/ARCHITECTURE.md)

![MobileNetV2 Dual-View EMA CAST](docs/cast_mobilenetv2_framework.png)
