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


## Current experiment: temporal consistency

See [log diagnosis, implementation, run commands and ablations](EXPERIMENT_TEMPORAL_CONSISTENCY.md).

```bash
bash scripts/run_temporal_consistency.sh
```

Behavioral tests: `python -m unittest discover -s tests -v`.

## MobileNetV2 dual-view EMA baseline

See [baseline implementation changes and training commands](IMPROVEMENTS.md).
The architecture figure below describes that baseline; the temporal additions are documented above.

[Code-audited architecture and figure downloads](docs/ARCHITECTURE.md)

![MobileNetV2 Dual-View EMA CAST](docs/cast_mobilenetv2_framework.png)
