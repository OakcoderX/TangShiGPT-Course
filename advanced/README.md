# Apple Silicon 进阶训练

此入口沿用项目从头比较宽度的 MLX 实现。工作目录为课程根目录。它用历史清洗文本和李白署名文本重新构造划分，生成的训练产物放到 `runs/mlx/`。通用训练四轮，随后进行 360 步李白增强。每个宽度保留两个阶段的检查点。教程的短练习用 `python -m tangshi train`，耗时更少。

```sh
python -m pip install mlx==0.32.2 numpy
python advanced/train_mlx.py benchmark
python advanced/train_mlx.py train --width 128
```

完整训练实验可换 `--width 256` 或 `--width 512`。脚本会核对 `weights/model_base.pt` 的哈希以确认档案未改变；训练以随机初始化起步。已有同名结果目录会退出。MLX 使用 Metal GPU，BF16 是该历史流程的默认精度。硬件与版本差异可能改变速度或采样结果。

课程随附的 512 维候选在这个从头流程之后又继续做了额外的通用训练与李白增强。此入口复现宽度对照的训练流程，续训证据和配置在 [历史续训报告](../experiments/scratch_width_mlx_20261106/width512_more_steps/实验报告.md)。
