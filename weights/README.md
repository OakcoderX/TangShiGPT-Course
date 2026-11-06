# 权重参考

完整课程 ZIP 随附下列两份原始文件。若从 GitHub 源码下载，请把 Release 附件按原文件名放到此目录。

- [`model_base.pt`](https://github.com/OakcoderX/TangShiGPT-Course/releases/download/v1.0.0/model_base.pt) 是早期 128 维、6 层、8 头的历史 PyTorch 权重。
- [`more_2400_li.safetensors`](https://github.com/OakcoderX/TangShiGPT-Course/releases/download/v1.0.0/more_2400_li.safetensors) 是 512 维、6 层、8 头的历史 MLX 检查点，位于情志专项训练之前。它是本项目生成盲评中保留的表现较好版本。
- `vocab.json` 保存两份历史权重共同使用的字符顺序。最后一个编号保留给起止标记。
- `SHA256SUMS` 保存权重和词表的哈希。在本目录执行 `shasum -a 256 -c SHA256SUMS` 可核对。

课程推理入口在 CPU 上以 FP32 读取权重。历史 MLX 评测采用 BF16，样本也使用不同的抽样实现，因此同一个数字种子不保证复刻历史诗稿。
