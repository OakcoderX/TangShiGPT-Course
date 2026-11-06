# 从 microGPT 到唐诗 GPT

一套面向会一点 Python 的初学者的中文动手课。先把《全唐诗》的 JSON 变成模型可读的字符序列，再亲手训练、保存、加载一个小 GPT。最后用真实的李白风格实验练习设计对照和判断失败。

**项目做成了能生成唐诗形式草稿的模型，李白风格仍不能稳定复现。**

## 从这里开始

1. [下载完整学习包](https://github.com/OakcoderX/TangShiGPT-Course/releases/download/v1.0.0/TangShiGPT-Course.zip)并解压。用浏览器打开 [HTML 自学教案](docs/index.html)，在项目根目录执行课程命令。
2. 讲课时打开 [PowerPoint 讲课版](docs/TangShiGPT-课程.pptx)。讲稿提示放在幻灯片备注。
3. 如果已下载仓库源码，只需从 [Release](https://github.com/OakcoderX/TangShiGPT-Course/releases/tag/v1.0.0) 单独下载权重，按原文件名放进 `weights/`。目录内已有词表和校验清单。

## 快速体验

需要 Python 3.10 以上。命令在项目根目录执行。Mac、Windows、Linux 可用 CPU；`--device auto` 会优先尝试可用的 CUDA 或 Apple MPS。

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m tangshi generate --weights weights/more_2400_li.safetensors --form seven --prompt 江上送故人 --seed 42
```

Windows PowerShell 使用 `py -m venv .venv` 与 `.venv\Scripts\Activate.ps1`，其余命令使用 `python`。如激活受本机策略限制，直接使用 `.venv\Scripts\python.exe` 执行命令。

## 从原始数据到自己的模型

```sh
python scripts/prepare_data.py
python scripts/next_char.py --char 月
python -m tangshi train --data runs/lesson-data/tangshi_clean.txt --limit 2000 --context 48 --width 64 --layers 2 --steps 200 --output runs/student.pt
python -m tangshi generate --weights runs/student.pt --form five --seed 42
```

短跑用于观察训练和生成，作品通常还很生涩。训练命令遇到同名输出会退出。第 5 章提供原项目的 Apple MLX 宽度实验入口，第 7 章用 `--init` 从自己的已保存模型继续训练。历史结果见[项目结案记录](docs/历史结案记录.md)。

## 包内材料

| 路径 | 用途 |
| --- | --- |
| `data/raw/` | 项目实际使用的 58 个上游《全唐诗》JSON，清洗前原文件 |
| `data/tangshi_clean.txt` | 已清洗的历史训练文本 |
| `data/tangshi_李白.txt` | 按上游作者字段提取的历史李白文本 |
| `scripts/` | 数据重建和下一字频数基线 |
| `tangshi/` | 教学实现，支持训练、保存、生成和继续训练 |
| `advanced/` | Apple Silicon 上的 MLX 训练流程 |
| `weights/` | 两份历史权重、共享词表和 SHA-256 清单 |
| `docs/` | HTML、自学练习、可编辑 PPTX 和历史结案 |
| `experiments/` | 课堂引用的实验记录与匿名诗稿 |

致谢与发布前事项见 [来源与发布说明](docs/来源与发布.md)。
