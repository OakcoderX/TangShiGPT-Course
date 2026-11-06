"""Train a lesson-sized GPT, continue training, or generate with a saved model."""
import argparse
import math
from pathlib import Path
import random

import torch
from torch.nn import functional as F

from .checkpoint import load_checkpoint, save_checkpoint
from .model import Config, GPT


def device_for(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("此环境没有可用的 MPS，请用 --device cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("此环境没有可用的 CUDA，请用 --device cpu")
    if name == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    return torch.device(name)


def batch(rows, stop, device):
    length = max(len(row) - 1 for row in rows)
    x = torch.full((len(rows), length), stop, dtype=torch.long)
    y = torch.full_like(x, -100)
    for i, row in enumerate(rows):
        x[i, :len(row) - 1] = torch.tensor(row[:-1])
        y[i, :len(row) - 1] = torch.tensor(row[1:])
    return x.to(device), y.to(device)


def loss_for(model, examples, stop, device):
    x, y = batch(examples, stop, device)
    # The actual ending token is supervised. Only padded targets equal -100.
    return F.cross_entropy(model(x).transpose(1, 2), y, ignore_index=-100)


def train(args):
    if args.steps < 1 or args.batch < 1 or (args.limit is not None and args.limit < 3):
        raise ValueError("steps 和 batch 必须为正，limit 至少为 3")
    if args.lr is not None and (not math.isfinite(args.lr) or args.lr <= 0):
        raise ValueError("学习率必须是有限正数")
    if args.output.exists():
        raise FileExistsError(f"输出已存在，请换一个文件名 {args.output}")
    if args.output.suffix != ".pt":
        raise ValueError("课堂训练输出请使用 .pt 扩展名")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    rows = list(dict.fromkeys(line.strip() for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()))
    config_args = {k:getattr(args,k) for k in ("width","layers","heads","context") if getattr(args,k) is not None}
    if args.init:
        if config_args:
            raise ValueError("继续训练沿用已保存的结构，请去掉宽度、层数、头数和窗口参数")
        model, chars = load_checkpoint(args.init, args.vocab)
        vocab = {char:i for i,char in enumerate(chars)}
        config = model.config
        retained = [line for line in rows if set(line) <= vocab.keys()]
        print(f"词表之外的字使 {len(rows)-len(retained)} 首诗被跳过")
        rows = retained
    else:
        config = Config(**config_args)
    rows = [line for line in rows if len(line)+1 <= config.context]
    rng.shuffle(rows)
    if args.limit is not None:
        rows = rows[:args.limit]
    if len(rows) < 3:
        raise ValueError("可用诗篇少于 3 首，请检查上下文长度和语料")
    if not args.init:
        chars = sorted(set("".join(rows)))
        vocab = {char:i for i,char in enumerate(chars)}
        model = GPT(config,len(chars)+1)
    stop = len(chars)
    sequences = [[stop]+[vocab[c] for c in row]+[stop] for row in rows]
    n_val = max(1,len(rows)//10)
    validation = sequences[:n_val]
    discovery = sequences[n_val:]
    device = device_for(args.device)
    model.to(device)
    rate = args.lr if args.lr is not None else (1e-4 if args.init else 6e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=rate)
    print(f"{device} | 训练 {len(discovery)} 首 | 验证 {len(validation)} 首 | 词表 {len(chars)+1}")
    for i in range(1,args.steps+1):
        examples = rng.choices(discovery,k=args.batch)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for(model,examples,stop,device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        if i in (1,args.steps) or i % args.log_every == 0:
            value = loss.item()
            if not math.isfinite(value):
                raise ArithmeticError("训练损失不是有限数")
            print(f"{i}/{args.steps} 训练损失 {value:.4f}",flush=True)
    model.eval()
    with torch.inference_mode():
        print(f"验证样本损失 {loss_for(model,validation[:64],stop,device).item():.4f}")
    save_checkpoint(args.output,model,chars)
    print(f"权重已保存至 {args.output}")


@torch.inference_mode()
def generate(args):
    if not math.isfinite(args.temperature) or args.temperature <= 0 or args.max_chars < 1 or args.count < 1:
        raise ValueError("温度、生成长度和首数必须为正")
    model,chars = load_checkpoint(args.weights,args.vocab)
    model.eval().to(device_for(args.device))
    vocab = {char:i for i,char in enumerate(chars)}
    stop = len(chars)
    if set(args.prompt)-vocab.keys():
        raise ValueError(f"首句含词表之外的字 {''.join(sorted(set(args.prompt)-vocab.keys()))}")
    generator = torch.Generator().manual_seed(args.seed)
    allowed = torch.tensor([i for i,c in enumerate(chars) if "\u3400" <= c <= "\u9fff" and c != "囗"])
    width = {"natural":0,"five":5,"seven":7}[args.form]
    if width and (len(args.prompt) > width or any(c not in {chars[i] for i in allowed.tolist()} for c in args.prompt)):
        raise ValueError("定长模式的开头只接受不超过首句字数的汉字")
    if width and (allowed.numel() == 0 or any(c not in vocab for c in "，。")):
        raise ValueError("当前词表缺少定长诗所需汉字或标点，请使用自然模式")
    if len(args.prompt) >= model.config.context or width*4+4 > model.config.context:
        raise ValueError("当前上下文窗口太短")

    def draw(ids, restricted=False):
        logits = model(torch.tensor([ids],device=next(model.parameters()).device))[0,-1].float().cpu()
        if restricted:
            logits = logits[allowed]
        index = torch.multinomial((logits/args.temperature).softmax(0),1,generator=generator).item()
        return allowed[index].item() if restricted else index

    for _ in range(args.count):
        ids = [stop]+[vocab[c] for c in args.prompt]
        if width:
            for i,mark in enumerate("，。，。"):
                for _ in range(width-(len(args.prompt) if i==0 else 0)):
                    ids.append(draw(ids,restricted=True))
                ids.append(vocab[mark])
        else:
            for _ in range(min(args.max_chars,model.config.context-len(ids)+1)):
                token = draw(ids)
                if token == stop:
                    break
                ids.append(token)
        print("".join(chars[i] for i in ids[1:]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode",required=True)
    for mode in ("train","generate"):
        cli = modes.add_parser(mode)
        cli.add_argument("--device",choices=["auto","cpu","mps","cuda"],default="auto")
        cli.add_argument("--seed",type=int,default=42)
        cli.add_argument("--vocab",type=Path)
        if mode == "train":
            cli.add_argument("--data",type=Path,default=Path("data/tangshi_clean.txt"))
            cli.add_argument("--output",type=Path,required=True)
            cli.add_argument("--init",type=Path)
            for name in ("width","layers","heads","context","limit"):
                cli.add_argument(f"--{name}",type=int)
            cli.add_argument("--steps",type=int,default=200)
            cli.add_argument("--batch",type=int,default=8)
            cli.add_argument("--lr",type=float)
            cli.add_argument("--log-every",type=int,default=50)
        else:
            cli.add_argument("--weights",type=Path,required=True)
            cli.add_argument("--prompt",default="")
            cli.add_argument("--form",choices=["natural","five","seven"],default="natural")
            cli.add_argument("--temperature",type=float,default=.65)
            cli.add_argument("--max-chars",type=int,default=100)
            cli.add_argument("--count",type=int,default=1)
    args=parser.parse_args()
    try:
        if args.mode=="train" and args.log_every<1:
            parser.error("--log-every 必须大于零")
        (train if args.mode=="train" else generate)(args)
    except (ValueError,OSError) as error:
        parser.error(str(error))


if __name__=="__main__":
    main()
