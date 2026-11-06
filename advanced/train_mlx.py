"""Reproducible scratch width comparison on MLX/Metal."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/mlx"
CORPUS = ROOT / "data/tangshi_clean.txt"
LI = ROOT / "data/tangshi_李白.txt"
HAN = re.compile(r"[^\u3400-\u9fff]")
BUCKETS = (32, 48, 64, 96, 128)
BATCH = 32
SEED = 20261106

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def body(text):
    return HAN.sub("", text)

def grams(text, n=8):
    return (text[i:i+n] for i in range(len(text) - n + 1))

def split_data():
    texts = CORPUS.read_text().splitlines()
    chars = sorted(set("".join(texts)))
    li_bodies = {body(text) for text in LI.read_text().splitlines()}
    poems = {}
    for text in texts:
        key = body(text)
        if (20 <= len(key) and len(text) < 128 and text.endswith(("。", "！", "？"))
                and "□" not in text and "囗" not in text):
            poems.setdefault(key, text)
    groups = defaultdict(list)
    for key, text in sorted(poems.items()):
        group = "li" if key in li_bodies else "other"
        h = hashlib.sha256(("width-v1/" + key).encode()).hexdigest()
        groups[group].append((h, key, text))
    train, dev, test = {}, {}, {}
    used_test, used_dev = set(), set()
    for group, rows in groups.items():
        rows.sort()
        count = 85 if group == "li" else 320
        test[group] = [text for _, _, text in rows[:count]]
        for _, key, _ in rows[:count]:
            used_test.update(grams(key))
    for group, rows in groups.items():
        count = 85 if group == "li" else 320
        candidates = ((key, text) for _, key, text in rows[count:])
        selected = []
        for key, text in candidates:
            if not any(g in used_test for g in grams(key)):
                selected.append(text)
                used_dev.update(grams(key))
                if len(selected) == count:
                    break
        assert len(selected) == count
        dev[group] = selected
    reserved = used_test | used_dev
    for group, rows in groups.items():
        train[group] = [text for _, key, text in rows if not any(g in reserved for g in grams(key))]
    return chars, {"train": train, "dev": dev, "test": test}

def linear(layer, x):
    return x @ layer.weight.astype(x.dtype).T

class Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.head_size = width // 8
        self.norm1 = nn.RMSNorm(width, eps=1e-6)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.projection = nn.Linear(width, width, bias=False)
        self.norm2 = nn.RMSNorm(width, eps=1e-6)
        self.up = nn.Linear(width, 4 * width, bias=False)
        self.down = nn.Linear(4 * width, width, bias=False)

    def __call__(self, x):
        batch, length, width = x.shape
        norm = self.norm1(x).astype(x.dtype)
        packed = linear(self.qkv, norm).reshape(batch, length, 3, 8, self.head_size)
        q, k, v = mx.split(packed.transpose(2, 0, 3, 1, 4), 3, axis=0)
        attended = mx.fast.scaled_dot_product_attention(
            q.squeeze(0), k.squeeze(0), v.squeeze(0),
            scale=self.head_size ** -0.5, mask="causal"
        )
        x = x + linear(self.projection, attended.transpose(0, 2, 1, 3).reshape(batch, length, width))
        norm = self.norm2(x).astype(x.dtype)
        return x + linear(self.down, nn.relu(linear(self.up, norm)))

class Model(nn.Module):
    def __init__(self, width, vocab, precision):
        super().__init__()
        self.dtype = mx.bfloat16 if precision == "bf16" else mx.float32
        self.token = nn.Embedding(vocab, width)
        self.position = nn.Embedding(128, width)
        self.blocks = [Block(width) for _ in range(6)]
        self.norm = nn.RMSNorm(width, eps=1e-6)
        self.head = nn.Linear(width, vocab, bias=False)
        params = []
        for name, tensor in tree_flatten(self.parameters()):
            if tensor.ndim == 2:
                scale = .02 / math.sqrt(12) if name.endswith(("projection.weight", "down.weight")) else .02
                tensor = mx.random.normal(tensor.shape) * scale
            params.append((name, tensor))
        self.load_weights(params)

    def __call__(self, ids):
        x = (self.token(ids) + self.position(mx.arange(ids.shape[1]))).astype(self.dtype)
        for block in self.blocks:
            x = block(x)
        return linear(self.head, self.norm(x).astype(self.dtype)).astype(mx.float32)

def encode(rows, vocab):
    end = len(vocab)
    return [[end] + [vocab[char] for char in text] + [end] for text in rows]

def collate(rows, end):
    width = next(n for n in BUCKETS if n >= max(map(len, rows)) - 1)
    x = np.full((BATCH, width), end, dtype=np.int32)
    y = np.zeros((BATCH, width), dtype=np.int32)
    mask = np.zeros((BATCH, width), dtype=np.bool_)
    for i, row in enumerate(rows):
        length = len(row) - 1
        x[i, :length], y[i, :length] = row[:-1], row[1:]
        mask[i, :length] = True
    return mx.array(x), mx.array(y), mx.array(mask), int(mask.sum())

def epoch_batches(rows, seed):
    rng = np.random.default_rng(seed)
    by_length = defaultdict(list)
    for index in rng.permutation(len(rows)):
        bucket = next(n for n in BUCKETS if n >= len(rows[index]) - 1)
        by_length[bucket].append(rows[index])
    batches = [group[i:i+BATCH] for group in by_length.values() for i in range(0, len(group), BATCH)]
    return [batches[i] for i in rng.permutation(len(batches))]

def loss_fn(model, x, y, mask):
    losses = nn.losses.cross_entropy(model(x), y, reduction="none")
    return mx.sum(mx.where(mask, losses, 0)) / mx.sum(mask)

def make_step(model, total, peak_lr, objective=loss_fn):
    warmup = min(100, total // 10)
    cosine = optim.cosine_decay(peak_lr, max(1, total - warmup), peak_lr / 8)
    schedule = optim.join_schedules([optim.linear_schedule(peak_lr / 10, peak_lr, warmup), cosine], [warmup])
    optimizer = optim.AdamW(learning_rate=schedule, betas=(.9, .95), weight_decay=.01)
    optimizer.init(model.trainable_parameters())
    state = [model.state, optimizer.state]
    grads = nn.value_and_grad(model, objective)

    def step(x, y, mask):
        loss, gradient = grads(model, x, y, mask)
        gradient, _ = optim.clip_grad_norm(gradient, 1.0)
        optimizer.update(model, gradient)
        return loss

    return mx.compile(step, inputs=state, outputs=state), state

def measure(model, split, vocab):
    result = {}
    for group, rows in split.items():
        body_loss = end_loss = body_count = end_count = 0
        encoded = encode(rows, vocab)
        for batch in epoch_batches(encoded, SEED):
            x, y, mask, _ = collate(batch, len(vocab))
            losses = nn.losses.cross_entropy(model(x), y, reduction="none")
            stop = mask & (y == len(vocab))
            ordinary = mask & ~stop
            body_loss += mx.sum(mx.where(ordinary, losses, 0)).item()
            end_loss += mx.sum(mx.where(stop, losses, 0)).item()
            body_count += mx.sum(ordinary).item()
            end_count += mx.sum(stop).item()
        result[group] = {"body_nll": body_loss / body_count, "end_nll": end_loss / end_count,
                         "poems": len(rows), "tokens": body_count}
    return result

def run_stage(model, batches, vocab, learning_rate, name):
    step, state = make_step(model, len(batches), learning_rate)
    start = time.monotonic()
    tokens = 0
    for i, rows in enumerate(batches, 1):
        x, y, mask, count = collate(rows, len(vocab))
        loss = step(x, y, mask)
        mx.eval(loss, state)
        tokens += count
        if i == 1 or i % 200 == 0 or i == len(batches):
            value = loss.item()
            if not math.isfinite(value):
                raise ArithmeticError(f"{name} diverged at step {i}")
            print(f"{name} {i}/{len(batches)} loss={value:.4f} tokens/s={tokens/(time.monotonic()-start):.0f}", flush=True)
    return {"seconds": time.monotonic()-start, "tokens": tokens, "steps": len(batches)}

def save(model, path):
    temporary = path.with_suffix(".tmp.safetensors")
    model.save_weights(str(temporary))
    temporary.replace(path)
    return sha(path)

def benchmark(vocab):
    result = {}
    rng = np.random.default_rng(SEED)
    x = mx.array(rng.integers(vocab + 1, size=(BATCH, 48), dtype=np.int32))
    y = mx.array(rng.integers(vocab + 1, size=(BATCH, 48), dtype=np.int32))
    mask = mx.ones(x.shape, dtype=mx.bool_)
    for width in (128, 256):
        for precision in ("fp32", "bf16"):
            mx.random.seed(SEED)
            model = Model(width, vocab+1, precision)
            step, state = make_step(model, 75, .0006)
            for _ in range(15):
                loss = step(x, y, mask)
                mx.eval(loss, state)
            start = time.monotonic()
            for _ in range(60):
                loss = step(x, y, mask)
                mx.eval(loss, state)
            seconds = time.monotonic() - start
            assert math.isfinite(loss.item())
            result[f"{width}_{precision}"] = {"tokens_per_second": 60 * x.size / seconds, "seconds": seconds, "loss": loss.item()}
            print(result, flush=True)
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["benchmark", "train"])
    parser.add_argument("--width", type=int, choices=[128,256,512], default=256)
    parser.add_argument("--precision", choices=["fp32","bf16"], default="bf16")
    args = parser.parse_args()
    mx.set_default_device(mx.gpu)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.mode == "benchmark":
        vocab = len(set("".join(CORPUS.read_text().splitlines())))
        report = {"device": mx.device_info(), "mlx": mx.__version__, "benchmarks": benchmark(vocab)}
        (OUT/"benchmark.json").write_text(json.dumps(report, indent=2))
        return

    root_hash = sha(ROOT/"weights/model_base.pt")
    chars, splits = split_data()
    vocab = {char: i for i, char in enumerate(chars)}
    width = args.width
    run = OUT / f"width{width}"
    run.mkdir(exist_ok=False)
    (run/"split.json").write_text(json.dumps(splits, ensure_ascii=False))
    (run/"vocab.json").write_text(json.dumps(chars, ensure_ascii=False))
    mx.random.seed(SEED)
    model = Model(width, len(vocab)+1, args.precision)
    report = {"width":width, "layers":6, "heads":8, "precision":args.precision,
              "parameters":sum(t.size for _,t in tree_flatten(model.parameters())),
              "base_sha256":root_hash, "corpus_sha256":sha(CORPUS), "li_sha256":sha(LI),
              "split_sha256":sha(run/"split.json"), "device":mx.device_info()}
    all_rows = encode(splits["train"]["li"]+splits["train"]["other"],vocab)
    batches = [batch for epoch in range(4) for batch in epoch_batches(all_rows, SEED+epoch)]
    print("split counts", {part:{a:len(rows) for a,rows in group.items()} for part,group in splits.items()}, flush=True)
    report["general"] = run_stage(model, batches, vocab, .0006, f"{width} general")
    report["general"]["dev"] = measure(model, splits["dev"], vocab)
    report["general"]["test"] = measure(model, splits["test"], vocab)
    report["general"]["sha256"] = save(model, run/"general.safetensors")
    (run/"results.json").write_text(json.dumps(report, indent=2))
    li = encode(splits["train"]["li"], vocab)
    other = encode(splits["train"]["other"], vocab)
    rng = np.random.default_rng(SEED+101)
    style_batches = [[li[i] for i in rng.integers(len(li), size=12)] +
                     [other[i] for i in rng.integers(len(other), size=20)] for _ in range(360)]
    report["li"] = run_stage(model, style_batches, vocab, .00015, f"{width} Li Bai")
    report["li"]["dev"] = measure(model, splits["dev"], vocab)
    report["li"]["test"] = measure(model, splits["test"], vocab)
    report["li"]["sha256"] = save(model, run/"li.safetensors")
    assert sha(ROOT/"weights/model_base.pt") == root_hash
    (run/"results.json").write_text(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
