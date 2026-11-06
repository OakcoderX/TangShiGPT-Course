"""Read course checkpoints and the two archived reference checkpoints."""
from dataclasses import asdict
import json
from pathlib import Path

import torch

from .model import Config, GPT


def legacy_state(source, layers):
    # The original PyTorch model used separate Q, K and V projections.
    state = {
        "token.weight": source["wte.weight"],
        "position.weight": source["wpe.weight"],
        "norm.weight": source["ln_f.weight"],
        "head.weight": source["lm_head.weight"],
    }
    for i in range(layers):
        old, new = f"layers.{i}.", f"blocks.{i}."
        for dst, src in (("norm1", "ln1"), ("norm2", "ln2"), ("projection", "attn.o_proj"),
                         ("up", "mlp.fc1"), ("down", "mlp.fc2")):
            state[new + dst + ".weight"] = source[old + src + ".weight"]
        state[new + "qkv.weight"] = torch.cat([source[old + f"attn.{part}_proj.weight"] for part in "qkv"])
    return state


def load_checkpoint(path, vocab_path=None):
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        vocab_path = vocab_path or path.with_name("vocab.json")
        chars = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
        state = load_file(str(path), device="cpu")
        config = Config(width=512, layers=6, heads=8, context=128)
    elif path.suffix == ".pt":
        source = torch.load(path, map_location="cpu", weights_only=True)
        if "state" in source:
            chars, state, config = source["chars"], source["state"], Config(**source["config"])
        else:
            vocab_path = vocab_path or path.with_name("vocab.json")
            chars = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
            config = Config(width=128, layers=6, heads=8, context=128, eps=torch.finfo(torch.float32).eps)
            state = legacy_state(source, config.layers)
    else:
        raise ValueError("权重扩展名需为 .pt 或 .safetensors")
    if not isinstance(chars, list) or len(set(chars)) != len(chars) or any(len(c) != 1 for c in chars):
        raise ValueError("词表必须是互不重复的单字符列表")
    model = GPT(config, len(chars) + 1)
    model.load_state_dict(state, strict=True)
    return model, chars


def save_checkpoint(path, model, chars):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config": asdict(model.config), "chars": chars,
               "state": {k:v.detach().cpu() for k,v in model.state_dict().items()}}
    # Exclusive creation protects the archived files and previous student runs.
    with path.open("xb") as handle:
        torch.save(payload, handle)
