"""A count-based baseline for the next-character prediction lesson."""
from collections import Counter, defaultdict
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/tangshi_clean.txt"))
    parser.add_argument("--char", default="月")
    args = parser.parse_args()
    pairs = defaultdict(Counter)
    for line in args.data.read_text(encoding="utf-8").splitlines():
        sequence = ["<起>"] + list(line) + ["<止>"]
        for current, target in zip(sequence, sequence[1:]):
            pairs[current][target] += 1
    row = pairs[args.char]
    total = row.total()
    if not total:
        parser.error("这个字符未出现在语料中")
    for char, count in row.most_common(10):
        print(f"{args.char} → {char}  {count:6d}  {count / total:7.2%}")

if __name__ == "__main__":
    main()
