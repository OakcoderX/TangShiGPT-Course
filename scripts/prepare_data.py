"""Turn the bundled upstream JSON snapshot into one poem per line."""
import argparse
from collections import Counter
import json
from pathlib import Path

from clean_rules import clean_poem

ROOT = Path(__file__).resolve().parents[1]

def write_lines(path, rows):
    path.write_text("".join(row + "\n" for row in rows), encoding="utf-8")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/lesson-data")
    args = parser.parse_args()
    files = sorted(args.raw.glob("全唐诗_poet.tang.*.json"),
                   key=lambda path: int(path.name.split(".")[-2]))
    if len(files) != 58:
        parser.error(f"预期 58 个原始 JSON，找到 {len(files)} 个")
    records = [row for path in files for row in json.loads(path.read_text(encoding="utf-8"))]
    raw, clean, li = [], [], []
    for row in records:
        poem = "".join(row.get("paragraphs") or []).strip()
        if poem:
            raw.append(poem)
            if row.get("author") == "李白":
                li.append(poem)
    # The historical cleaner reads line by line after extraction.
    lines = [line.strip() for poem in raw for line in poem.splitlines() if line.strip()]
    for line in lines:
        text = clean_poem(line)
        if text:
            clean.append(text)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("tangshi.txt", raw), ("tangshi_clean.txt", clean), ("tangshi_李白.txt", li)):
        write_lines(args.output / name, rows)
    print(f"JSON 记录 {len(records)}；提取 {len(raw)}；清洗后 {len(clean)}；李白署名 {len(li)}")
    print("常见字符", Counter("".join(clean)).most_common(6))
    print(args.output)

if __name__ == "__main__":
    main()
