#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


def load_map(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_pattern(keys: list[str]) -> re.Pattern:
    escaped = [re.escape(k) for k in sorted(keys, key=len, reverse=True)]
    pattern = r"(?<![A-Za-z0-9_])(" + "|".join(escaped) + r")(?![A-Za-z0-9_])"
    return re.compile(pattern)


def transform_text(text: str, mapping: dict[str, str], pattern: re.Pattern) -> str:
    return pattern.sub(lambda m: mapping[m.group(1)], text)


def convert_dir(input_dir: Path, output_dir: Path, mapping: dict[str, str]) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = build_pattern(list(mapping.keys()))
    converted = 0

    for src in sorted(input_dir.rglob("*")):
        if not src.is_file() or src.suffix.lower() not in {".md", ".txt"}:
            continue

        rel = src.relative_to(input_dir)
        dst = output_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        raw = src.read_text(encoding="utf-8")
        dst.write_text(transform_text(raw, mapping, pattern), encoding="utf-8")
        converted += 1

    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description="Replace domain terms in KB files.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--map", required=True, dest="map_path")
    args = parser.parse_args()

    mapping = load_map(Path(args.map_path))
    count = convert_dir(Path(args.input_dir), Path(args.output_dir), mapping)
    print(f"Converted files: {count}")


if __name__ == "__main__":
    main()
