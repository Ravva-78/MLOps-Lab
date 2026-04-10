#!/usr/bin/env python3

import json
from pathlib import Path


def ensure_directory(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data, filepath):
    ensure_directory(Path(filepath).parent)

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_json(filepath):
    with open(filepath) as f:
        return json.load(f)


def print_section(title):
    print("\n==============================")
    print(title)
    print("==============================\n")


if __name__ == "__main__":
    print_section("Pipeline Utilities")
    print("✓ json helpers loaded")