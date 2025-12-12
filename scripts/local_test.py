"""
Quick local test helper: runs the MODNet pipeline on a local image and writes
an RGBA PNG to disk. This bypasses the API and R2 upload layers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Ensure project root is importable when running from scripts/
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modnet_service.pipeline import process_image_bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MODNet on a local image")
    parser.add_argument("--input", required=True, help="Path to the input image")
    parser.add_argument("--output", required=True, help="Path to write the RGBA PNG")
    parser.add_argument("--mode", default="standard", choices=["fast", "standard", "high"], help="Quality mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    image_bytes = input_path.read_bytes()
    png_bytes = process_image_bytes(image_bytes, quality_mode=args.mode)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(png_bytes)
    print(f"Wrote RGBA output to {output_path}")


if __name__ == "__main__":
    main()
