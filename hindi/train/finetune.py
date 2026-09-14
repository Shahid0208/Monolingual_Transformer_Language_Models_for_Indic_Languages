#!/usr/bin/env python3
"""
Hindi (Model H) Reasoning Finetuning Entry Point
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.finetune_reasoning import train_finetune, parse_args

if __name__ == "__main__":
    args = parse_args()
    args.language = "Hindi"
    train_finetune(args)
