#!/usr/bin/env python
"""评估入口: 在仓库根目录下运行

    python scripts/run_evaluate.py
    python scripts/run_evaluate.py --test-manual ... --test-bert ... --weights ...

参数说明见 `python scripts/run_evaluate.py --help`,或直接看 src/evaluate.py 顶部的文档。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import main  # noqa: E402

if __name__ == '__main__':
    main()
