#!/usr/bin/env python
"""训练入口: 在仓库根目录下运行

    python scripts/run_train.py

产出(权重、OOF、损失曲线、ROC/PR曲线、测试集预测结果)默认写到 outputs/。
"""
import sys
from pathlib import Path

# 把仓库根目录加入 sys.path,这样无论从哪个工作目录运行本脚本,
# `from src...` 这种包内相对导入都能正常工作。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train import main  # noqa: E402

if __name__ == '__main__':
    main()
