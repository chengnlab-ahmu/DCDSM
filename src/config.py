"""全局配置：路径、随机种子、模型结构与训练超参数。

把所有可调参数集中在这一个文件里,方便复现和调参 —— clone 仓库、把数据放进
data/ 目录之后,通常只需要改这一个文件(或者用命令行参数覆盖,见 scripts/)
就能跑起来,不需要去翻各个模块内部找写死的数字。

本仓库只保留最终选定的一套配置(方案D 双向交叉注意力融合 + BCE/Tversky
混合损失,对应原实验记录里的 EXPERIMENT=2.5),不再包含其它探索过的融合方式
/损失函数的开关,如果需要回顾对比实验,请看 git 历史或实验记录文档。
"""
import os
from pathlib import Path

# ===================== 路径 =====================
# 项目根目录 = 本文件所在目录(src/)的上一级
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 数据目录默认是仓库里的 data/ 文件夹;如果你的数据放在别处,
# 设置环境变量 DCDSM_DATA_DIR 覆盖,不用改代码。
DATA_DIR = Path(os.environ.get("DCDSM_DATA_DIR", PROJECT_ROOT / "data"))

# 训练/评估产出(权重、OOF、曲线图、预测结果)的默认输出目录,同样可用
# 环境变量 DCDSM_OUTPUT_DIR 覆盖。
OUTPUT_DIR = Path(os.environ.get("DCDSM_OUTPUT_DIR", PROJECT_ROOT / "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 训练集/测试集四个 CSV 文件,文件名需和 data/README.md 里说明的保持一致
TRAIN_MANUAL_CSV = DATA_DIR / "train_lightGBM_threshold-1.csv"
TRAIN_BERT_CSV = DATA_DIR / "train_bert_lightGBM_processed.csv"
TEST_MANUAL_CSV = DATA_DIR / "test02_lightGBM_threshold-1.csv"
TEST_BERT_CSV = DATA_DIR / "test02_bert_lightGBM_processed.csv"

# ===================== 随机种子 =====================
# 同时用于: numpy / python random / tensorflow / StratifiedKFold / 训练集shuffle
RANDOM_SEED = 42

# ===================== 融合层结构 (方案D: 双向交叉注意力 + 门控残差) =====================
FUSION_D_MODEL = 128
FUSION_NUM_HEADS = 4
FUSION_DROPOUT = 0.1

# ===================== 损失函数: BCE + Tversky 混合 =====================
# Tversky_index = TP / (TP + alpha*FP + beta*FN)
#   alpha 越大 -> 越惩罚假阳性(FP) -> 越保守 -> 提升 Specificity
#   beta  越大 -> 越惩罚假阴性(FN) -> 越倾向多抓正类 -> 提升 Sensitivity
TVERSKY_ALPHA = 0.8
TVERSKY_BETA = 0.2

# 总 loss = BCE_WEIGHT * BCE + (1 - BCE_WEIGHT) * (1 - Tversky_index)
# 越接近1越像纯BCE(排序质量AUC/AUPRC更好,但Sn/Sp可能失衡);
# 越接近0越像纯Tversky(Sn/Sp更平衡,但AUC/AUPRC可能下降)。
# 0.3 是原实验记录里对比过 0.5 后最终选定的取值。
BCE_WEIGHT = 0.1

# ===================== 训练超参数 =====================
NUM_FOLDS = 10
EPOCHS = 300
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EARLY_STOPPING_PATIENCE = 20  # 监控 val_loss, restore_best_weights=True

# ===================== 评估 =====================
# 固定阈值,不做阈值搜索,符合论文评估协议(所有指标统一用0.5切分)
DECISION_THRESHOLD = 0.5

# ===================== 输出文件命名 =====================
# 所有权重/曲线图/OOF/预测结果文件名的统一前缀
MODEL_TAG = "crossattn_bce_tversky"
