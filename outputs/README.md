# DCDSM Cross-Attention (方案D + BCE/Tversky混合损失)

双分支 CNN-BiLSTM 多模态(语言模型特征 + 手工特征)二分类模型,融合层用
**双向交叉注意力 + 门控残差**(方案D),损失函数用 **BCE + Tversky 混合损失**
来平衡 Sensitivity / Specificity 和 AUC/AUPRC。

这是从探索阶段的多方案对比(不同融合层: token自注意力/GMU/SE通道重标定/
直接拼接/交叉注意力;不同损失: 纯BCE/纯Tversky/BCE+Tversky混合/难负样本
挖掘)里最终选定的一套配置,整理成的可复现、可发布到 GitHub 的最小实现,
去掉了探索阶段用不到的代码分支。

## 目录结构

```
dcdsm-cross-attn/
├── README.md                本文件
├── requirements.txt         Python 依赖(见文件内注释,强烈建议 pip freeze 锁定精确版本)
├── .gitignore
├── data/                    放训练/测试 CSV 的地方(不提交到git,见 data/README.md)
│   └── README.md
├── outputs/                 训练/评估产出:权重、曲线图、OOF、预测结果(不提交到git)
├── src/                     核心代码,一个 Python package
│   ├── __init__.py
│   ├── config.py            全局配置:路径、随机种子、模型结构与训练超参数
│   ├── fusion.py            融合模块:方案D 双向交叉注意力 + 门控残差
│   ├── losses.py            损失函数:BCE + Tversky 混合损失
│   ├── model.py             组装完整模型(CNN-BiLSTM编码器 + 融合层 + 分类头)
│   ├── data_utils.py        数据加载、标准化、维度调整
│   ├── train.py             训练主逻辑(10折交叉验证 + 测试集评估)
│   ├── evaluate.py          在(自带的或全新外部的)测试集上加载权重做评估
│   └── compat.py            h5py 新旧版本兼容性补丁(仅特定环境需要,见文件内说明)
└── scripts/                 命令行入口(负责把仓库根目录加进 sys.path)
    ├── run_train.py
    └── run_evaluate.py
```

## 环境安装

```bash
conda create -n dcdsm python=3.9   # Python版本按你训练时用的来定,3.9仅作示例
conda activate dcdsm
pip install -r requirements.txt
```

> `tf.keras.layers.MultiHeadAttention` 需要 TensorFlow >= 2.4,融合层
> (方案D)用到了这个层,如果环境里 TensorFlow 版本太旧会在建模型这一步
> 直接报 `AttributeError: module '...keras...layers' has no attribute
> 'MultiHeadAttention'`,这是环境问题,和代码无关,升级 TensorFlow 版本
> 或换到能跑这个模型的环境即可。

## 数据准备

见 [`data/README.md`](data/README.md):把 4 个 CSV(训练/测试各一份手工特征
+ 一份语言模型特征)放进 `data/` 目录,文件名要对上。

## 训练模型

```bash
python scripts/run_train.py
```

会执行:10折分层交叉验证(`StratifiedKFold`,`random_state` 固定为
`src/config.py` 里的 `RANDOM_SEED=42`)→ 每折训练模型、记录 Acc/AUC/AUPRC/
Sn/Sp/MCC、保存该折权重和loss曲线 → 按验证集 accuracy 挑最佳折 → 用最佳
折权重在测试集上算最终指标、画 ROC/PR 曲线、保存逐样本预测结果。

所有产出默认写到 `outputs/`(文件名前缀是 `config.py` 里的 `MODEL_TAG =
"crossattn_bce_tversky"`):

| 文件 | 内容 |
|---|---|
| `crossattn_bce_tversky_fold{N}.weights.h5` | 每一折的模型权重 |
| `crossattn_bce_tversky_loss_fold{N}.png` | 每一折的训练/验证loss曲线 |
| `crossattn_bce_tversky_oof.npy` | 10折交叉验证的 out-of-fold 预测概率 |
| `crossattn_bce_tversky_best.weights.h5` | 验证集accuracy最高那一折的权重(最终模型) |
| `crossattn_bce_tversky_roc_test.png` / `_pr_test.png` | 测试集 ROC / PR 曲线 |
| `crossattn_bce_tversky_test_predictions.csv` | 测试集逐样本预测结果 |

## 在新的外部测试集上评估

```bash
python scripts/run_evaluate.py \
    --test-manual data/新测试集_手工特征.csv \
    --test-bert data/新测试集_语言模型特征.csv \
    --weights outputs/crossattn_bce_tversky_best.weights.h5
```

不加任何参数时,默认用 `data/README.md` 里说明的自带测试集 + 训练产出的
最佳权重。

**这里有两个很容易踩的坑,脚本已经处理好了,但了解原理有助于你以后复用:**

1. **标准化基准必须来自训练集,不能在新测试集上重新 fit。** 脚本会重新
   加载 `train_*.csv` 来重建 `StandardScaler`(这一步是确定性的,只要训练
   CSV 内容不变,结果和当初训练时完全一致),再对新测试集做 `transform`。
   如果直接在新数据上 `fit_transform`,标准化基准就错了,所有指标都会失真。
2. **权重文件只存了权重,没存网络结构。** 所以评估脚本用 `src/model.py`
   里和训练时完全一致的 `create_cnn_model()` 重建结构后再 `load_weights`,
   保证结构对得上。

## 复现性说明

- 所有随机源(`numpy`/`python random`/`tensorflow`/`StratifiedKFold`/
  训练集shuffle)都用同一个 `RANDOM_SEED=42`(见 `src/config.py`)。
- 决策阈值固定为 0.5,不做阈值搜索,训练和评估全程一致。
- **权重文件是 weight-only checkpoint,不含网络结构**,所以复现时必须用
  这个仓库里 `src/model.py` 定义的完全相同的结构去 `load_weights`,且强烈
  建议锁定和训练时完全一致的 TensorFlow 版本(见 `requirements.txt` 里的
  说明和 `pip freeze` 建议)。跨大版本的 TensorFlow(比如从2.0跳到2.4+)
  在权重文件的 HDF5 读取细节上可能不兼容,遇到
  `AttributeError: 'str' object has no attribute 'decode'` 这类报错时,
  可以参考 `src/compat.py` 里的说明和修复方法。

## 方法说明

**融合层(方案D: 双向交叉注意力 + 门控残差)** —— 语言模型特征和手工特征
各自先过 CNN-BiLSTM 编码,再互相做双向 cross-attention(语言模型→手工、
手工→语言模型各一个方向),每个方向都带残差+LayerNorm,最后用一个可学习
的门控向量加权合并两个方向的输出。参考: Tsai et al., ACL 2019 (MulT);
Nagrani et al., NeurIPS 2021。

原始基线代码在 2D 特征上直接用 `tf.keras.layers.Attention()`,这个层期望
3D 输入,2D 输入下会退化成"样本之间"做注意力(batch内信息泄漏、结果依赖
batch大小和样本顺序、无法单样本推理),且没有可学习的 Q/K/V 投影。方案D
修复了这个问题,并保留了 cross-attention 的设计意图。

**损失函数(BCE + Tversky 混合)** —— 纯 BCE 基线下 Sensitivity 很高、
Specificity 很低;换成纯 Tversky Loss(`TP / (TP + α·FP + β·FN)`,
α=0.7, β=0.3,加大对假阳性的惩罚)后 Sn/Sp 平衡了,但 AUC/AUPRC 掉得比较
多,原因是纯 Tversky 只关心固定阈值附近的 TP/FP/FN 重叠区域,不约束整体
排序质量。混合两者(`0.3·BCE + 0.7·Tversky`,类似 nnU-Net 等医学图像分割
工作里常见的组合损失思路)用来兼顾两者。

## 参考文献

- Tsai, Y.-H. H., et al. "Multimodal Transformer for Unaligned Multimodal
  Language Sequences." ACL 2019. (MulT, 交叉注意力)
- Nagrani, A., et al. "Attention Bottlenecks for Multimodal Fusion."
  NeurIPS 2021.
- Salehi, S. S. M., et al. "Tversky loss function for image segmentation
  using 3D fully convolutional deep networks." MICCAI-DLMIA 2017.
