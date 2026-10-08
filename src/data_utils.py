"""数据加载与预处理: 读取CSV、标准化、维度调整。"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from . import config


def load_and_preprocess(random_seed=None):
    """加载训练/测试集,做标准化和shuffle,返回模型可直接使用的张量字典。

    注意:
      - 只对训练集做 shuffle(同步打乱手工特征/语言模型特征/标签三者的样本
        顺序,三者用同一个 random_state 保证仍然一一对应),测试集保持CSV
        原始顺序 —— 融合层修复后不再存在跨样本依赖,打乱测试集顺序不影响
        任何指标,反而不利于结果复核。
      - StandardScaler 只在训练集上 fit,测试集只 transform,避免信息泄漏。
      - 同时返回 fit 好的 scaler 对象,方便之后在全新的外部测试集上复用
        同一套标准化基准(不能在新数据上重新 fit,否则标准化基准错了,
        所有指标都会失真,见 evaluate.py 里的用法)。

    返回:
        dict,包含:
          x_train / x_test: {'bert_features':..., 'manual_features':...}
          y_train / y_test: 标签数组
          manual_feature_dim / bert_feature_dim: 特征维度
          manual_feature_cols / bert_feature_cols: 特征列名列表(按训练集
            列顺序),在新测试集上取值时要按这个顺序,并检查列是否齐全。
          scaler_manual / scaler_bert: 训练集上 fit 好的 StandardScaler
    """
    random_seed = config.RANDOM_SEED if random_seed is None else random_seed

    train_manual_df = pd.read_csv(config.TRAIN_MANUAL_CSV)
    test_manual_df = pd.read_csv(config.TEST_MANUAL_CSV)
    train_bert_df = pd.read_csv(config.TRAIN_BERT_CSV)
    test_bert_df = pd.read_csv(config.TEST_BERT_CSV)

    y_train = train_manual_df['LABEL'].values
    y_test = test_manual_df['LABEL'].values

    manual_feature_cols = [c for c in train_manual_df.columns if c not in ['sequence_id', 'LABEL']]
    bert_feature_cols = [c for c in train_bert_df.columns if c not in ['sequence_id', 'LABEL']]

    x_train_manual = train_manual_df[manual_feature_cols].values
    x_test_manual = test_manual_df[manual_feature_cols].values
    x_train_bert = train_bert_df[bert_feature_cols].values
    x_test_bert = test_bert_df[bert_feature_cols].values

    print(f"手工特征形状 - Train: {x_train_manual.shape}, Test: {x_test_manual.shape}")
    print(f"语言模型特征形状 - Train: {x_train_bert.shape}, Test: {x_test_bert.shape}")

    x_train_manual, x_train_bert, y_train = shuffle(
        x_train_manual, x_train_bert, y_train, random_state=random_seed
    )

    scaler_manual = StandardScaler()
    x_train_manual = scaler_manual.fit_transform(x_train_manual)
    x_test_manual = scaler_manual.transform(x_test_manual)

    scaler_bert = StandardScaler()
    x_train_bert = scaler_bert.fit_transform(x_train_bert)
    x_test_bert = scaler_bert.transform(x_test_bert)

    manual_feature_dim = x_train_manual.shape[1]
    bert_feature_dim = x_train_bert.shape[1]

    x_train_bert = np.expand_dims(x_train_bert, axis=2)
    x_test_bert = np.expand_dims(x_test_bert, axis=2)
    x_train_manual = np.expand_dims(x_train_manual, axis=2)
    x_test_manual = np.expand_dims(x_test_manual, axis=2)

    print("标准化和维度调整后:")
    print(f"  手工特征 - Train: {x_train_manual.shape}, Test: {x_test_manual.shape}")
    print(f"  语言模型特征 - Train: {x_train_bert.shape}, Test: {x_test_bert.shape}")

    return {
        'x_train': {'bert_features': x_train_bert, 'manual_features': x_train_manual},
        'x_test': {'bert_features': x_test_bert, 'manual_features': x_test_manual},
        'y_train': y_train,
        'y_test': y_test,
        'manual_feature_dim': manual_feature_dim,
        'bert_feature_dim': bert_feature_dim,
        'manual_feature_cols': manual_feature_cols,
        'bert_feature_cols': bert_feature_cols,
        'scaler_manual': scaler_manual,
        'scaler_bert': scaler_bert,
    }
