"""模型结构: 双分支 CNN-BiLSTM 编码器 + 方案D融合层 + MLP分类头。

CNN-BiLSTM 编码器与 MLP 分类头的结构与原始代码完全一致,唯一改动是把
融合层换成了 fusion.fusion_bidirectional_cross_attention (方案D),
并把损失函数换成 losses.bce_tversky_loss (BCE+Tversky 混合)。
"""
import tensorflow as tf
from tensorflow.keras import layers

from . import config
from .fusion import fusion_bidirectional_cross_attention
from .losses import bce_tversky_loss


def create_cnn_model(bert_feature_dim, manual_feature_dim):
    """构建多模态深度学习模型。

    - 输入1 'bert_features': 语言模型特征, shape (N, bert_feature_dim, 1)
    - 输入2 'manual_features': 手工特征, shape (N, manual_feature_dim, 1)
    - 融合层: 方案D 双向交叉注意力 + 门控残差
    - 损失: BCE + Tversky 混合损失
    """
    # ---------- 语言模型分支 ----------
    input_bert = tf.keras.Input(shape=(bert_feature_dim, 1), name='bert_features')
    y = layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(input_bert)
    y = layers.MaxPooling1D(pool_size=2)(y)
    y = layers.Bidirectional(layers.LSTM(units=128, return_sequences=True))(y)
    y = layers.Bidirectional(layers.LSTM(units=64, return_sequences=False))(y)
    y = layers.Dropout(0.1)(y)
    bert_output = layers.Dense(128, activation='relu', name='bert_encoding')(y)

    # ---------- 手工特征分支 ----------
    input_manual = tf.keras.Input(shape=(manual_feature_dim, 1), name='manual_features')
    x = layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(input_manual)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Bidirectional(layers.LSTM(units=128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(units=64, return_sequences=False))(x)
    x = layers.Dropout(0.1)(x)
    manual_output = layers.Dense(128, activation='relu', name='manual_encoding')(x)

    # ---------- 融合层 (方案D) ----------
    fused = fusion_bidirectional_cross_attention(
        bert_output, manual_output,
        d_model=config.FUSION_D_MODEL,
        num_heads=config.FUSION_NUM_HEADS,
        dropout=config.FUSION_DROPOUT,
    )

    # ---------- 分类头 ----------
    d = layers.Dense(256, activation='relu')(fused)
    d = layers.Dropout(0.1)(d)
    d = layers.Dense(128, activation='relu')(d)
    output = layers.Dense(1, activation='sigmoid')(d)

    model = tf.keras.Model([input_bert, input_manual], output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=bce_tversky_loss(
            bce_weight=config.BCE_WEIGHT,
            alpha=config.TVERSKY_ALPHA,
            beta=config.TVERSKY_BETA,
        ),
        metrics=['accuracy'],
    )
    return model
