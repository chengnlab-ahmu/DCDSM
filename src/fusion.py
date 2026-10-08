"""融合模块:方案D 双向交叉注意力 + 门控残差。

参考文献: Tsai et al., ACL 2019 (MulT); Nagrani et al., NeurIPS 2021

原代码融合层问题:
    merged = tf.keras.layers.concatenate([bert_output, manual_output])  # (batch,256), 2D
    attention_layer = tf.keras.layers.Attention()([merged, merged, merged])
    tf.keras.layers.Attention 期望 3D 输入 (batch, T, dim)。传入 2D 时会退化为
    "样本之间"做注意力 (matmul(Q, K^T) -> (batch, batch)),造成 batch 内样本
    信息泄漏、结果依赖 batch 大小与样本顺序、无法单样本推理,且无可学习的
    W_Q/W_K/W_V,与论文公式描述不符。

探索阶段还实现、比较过 token 级自注意力(方案A)/GMU(方案B)/SE通道重标定
(方案C)/直接拼接(基线)等融合方式,最终选定本文件里的方案D。仓库只保留
实际用在最终模型里的这一种实现,减少无关代码;如果需要回顾其它方案的
对比结果,请看实验记录或 git 历史。
"""
import tensorflow as tf
from tensorflow.keras import layers


def fusion_bidirectional_cross_attention(bert_output, manual_output,
                                          d_model=128, num_heads=4, dropout=0.1):
    """双向交叉注意力,每个方向都带门控残差,避免削弱通道内语义。

    bert_output, manual_output: 均为 (batch, 128) 的 2D 张量
    返回: (batch, d_model) 的 2D 融合表示,可直接接 MLP 分类头。
    """
    b = layers.Reshape((1, -1))(layers.Dense(d_model, name='xa_proj_b')(bert_output))
    m = layers.Reshape((1, -1))(layers.Dense(d_model, name='xa_proj_m')(manual_output))

    b2m = layers.MultiHeadAttention(num_heads, d_model // num_heads,
                                    dropout=dropout, name='xa_b2m')(b, m, m)
    m2b = layers.MultiHeadAttention(num_heads, d_model // num_heads,
                                    dropout=dropout, name='xa_m2b')(m, b, b)

    b_out = layers.LayerNormalization(epsilon=1e-6)(layers.Add()([b, b2m]))
    m_out = layers.LayerNormalization(epsilon=1e-6)(layers.Add()([m, m2b]))

    b_out = layers.Flatten()(b_out)
    m_out = layers.Flatten()(m_out)

    # 门控加权合并两个方向
    gate = layers.Dense(d_model, activation='sigmoid',
                        name='xa_gate')(layers.Concatenate()([b_out, m_out]))
    one_minus = layers.Lambda(lambda t: 1.0 - t)(gate)
    fused = layers.Add(name='fused_repr')([
        layers.Multiply()([gate, b_out]),
        layers.Multiply()([one_minus, m_out]),
    ])
    return fused


def check_no_batch_leakage(model, x_dict, atol=1e-5):
    """自检: 验证融合层不再跨样本泄漏。

    同一个样本,单独预测 vs 放在 batch 里预测,结果必须一致。
    原来退化成"样本间注意力"的写法会在这里失败,方案D修复后应通过。
    """
    import numpy as np
    full = model.predict(x_dict, batch_size=32, verbose=0)
    single = model.predict(
        {k: v[:1] for k, v in x_dict.items()}, batch_size=1, verbose=0)
    ok = np.allclose(full[0], single[0], atol=atol)
    print(f"[batch 泄漏自检] {'通过' if ok else '未通过'}: "
          f"batch内={full[0][0]:.6f}, 单样本={single[0][0]:.6f}")
    return ok
