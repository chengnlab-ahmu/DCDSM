"""损失函数: BCE + Tversky 混合损失。

背景: 纯 BCE(baseline)下 Sensitivity 很高、Specificity 很低; 换成纯
Tversky Loss 后 Sn/Sp 平衡了,但 AUC/AUPRC 掉得比较多。混合两者(类似
nnU-Net 等医学图像分割工作里常见的组合损失思路)用来兼顾"固定阈值0.5下
的Sn/Sp平衡"和"整体概率排序质量(AUC/AUPRC)"。

探索阶段还单独实现过一版可独立调用的纯 tversky_loss(),作为单独一档
实验方案对比过;既然最终选型是"混合损失",这里就不再单独暴露纯 Tversky
版本的公开接口,把它的计算逻辑直接内联进 bce_tversky_loss() 里
(数学上完全等价,只是不再维护一个用不到的独立函数),避免仓库里留着
无关代码。
"""
import tensorflow as tf


def bce_tversky_loss(bce_weight=0.3, alpha=0.7, beta=0.3, smooth=1e-6):
    """返回一个可直接传给 model.compile(loss=...) 的损失函数。

    loss = bce_weight * BCE + (1 - bce_weight) * (1 - Tversky_index)
    Tversky_index = TP / (TP + alpha*FP + beta*FN)

    参数:
        bce_weight: BCE 在总loss里的占比(0~1)。越接近1越偏向排序质量
            (AUC/AUPRC),越接近0越偏向固定阈值0.5下的Sp提升。
        alpha: 假阳性(FP)的惩罚权重,越大越保守、越不容易把负样本判成
            正类(提升Sp)。
        beta:  假阴性(FN)的惩罚权重,越大越倾向多抓正类(提升Sn)。
        smooth: 防止除零的平滑项。
    """
    bce_fn = tf.keras.losses.BinaryCrossentropy()

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])

        tp = tf.reduce_sum(y_true_flat * y_pred_flat)
        fp = tf.reduce_sum((1.0 - y_true_flat) * y_pred_flat)
        fn = tf.reduce_sum(y_true_flat * (1.0 - y_pred_flat))

        tversky_index = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
        tversky_loss_value = 1.0 - tversky_index

        return (bce_weight * bce_fn(y_true, y_pred)
                + (1.0 - bce_weight) * tversky_loss_value)

    return loss
