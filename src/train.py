"""训练脚本: 10折交叉验证 + 测试集最终评估。

对应原实验记录里的 EXPERIMENT=2.5(方案D融合 + BCE/Tversky混合损失)。
原脚本里还有 EXPERIMENT=0/1/2/3 几个对照实验分支(纯baseline、按MCC选
checkpoint、纯Tversky loss、难负样本挖掘的sample_weight),既然最终选型
已经确定为 2.5,这里只保留这一条路径,不再包含 if/elif 开关和不会走到
的其它分支,方便阅读和维护。如需要回顾对比实验的做法,请看 git 历史。

用法:
    python scripts/run_train.py
或者从仓库根目录:
    python -m src.train
"""
import os
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 无图形界面环境下也能跑(比如服务器/CI),只保存图片不弹窗
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_curve, roc_auc_score, confusion_matrix,
                             accuracy_score, f1_score, matthews_corrcoef,
                             precision_recall_curve, auc)

from . import config
from .data_utils import load_and_preprocess
from .model import create_cnn_model
from .fusion import check_no_batch_leakage


def set_seeds(seed):
    """固定所有随机源,保证结果可复现。"""
    np.random.seed(seed)
    random.seed(seed)
    try:
        tf.random.set_seed(seed)  # TensorFlow 2.x
    except AttributeError:
        tf.set_random_seed(seed)  # TensorFlow 1.x
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"随机种子已设置: {seed}")


def main():
    set_seeds(config.RANDOM_SEED)
    print(f"融合方案: 方案D 双向交叉注意力 + 门控残差")
    print(f"损失函数: BCE({config.BCE_WEIGHT}) + Tversky(alpha={config.TVERSKY_ALPHA}, beta={config.TVERSKY_BETA})")

    print("\n加载数据中...")
    data = load_and_preprocess()
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']
    manual_feature_dim = data['manual_feature_dim']
    bert_feature_dim = data['bert_feature_dim']

    # ===================== 10折交叉验证训练 =====================
    print("\n" + "=" * 60)
    print("开始10折交叉验证训练 —— 融合方案: 方案D 双向交叉注意力 + 门控残差")
    print("=" * 60)

    skf = StratifiedKFold(n_splits=config.NUM_FOLDS, shuffle=True,
                          random_state=config.RANDOM_SEED)

    best_accuracy = 0.0
    best_model = None
    best_params = None
    best_fold = None
    fold_accuracies, fold_aucs, fold_auprcs = [], [], []
    fold_sensitivities, fold_specificities, fold_mccs = [], [], []
    oof_proba = np.zeros(len(y_train), dtype=np.float64)

    for fold, (train_idx, val_idx) in enumerate(skf.split(x_train['bert_features'], y_train)):
        print(f"\n{'=' * 60}")
        print(f"Fold {fold + 1}/{config.NUM_FOLDS}")
        print(f"{'=' * 60}")

        x_train_fold = {
            'bert_features': x_train['bert_features'][train_idx],
            'manual_features': x_train['manual_features'][train_idx],
        }
        x_val_fold = {
            'bert_features': x_train['bert_features'][val_idx],
            'manual_features': x_train['manual_features'][val_idx],
        }
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        model = create_cnn_model(bert_feature_dim, manual_feature_dim)
        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=config.EARLY_STOPPING_PATIENCE,
                                       restore_best_weights=True)

        history = model.fit(
            x_train_fold, y_train_fold,
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            validation_data=(x_val_fold, y_val_fold),
            callbacks=[early_stopping],
            verbose=1,
        )

        val_proba = model.predict(x_val_fold, verbose=0)
        val_predictions = (val_proba > config.DECISION_THRESHOLD).astype(int)
        oof_proba[val_idx] = val_proba.ravel()

        val_accuracy = accuracy_score(y_val_fold, val_predictions)
        val_auc = roc_auc_score(y_val_fold, val_proba)

        precision, recall, _ = precision_recall_curve(y_val_fold, val_proba)
        val_auprc = auc(recall, precision)

        fold_accuracies.append(val_accuracy)
        fold_aucs.append(val_auc)
        fold_auprcs.append(val_auprc)

        conf_matrix = confusion_matrix(y_val_fold, val_predictions)
        tn, fp, fn, tp = conf_matrix.ravel()
        sp = tn / (tn + fp) if (tn + fp) > 0 else 0
        sn = tp / (tp + fn) if (tp + fn) > 0 else 0
        fold_specificities.append(sp)
        fold_sensitivities.append(sn)

        mcc = matthews_corrcoef(y_val_fold, val_predictions)
        fold_mccs.append(mcc)

        print(f"Fold {fold + 1} Validation - Acc: {val_accuracy:.4f}, AUC: {val_auc:.4f}, "
              f"AUPRC: {val_auprc:.4f}, Sn: {sn:.4f}, Sp: {sp:.4f}, MCC: {mcc:.4f}")

        # 损失曲线
        plt.figure(figsize=(10, 4))
        plt.plot(history.history['loss'], label='train_loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.title(f'Fold {fold + 1} - Model Loss ({config.MODEL_TAG})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(config.OUTPUT_DIR / f'{config.MODEL_TAG}_loss_fold{fold + 1}.png',
                   dpi=100, bbox_inches='tight')
        plt.close()

        # 每一折的权重都落盘(如果之后想做多折集成预测,可以直接复用)
        model.save_weights(str(config.OUTPUT_DIR / f'{config.MODEL_TAG}_fold{fold + 1}.weights.h5'))

        # 按验证集 accuracy 挑最佳 fold
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model = model
            best_params = model.get_weights()
            best_fold = fold + 1
            print(f"✓ 新的最佳模型 (Accuracy: {best_accuracy:.4f})")

    np.save(str(config.OUTPUT_DIR / f'{config.MODEL_TAG}_oof.npy'), oof_proba)
    print(f"\n本次训练OOF已保存至: {config.MODEL_TAG}_oof.npy")

    # ===================== 交叉验证结果汇总 =====================
    print(f"\n{'=' * 60}")
    print(f"10折交叉验证结果 —— 融合方案: 方案D 双向交叉注意力 + 门控残差")
    print(f"{'=' * 60}")
    for i in range(config.NUM_FOLDS):
        print(f"Fold {i + 1:2d}: Acc={fold_accuracies[i]:.4f}, AUC={fold_aucs[i]:.4f}, "
              f"AUPRC={fold_auprcs[i]:.4f}, Sn={fold_sensitivities[i]:.4f}, Sp={fold_specificities[i]:.4f}")

    print(f"\n平均验证结果:")
    print(f"  Accuracy:    {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
    print(f"  AUC:         {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print(f"  AUPRC:       {np.mean(fold_auprcs):.4f} ± {np.std(fold_auprcs):.4f}")
    print(f"  Sensitivity: {np.mean(fold_sensitivities):.4f} ± {np.std(fold_sensitivities):.4f}")
    print(f"  Specificity: {np.mean(fold_specificities):.4f} ± {np.std(fold_specificities):.4f}")
    print(f"  MCC:         {np.mean(fold_mccs):.4f} ± {np.std(fold_mccs):.4f}")

    # ===================== 测试集最终评估 =====================
    print(f"\n{'=' * 60}")
    print(f"最佳模型测试集结果 —— 来自 Fold {best_fold}")
    print(f"{'=' * 60}")

    best_model.set_weights(best_params)
    best_weights_path = config.OUTPUT_DIR / f'{config.MODEL_TAG}_best.weights.h5'
    best_model.save_weights(str(best_weights_path))
    print(f"最佳模型权重已保存至: {best_weights_path}")

    check_no_batch_leakage(best_model, x_test)

    test_proba = best_model.predict(x_test, verbose=0)
    test_predictions = (test_proba > config.DECISION_THRESHOLD).astype(int)

    test_accuracy = accuracy_score(y_test, test_predictions)
    test_auc = roc_auc_score(y_test, test_proba)
    test_f1 = f1_score(y_test, test_predictions)

    precision_test, recall_test, _ = precision_recall_curve(y_test, test_proba)
    test_auprc = auc(recall_test, precision_test)

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test AUC:      {test_auc:.4f}")
    print(f"Test AUPRC:    {test_auprc:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")

    cm = confusion_matrix(y_test, test_predictions)
    tn, fp, fn, tp = cm.ravel()
    sp = tn / (tn + fp)
    sn = tp / (tp + fn)
    precision = tp / (tp + fp)
    mcc = matthews_corrcoef(y_test, test_predictions)

    print(f"\n混淆矩阵:\n{cm}")
    print(f"\n详细指标:")
    print(f"  Sensitivity (Sn/Recall): {sn:.4f}")
    print(f"  Specificity (Sp):        {sp:.4f}")
    print(f"  Precision:               {precision:.4f}")
    print(f"  MCC:                     {mcc:.4f}")

    # ROC 曲线
    fpr, tpr, _ = roc_curve(y_test, test_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {test_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Test Set ({config.MODEL_TAG})')
    plt.legend(loc="lower right"); plt.grid(True, alpha=0.3)
    plt.savefig(config.OUTPUT_DIR / f'{config.MODEL_TAG}_roc_test.png', dpi=100, bbox_inches='tight')
    plt.close()

    # PR 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(recall_test, precision_test, color='blue', lw=2,
             label=f'PR curve (AUPRC = {test_auprc:.4f})')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - Test Set ({config.MODEL_TAG})')
    plt.legend(loc="lower left"); plt.grid(True, alpha=0.3)
    plt.savefig(config.OUTPUT_DIR / f'{config.MODEL_TAG}_pr_test.png', dpi=100, bbox_inches='tight')
    plt.close()

    # 逐样本测试集预测结果,方便复核/画图
    test_manual_df_seq = None
    try:
        test_manual_df_seq = pd.read_csv(config.TEST_MANUAL_CSV)['sequence_id']
    except Exception:
        pass
    out_df = pd.DataFrame({
        'sequence_id': test_manual_df_seq if test_manual_df_seq is not None else np.arange(len(y_test)),
        'y_true': y_test,
        'y_proba': test_proba.ravel(),
        'y_pred': test_predictions.ravel(),
    })
    out_df.to_csv(str(config.OUTPUT_DIR / f'{config.MODEL_TAG}_test_predictions.csv'), index=False)

    print(f"\n✓ 训练完成！融合方案: 方案D 双向交叉注意力 + 门控残差")
    print(f"  所有产出文件在: {config.OUTPUT_DIR}")


if __name__ == '__main__':
    main()
