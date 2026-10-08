"""在测试集上加载已训练权重并评估指标。

默认用 config.py 里配置的测试集 + 训练脚本产出的最佳权重;也可以通过
命令行参数指向一份全新的外部测试集(比如另一批临床样本),用法见下方
"外部测试集" 示例。

用法:
    # 默认: 用仓库自带的测试集 + 训练脚本产出的最佳权重
    python scripts/run_evaluate.py

    # 外部测试集: 指定一份全新的测试集 + 权重文件
    python scripts/run_evaluate.py \\
        --test-manual data/new_test_manual.csv \\
        --test-bert data/new_test_bert.csv \\
        --weights outputs/crossattn_bce_tversky_best.weights.h5

关键点(和调试外部测试集时的经验一致,这里已经内置处理好了):
  1. 标准化用的 StandardScaler 是重新在"原始训练集"上 fit 的,不是在
     新测试集上 fit —— 这是保证指标真实有效的关键,绝不能省略。
  2. 权重文件(.weights.h5)只存了权重、没存网络结构,所以这里复用
     src/model.py 里和训练时完全一致的 create_cnn_model() 重建结构后
     再 load_weights,保证结构对得上。
  3. 如果在比训练时更旧的 TensorFlow 环境里加载权重,报
     `AttributeError: 'str' object has no attribute 'decode'`,是
     h5py 新旧版本兼容性问题,和权重本身无关,可以取消下面
     `patch_h5py_for_legacy_tf()` 那行的注释来修复,见 src/compat.py。
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_curve, roc_auc_score, confusion_matrix,
                             accuracy_score, f1_score, matthews_corrcoef,
                             precision_recall_curve, auc, classification_report)

from . import config
from .model import create_cnn_model
from .fusion import check_no_batch_leakage
# 如果需要在旧版TF环境里加载权重,取消下面这行注释:
# from .compat import patch_h5py_for_legacy_tf


def parse_args():
    parser = argparse.ArgumentParser(description="在测试集上评估已训练模型")
    parser.add_argument('--test-manual', default=str(config.TEST_MANUAL_CSV),
                       help="测试集手工特征CSV路径,默认用 config.py 里的自带测试集")
    parser.add_argument('--test-bert', default=str(config.TEST_BERT_CSV),
                       help="测试集语言模型特征CSV路径,默认用 config.py 里的自带测试集")
    parser.add_argument('--weights',
                       default=str(config.OUTPUT_DIR / f'{config.MODEL_TAG}_best.weights.h5'),
                       help="模型权重文件路径(.weights.h5)")
    parser.add_argument('--threshold', type=float, default=config.DECISION_THRESHOLD,
                       help="分类决策阈值,默认与训练/评估协议一致(0.5)")
    return parser.parse_args()


def main():
    args = parse_args()
    # patch_h5py_for_legacy_tf()  # 见上方说明,按需取消注释

    print("加载训练集(用于重建 scaler 与特征列定义,评估外部测试集时不能省略这一步)...")
    train_manual_df = pd.read_csv(config.TRAIN_MANUAL_CSV)
    train_bert_df = pd.read_csv(config.TRAIN_BERT_CSV)

    manual_feature_cols = [c for c in train_manual_df.columns if c not in ['sequence_id', 'LABEL']]
    bert_feature_cols = [c for c in train_bert_df.columns if c not in ['sequence_id', 'LABEL']]
    manual_feature_dim = len(manual_feature_cols)
    bert_feature_dim = len(bert_feature_cols)
    print(f"训练集特征维度 - 手工特征: {manual_feature_dim}, 语言模型特征: {bert_feature_dim}")

    scaler_manual = StandardScaler().fit(train_manual_df[manual_feature_cols].values)
    scaler_bert = StandardScaler().fit(train_bert_df[bert_feature_cols].values)

    print(f"\n加载测试集:\n  手工特征: {args.test_manual}\n  语言模型特征: {args.test_bert}")
    test_manual_df = pd.read_csv(args.test_manual)
    test_bert_df = pd.read_csv(args.test_bert)

    if 'sequence_id' in test_manual_df.columns and 'sequence_id' in test_bert_df.columns:
        test_manual_df = test_manual_df.sort_values('sequence_id').reset_index(drop=True)
        test_bert_df = test_bert_df.sort_values('sequence_id').reset_index(drop=True)
        if not (test_manual_df['sequence_id'].values == test_bert_df['sequence_id'].values).all():
            raise ValueError(
                "两个测试集文件按 sequence_id 排序后仍不能一一对应,"
                "请检查是否是同一批样本、有没有缺失/重复的 sequence_id。"
            )
    else:
        print("警告: 未找到 sequence_id 列,假设两个测试文件的行是按顺序一一对应的,"
              "请自行确认这一点,否则手工特征和语言模型特征会错位。")

    missing_manual = set(manual_feature_cols) - set(test_manual_df.columns)
    missing_bert = set(bert_feature_cols) - set(test_bert_df.columns)
    if missing_manual:
        raise ValueError(f"测试集手工特征缺少训练时使用的列: {sorted(missing_manual)}")
    if missing_bert:
        raise ValueError(f"测试集语言模型特征缺少训练时使用的列: {sorted(missing_bert)}")

    y_test = test_manual_df['LABEL'].values
    if 'LABEL' in test_bert_df.columns:
        if not np.array_equal(y_test, test_bert_df['LABEL'].values):
            raise ValueError("两个测试集文件里的 LABEL 不一致,请检查数据对齐是否正确。")

    x_test_manual = scaler_manual.transform(test_manual_df[manual_feature_cols].values)
    x_test_bert = scaler_bert.transform(test_bert_df[bert_feature_cols].values)
    x_test_manual = np.expand_dims(x_test_manual, axis=2)
    x_test_bert = np.expand_dims(x_test_bert, axis=2)
    x_test_dict = {'bert_features': x_test_bert, 'manual_features': x_test_manual}
    print(f"测试集样本数: {len(y_test)}, 正样本占比: {y_test.mean():.4f}")

    print(f"\n重建模型结构并加载权重: {args.weights}")
    model = create_cnn_model(bert_feature_dim, manual_feature_dim)
    model.load_weights(args.weights)

    check_no_batch_leakage(model, x_test_dict)

    test_proba = model.predict(x_test_dict, verbose=0).ravel()
    test_predictions = (test_proba > args.threshold).astype(int)

    test_accuracy = accuracy_score(y_test, test_predictions)
    test_auc = roc_auc_score(y_test, test_proba)
    test_f1 = f1_score(y_test, test_predictions)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, test_proba)
    test_auprc = auc(recall_curve, precision_curve)

    cm = confusion_matrix(y_test, test_predictions)
    tn, fp, fn, tp = cm.ravel()
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    mcc = matthews_corrcoef(y_test, test_predictions)

    print(f"\n{'=' * 60}")
    print(f"测试集评估结果 —— {config.MODEL_TAG} (决策阈值 {args.threshold})")
    print(f"{'=' * 60}")
    print(f"Accuracy:    {test_accuracy:.4f}")
    print(f"AUC:         {test_auc:.4f}")
    print(f"AUPRC:       {test_auprc:.4f}")
    print(f"F1-Score:    {test_f1:.4f}")
    print(f"Sensitivity: {sn:.4f}")
    print(f"Specificity: {sp:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"MCC:         {mcc:.4f}")
    print(f"\n混淆矩阵 (行=真实, 列=预测):\n{cm}")
    print(f"\n{classification_report(y_test, test_predictions, digits=4)}")

    fpr, tpr, _ = roc_curve(y_test, test_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {test_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {config.MODEL_TAG}')
    plt.legend(loc="lower right"); plt.grid(True, alpha=0.3)
    plt.savefig(str(config.OUTPUT_DIR / f'{config.MODEL_TAG}_eval_roc.png'), dpi=100, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, color='blue', lw=2,
             label=f'PR curve (AUPRC = {test_auprc:.4f})')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {config.MODEL_TAG}')
    plt.legend(loc="lower left"); plt.grid(True, alpha=0.3)
    plt.savefig(str(config.OUTPUT_DIR / f'{config.MODEL_TAG}_eval_pr.png'), dpi=100, bbox_inches='tight')
    plt.close()

    out_df = pd.DataFrame({
        'sequence_id': test_manual_df['sequence_id'] if 'sequence_id' in test_manual_df.columns
                      else np.arange(len(y_test)),
        'y_true': y_test,
        'y_proba': test_proba,
        'y_pred': test_predictions,
    })
    out_path = config.OUTPUT_DIR / f'{config.MODEL_TAG}_eval_predictions.csv'
    out_df.to_csv(str(out_path), index=False)
    print(f"\n逐样本预测结果已保存至: {out_path}")
    print(f"ROC/PR 曲线已保存至: {config.OUTPUT_DIR}")


if __name__ == '__main__':
    main()
