"""Count the pseudo-labels selected by each real training batch."""

import numpy as np

from emotion_radius import NEGATIVE_CLASSES, POSITIVE_CLASSES


CLASS_NAMES = ('惊讶', '恐惧', '厌恶', '开心', '悲伤', '愤怒', '中性')
STAGES = {
    'before': '筛选前',
    'agreement': '双视图预测一致',
    'confidence': '高置信候选',
    'confidence_knn': '高置信路径通过融合 kNN',
    'low_confidence': '双视图一致的低置信样本',
    'radius_candidates': '通过分组半径',
    'radius_final': '通过半径及严格双域 kNN',
    'final': '最终接纳',
    'rejected': '最终未接纳',
}


class TrainingPseudoLabelAudit:
    def __init__(self):
        self.counts = {stage: np.zeros(len(CLASS_NAMES), dtype=np.int64) for stage in STAGES}
        self.correct = {stage: np.zeros_like(counts) for stage, counts in self.counts.items()}

    def update(self, predictions, truth, agreement, confidence, radius_candidates, final):
        matches = predictions == truth
        masks = {
            'before': np.ones_like(final),
            'agreement': agreement,
            'confidence': confidence,
            'confidence_knn': confidence & final,
            'low_confidence': agreement & ~confidence,
            'radius_candidates': radius_candidates,
            'radius_final': radius_candidates & final,
            'final': final,
            'rejected': ~final,
        }
        for stage, mask in masks.items():
            self.counts[stage] += np.bincount(predictions[mask], minlength=len(CLASS_NAMES))
            self.correct[stage] += np.bincount(predictions[mask & matches], minlength=len(CLASS_NAMES))

    def markdown(self, epoch_index):
        def rate(numerator, denominator):
            return f'{numerator / denominator:.2%}' if denominator else '—'

        lines = [
            f'### 第 {epoch_index + 1} 轮训练现场伪标签统计（最佳目标域验证轮次）',
            '',
            '| 筛选阶段 | 样本数 | 正确数 | 准确率 | 覆盖率 |',
            '| --- | ---: | ---: | ---: | ---: |',
        ]
        total = self.counts['before'].sum()
        for stage, label in STAGES.items():
            count = self.counts[stage].sum()
            correct = self.correct[stage].sum()
            lines.append(f'| {label} | {count} | {correct} | {rate(correct, count)} | {rate(count, total)} |')

        lines.extend([
            '',
            '| 预测类别 | 最终接纳数 | 正确数 | 最终准确率 | 半径救援数 | 救援正确数 | 救援正确率 |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
        ])
        for class_id, label in enumerate(CLASS_NAMES):
            count = self.counts['final'][class_id]
            correct = self.correct['final'][class_id]
            rescued = self.counts['radius_final'][class_id]
            rescue_correct = self.correct['radius_final'][class_id]
            lines.append(
                f'| {label} | {count} | {correct} | {rate(correct, count)} | '
                f'{rescued} | {rescue_correct} | {rate(rescue_correct, rescued)} |'
            )

        lines.extend([
            '',
            '| 半径组 | 最终救援数 | 正确数 | 错误数 | 正确率 |',
            '| --- | ---: | ---: | ---: | ---: |',
        ])
        for label, ids in (
            ('积极', POSITIVE_CLASSES),
            ('消极', NEGATIVE_CLASSES),
            ('合计', range(len(CLASS_NAMES))),
        ):
            rescued = self.counts['radius_final'][list(ids)].sum()
            correct = self.correct['radius_final'][list(ids)].sum()
            lines.append(f'| {label} | {rescued} | {correct} | {rescued - correct} | {rate(correct, rescued)} |')
        return '\n'.join(lines)
