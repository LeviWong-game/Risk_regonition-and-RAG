from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from risk_scoring import assess_case


LABELS: Tuple[str, ...] = ("L1", "L2", "L3", "L4")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def round_metric(value: float) -> float:
    return round(float(value), 4)


def risk_level_order(level: str) -> int:
    try:
        return LABELS.index(level)
    except ValueError:
        return -1


def evaluate_rows(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    predictions: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    confusion = {actual: {predicted: 0 for predicted in LABELS} for actual in LABELS}
    score_range_count = 0
    score_range_hits = 0
    score_abs_errors: List[float] = []

    for row in rows:
        text = row.get("case_text") or row.get("input") or row.get("description") or ""
        assessment = assess_case(text)

        actual_level = str(row.get("risk_level", ""))
        predicted_level = assessment.risk_level
        predicted_score = assessment.risk_score
        correct = actual_level == predicted_level

        min_score = row.get("risk_score_min")
        max_score = row.get("risk_score_max")
        score_in_range = None
        score_abs_error_to_midpoint = None
        if isinstance(min_score, (int, float)) and isinstance(max_score, (int, float)):
            score_range_count += 1
            score_in_range = min_score <= predicted_score <= max_score
            score_range_hits += int(score_in_range)
            midpoint = (float(min_score) + float(max_score)) / 2.0
            score_abs_error_to_midpoint = abs(float(predicted_score) - midpoint)
            score_abs_errors.append(score_abs_error_to_midpoint)

        if actual_level in confusion and predicted_level in confusion[actual_level]:
            confusion[actual_level][predicted_level] += 1

        direction = "correct"
        actual_order = risk_level_order(actual_level)
        predicted_order = risk_level_order(predicted_level)
        if actual_order >= 0 and predicted_order >= 0:
            if predicted_order > actual_order:
                direction = "overestimated"
            elif predicted_order < actual_order:
                direction = "underestimated"

        prediction = {
            "sample_id": row.get("sample_id"),
            "case_type": row.get("case_type"),
            "title": row.get("title"),
            "actual_level": actual_level,
            "predicted_level": predicted_level,
            "correct": correct,
            "actual_score_range": [min_score, max_score],
            "predicted_score": predicted_score,
            "score_in_range": score_in_range,
            "score_abs_error_to_range_midpoint": (
                round_metric(score_abs_error_to_midpoint) if score_abs_error_to_midpoint is not None else None
            ),
            "direction": direction,
            "trigger_flags": assessment.trigger_flags,
            "dimension_scores": {
                "P": assessment.P,
                "I": assessment.I,
                "D": assessment.D,
            },
            "recommendation": assessment.recommendation,
            "reason_text": assessment.reason_text,
            "label_confidence": row.get("label_confidence"),
            "label_conflict": row.get("label_conflict"),
            "resolution_strategy": row.get("resolution_strategy"),
            "source_count": row.get("source_count"),
            "source_refs": row.get("source_refs"),
            "text_preview": text[:160],
        }
        predictions.append(prediction)

        if not correct:
            errors.append(
                {
                    "sample_id": row.get("sample_id"),
                    "case_type": row.get("case_type"),
                    "title": row.get("title"),
                    "actual_level": actual_level,
                    "predicted_level": predicted_level,
                    "predicted_score": predicted_score,
                    "actual_score_range": [min_score, max_score],
                    "score_in_range": score_in_range,
                    "direction": direction,
                    "trigger_flags": assessment.trigger_flags,
                    "label_confidence": row.get("label_confidence"),
                    "label_conflict": row.get("label_conflict"),
                    "resolution_strategy": row.get("resolution_strategy"),
                    "source_refs": row.get("source_refs"),
                    "reason_text": assessment.reason_text,
                    "text_preview": text[:160],
                }
            )

    metrics = build_metrics(rows, predictions, confusion, score_range_count, score_range_hits, score_abs_errors)
    return predictions, metrics, errors


def build_metrics(
    rows: Sequence[Dict[str, Any]],
    predictions: Sequence[Dict[str, Any]],
    confusion: Dict[str, Dict[str, int]],
    score_range_count: int,
    score_range_hits: int,
    score_abs_errors: Sequence[float],
) -> Dict[str, Any]:
    total = len(predictions)
    correct = sum(1 for item in predictions if item["correct"])
    actual_distribution = Counter(str(row.get("risk_level", "")) for row in rows)
    predicted_distribution = Counter(item["predicted_level"] for item in predictions)

    per_label: Dict[str, Dict[str, Any]] = {}
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[actual][label] for actual in LABELS if actual != label)
        fn = sum(confusion[label][predicted] for predicted in LABELS if predicted != label)
        support = sum(confusion[label][predicted] for predicted in LABELS)
        predicted_count = sum(confusion[actual][label] for actual in LABELS)
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        per_label[label] = {
            "support": support,
            "predicted_count": predicted_count,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": round_metric(precision),
            "recall": round_metric(recall),
            "f1": round_metric(f1),
        }

    macro_precision = sum(per_label[label]["precision"] for label in LABELS) / len(LABELS)
    macro_recall = sum(per_label[label]["recall"] for label in LABELS) / len(LABELS)
    macro_f1 = sum(per_label[label]["f1"] for label in LABELS) / len(LABELS)

    weighted_precision = safe_divide(
        sum(per_label[label]["precision"] * per_label[label]["support"] for label in LABELS),
        total,
    )
    weighted_recall = safe_divide(
        sum(per_label[label]["recall"] * per_label[label]["support"] for label in LABELS),
        total,
    )
    weighted_f1 = safe_divide(
        sum(per_label[label]["f1"] * per_label[label]["support"] for label in LABELS),
        total,
    )

    present_labels = [label for label in LABELS if per_label[label]["support"] > 0]
    macro_present = {
        "precision": round_metric(
            safe_divide(sum(per_label[label]["precision"] for label in present_labels), len(present_labels))
        ),
        "recall": round_metric(safe_divide(sum(per_label[label]["recall"] for label in present_labels), len(present_labels))),
        "f1": round_metric(safe_divide(sum(per_label[label]["f1"] for label in present_labels), len(present_labels))),
    }

    overestimated = sum(1 for item in predictions if item["direction"] == "overestimated")
    underestimated = sum(1 for item in predictions if item["direction"] == "underestimated")
    error_pairs = Counter(
        f"{item['actual_level']}->{item['predicted_level']}" for item in predictions if not item["correct"]
    )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "test_dataset": "risk_val_clean.jsonl",
        "algorithm_entry": "risk_scoring.assess_case",
        "labels": list(LABELS),
        "total_samples": total,
        "correct_samples": correct,
        "error_samples": total - correct,
        "accuracy": round_metric(safe_divide(correct, total)),
        "actual_label_distribution": dict(actual_distribution),
        "predicted_label_distribution": dict(predicted_distribution),
        "per_label": per_label,
        "macro_avg_all_labels": {
            "precision": round_metric(macro_precision),
            "recall": round_metric(macro_recall),
            "f1": round_metric(macro_f1),
        },
        "macro_avg_present_labels": macro_present,
        "weighted_avg": {
            "precision": round_metric(weighted_precision),
            "recall": round_metric(weighted_recall),
            "f1": round_metric(weighted_f1),
        },
        "score_range": {
            "evaluated_samples": score_range_count,
            "hit_samples": score_range_hits,
            "hit_rate": round_metric(safe_divide(score_range_hits, score_range_count)),
            "mean_abs_error_to_range_midpoint": round_metric(
                safe_divide(sum(score_abs_errors), len(score_abs_errors))
            ),
        },
        "error_analysis": {
            "overestimated": overestimated,
            "underestimated": underestimated,
            "confusion_pairs": dict(error_pairs),
        },
        "confusion_matrix": confusion,
    }


def write_confusion_matrix(path: Path, confusion: Dict[str, Dict[str, int]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["actual\\predicted", *LABELS, "total"])
        for actual in LABELS:
            values = [confusion[actual][predicted] for predicted in LABELS]
            writer.writerow([actual, *values, sum(values)])
        totals = [sum(confusion[actual][predicted] for actual in LABELS) for predicted in LABELS]
        writer.writerow(["total", *totals, sum(totals)])


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([header_line, separator, *row_lines])


def build_report(metrics: Dict[str, Any], errors: Sequence[Dict[str, Any]]) -> str:
    confusion_rows = [
        [actual, *[metrics["confusion_matrix"][actual][predicted] for predicted in LABELS]]
        for actual in LABELS
    ]
    label_rows = [
        [
            label,
            metrics["per_label"][label]["support"],
            metrics["per_label"][label]["precision"],
            metrics["per_label"][label]["recall"],
            metrics["per_label"][label]["f1"],
        ]
        for label in LABELS
    ]
    error_rows = [
        [
            item.get("sample_id"),
            item.get("actual_level"),
            item.get("predicted_level"),
            item.get("predicted_score"),
            item.get("direction"),
            item.get("label_confidence"),
            item.get("label_conflict"),
        ]
        for item in errors[:20]
    ]

    score_range = metrics["score_range"]
    weighted = metrics["weighted_avg"]
    macro_all = metrics["macro_avg_all_labels"]
    macro_present = metrics["macro_avg_present_labels"]
    error_analysis = metrics["error_analysis"]

    report = f"""# 风险预测算法测试结果报告

生成时间：{metrics["generated_at"]}

## 1. 测试对象与数据

- 测试对象：`risk_data` 目录中的现有规则风险预测算法，入口为 `{metrics["algorithm_entry"]}`。
- 测试数据：`{metrics["test_dataset"]}`，共 `{metrics["total_samples"]}` 条验证样本。
- 测试方式：离线批量评估。逐条读取验证集样本，将 `case_text` 作为算法输入；如果 `case_text` 为空，则回退使用 `input` 字段。
- 对比标准：以样本标注的 `risk_level` 作为风险等级分类标准，以 `risk_score_min` / `risk_score_max` 作为风险分数区间评估标准。

## 2. 测试方法

1. 等级分类评估：比较算法预测的 `predicted_level` 与人工/主数据标注的 `actual_level` 是否一致。
2. 分数区间评估：检查算法输出的 `predicted_score` 是否落入样本标注的风险分数区间。
3. 混淆矩阵分析：统计每个真实等级被预测成各等级的数量，用于观察高估、低估和易混淆类别。
4. 错误样本分析：单独保存预测错误的样本，保留样本 ID、真实等级、预测等级、预测分数、标签置信度和标签冲突信息，便于复查。

## 3. 评估指标说明

- Accuracy：预测等级完全等于标注等级的样本占比。
- Precision：被预测为某等级的样本中，真实也属于该等级的比例。
- Recall：真实属于某等级的样本中，被算法正确找回的比例。
- F1：Precision 和 Recall 的调和平均，更适合观察单个等级的综合表现。
- Confusion Matrix：真实等级与预测等级的交叉统计表。
- Score Range Hit Rate：预测分数落入标注分数区间的比例。

## 4. 总体结果

- 样本总数：{metrics["total_samples"]}
- 预测正确：{metrics["correct_samples"]}
- 预测错误：{metrics["error_samples"]}
- Accuracy：{metrics["accuracy"]}
- Macro Precision / Recall / F1（L1-L4 全标签）：{macro_all["precision"]} / {macro_all["recall"]} / {macro_all["f1"]}
- Macro Precision / Recall / F1（仅验证集中出现的标签）：{macro_present["precision"]} / {macro_present["recall"]} / {macro_present["f1"]}
- Weighted Precision / Recall / F1：{weighted["precision"]} / {weighted["recall"]} / {weighted["f1"]}
- 分数区间命中：{score_range["hit_samples"]}/{score_range["evaluated_samples"]}
- Score Range Hit Rate：{score_range["hit_rate"]}
- 预测分数到标注区间中点的平均绝对误差：{score_range["mean_abs_error_to_range_midpoint"]}

## 5. 标签分布

真实标签分布：`{json.dumps(metrics["actual_label_distribution"], ensure_ascii=False)}`

预测标签分布：`{json.dumps(metrics["predicted_label_distribution"], ensure_ascii=False)}`

## 6. 混淆矩阵

{markdown_table(["actual\\predicted", *LABELS], confusion_rows)}

## 7. 分等级指标

{markdown_table(["label", "support", "precision", "recall", "f1"], label_rows)}

## 8. 错误样本摘要

- 高估样本数：{error_analysis["overestimated"]}
- 低估样本数：{error_analysis["underestimated"]}
- 主要误判对：`{json.dumps(error_analysis["confusion_pairs"], ensure_ascii=False)}`

{markdown_table(["sample_id", "actual", "predicted", "score", "direction", "label_confidence", "label_conflict"], error_rows) if error_rows else "无错误样本。"}

## 9. 结果解读与建议

- 当前规则引擎在验证集上的等级准确率为 {metrics["accuracy"]}，可以作为现有算法的基线测试结果。
- 分数区间命中率为 {score_range["hit_rate"]}，说明等级预测和连续分数校准仍有同步优化空间。
- 验证集中没有 L4 真实样本，因此 L4 的召回能力不能通过本次数据充分验证；后续应补充高风险样本做专项测试。
- 错误样本中需要重点复查高估和低估案例，结合 `trigger_flags` 与 `reason_text` 判断是否由关键词触发过强、案件类型先验偏高或风险触发规则不足导致。
- 如果后续要优化算法，建议优先从误判对最多的等级组合入手，调整关键词权重、触发规则下限和案件类型先验。
"""
    return report


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    val_path = base_dir / "risk_val_clean.jsonl"
    output_dir = base_dir / "test_result"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(val_path)
    predictions, metrics, errors = evaluate_rows(rows)

    write_jsonl(output_dir / "predictions.jsonl", predictions)
    write_jsonl(output_dir / "error_cases.jsonl", errors)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
        f.write("\n")
    write_confusion_matrix(output_dir / "confusion_matrix.csv", metrics["confusion_matrix"])
    (output_dir / "algorithm_test_report.md").write_text(build_report(metrics, errors), encoding="utf-8")

    print(f"Test finished. samples={metrics['total_samples']}, accuracy={metrics['accuracy']}")
    print(f"Results written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
