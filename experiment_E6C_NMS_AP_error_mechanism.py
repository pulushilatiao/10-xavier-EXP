
# -*- coding: utf-8 -*-
"""
E6-C: NMS Candidate Budget and AP Error Mechanism

Analyses:
  Per-class AP50 and AP50:95
  TP / FP / FN
  Precision / Recall
  Score-threshold sensitivity
  Image-level errors

K = 10, 30, 40, 50, 200, unlimited

Output:
  E6C_NMS_AP_mechanism_summary.csv
  E6C_NMS_precision_recall_points.csv
  E6C_NMS_image_errors.csv
  E6C_NMS_AP_mechanism.json

No torch, ultralytics or pycuda.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import experiment_e1_topk as e1
import experiment_e5_sensitivity as e5


ROOT = Path(__file__).resolve().parent

METHODS = [
    ("K10", 10),
    ("K30", 30),
    ("K40", 40),
    ("K50", 50),
    ("K200", 200),
    ("Unlimited", None),
]

SCORE_THRESHOLDS = [
    0.001, 0.01, 0.05, 0.25, 0.50
]

IOU_THRESHOLDS = [0.50, 0.75]

CLASS_NAMES = ["tomato", "leaf", "stem"]


def match_one_image(
    predictions,
    ground_truth,
    class_id,
    score_threshold,
    iou_threshold
):
    """
    Greedy one-to-one detection-to-GT matching.

    Returns TP, FP and FN for one class/image.
    """

    gt = ground_truth[
        ground_truth[:, 4] == class_id
    ]

    pred = predictions[
        (predictions[:, 5] == class_id)
        & (predictions[:, 4] >= score_threshold)
    ]

    pred = pred[
        np.argsort(-pred[:, 4])
    ]

    matched = np.zeros(
        len(gt), dtype=bool
    )

    tp = 0
    fp = 0

    for p in pred:

        if len(gt) == 0:
            fp += 1
            continue

        ious = e1.box_iou_one_to_many(
            p[:4],
            gt[:, :4]
        )

        j = int(np.argmax(ious))

        if (
            ious[j] >= iou_threshold
            and not matched[j]
        ):
            tp += 1
            matched[j] = True
        else:
            fp += 1

    fn = int(len(gt) - matched.sum())

    return tp, fp, fn


def calculate_pr(
    predictions,
    ground_truths,
    class_id,
    score_threshold,
    iou_threshold
):

    TP = 0
    FP = 0
    FN = 0

    for pred, gt in zip(
        predictions, ground_truths
    ):

        tp, fp, fn = match_one_image(
            pred,
            gt,
            class_id,
            score_threshold,
            iou_threshold
        )

        TP += tp
        FP += fp
        FN += fn

    precision = (
        TP / (TP + FP)
        if TP + FP else 0.0
    )

    recall = (
        TP / (TP + FN)
        if TP + FN else 0.0
    )

    return {
        "TP": int(TP),
        "FP": int(FP),
        "FN": int(FN),
        "precision": float(precision),
        "recall": float(recall),
    }


def write_csv(path, rows):

    with open(
        path, "w", newline="", encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-images", type=int, default=0
    )

    args = parser.parse_args()

    if args.max_images < 0:
        parser.error("Invalid --max-images")

    folder, paths = e1.get_image_paths()

    if args.max_images:
        paths = paths[:args.max_images]

    cuda = e1.base.Cuda()

    summary_rows = []
    pr_rows = []
    image_rows = []

    output_json = {}

    for model_name, engine_path in e1.MODELS.items():

        print(
            "\nCollecting:", model_name,
            flush=True
        )

        outputs, gts = e1.collect_outputs(
            model_name,
            engine_path,
            folder,
            paths,
            cuda
        )

        output_json[model_name] = {}

        for method_name, k in METHODS:

            predictions = []
            details = []

            for item in outputs:

                pred, detail = e5.postprocess_e5(
                    item,
                    k,
                    300
                )

                predictions.append(pred)
                details.append(detail)

            metrics = e1.evaluate_map(
                predictions,
                gts
            )

            output_json[model_name][method_name] = {
                "metrics": metrics,
                "candidate_counts": [
                    {
                        "image": item["name"],
                        "before": d["candidates_before"],
                        "after_topk": d["candidates_after"],
                        "after_nms": d["after_nms"],
                        "kept": d["kept"]
                    }
                    for item, d in zip(outputs, details)
                ],
            }

            print(
                "{} / {}  mAP50={:.4f} "
                "mAP50:95={:.4f}".format(
                    model_name,
                    method_name,
                    metrics["mAP50"],
                    metrics["mAP50_95"]
                ),
                flush=True
            )

            for class_id, class_name in enumerate(
                CLASS_NAMES
            ):

                cls_ap = metrics[
                    "per_class"
                ][class_name]

                if cls_ap is None:
                    continue

                # AP at IoU 0.75 from the same
                # evaluator used in E1-E6.
                ap75 = e1.evaluate_class_iou(
                    predictions,
                    gts,
                    class_id,
                    0.75
                )

                result_025 = calculate_pr(
                    predictions,
                    gts,
                    class_id,
                    0.25,
                    0.50
                )

                result_001 = calculate_pr(
                    predictions,
                    gts,
                    class_id,
                    0.001,
                    0.50
                )

                summary_rows.append({
                    "model": model_name,
                    "method": method_name,
                    "class": class_name,
                    "topk": (
                        "unlimited"
                        if k is None else k
                    ),
                    "AP50": cls_ap["AP50"],
                    "AP75": ap75,
                    "AP50_95": cls_ap["AP50_95"],
                    "TP_conf025": result_025["TP"],
                    "FP_conf025": result_025["FP"],
                    "FN_conf025": result_025["FN"],
                    "precision_conf025": (
                        result_025["precision"]
                    ),
                    "recall_conf025": (
                        result_025["recall"]
                    ),
                    "recall_conf0001": (
                        result_001["recall"]
                    ),
                })

                # PR operating points
                for iou in IOU_THRESHOLDS:

                    for score in SCORE_THRESHOLDS:

                        pr = calculate_pr(
                            predictions,
                            gts,
                            class_id,
                            score,
                            iou
                        )

                        pr_rows.append({
                            "model": model_name,
                            "method": method_name,
                            "class": class_name,
                            "iou": iou,
                            "confidence": score,
                            **pr,
                        })

                # Per-image errors at score 0.25.
                for item, pred, gt in zip(
                    outputs,
                    predictions,
                    gts
                ):

                    tp, fp, fn = match_one_image(
                        pred,
                        gt,
                        class_id,
                        0.25,
                        0.50
                    )

                    image_rows.append({
                        "model": model_name,
                        "method": method_name,
                        "image": item["name"],
                        "class": class_name,
                        "TP": tp,
                        "FP": fp,
                        "FN": fn,
                    })

    summary_path = (
        ROOT / "E6C_NMS_AP_mechanism_summary.csv"
    )

    pr_path = (
        ROOT / "E6C_NMS_precision_recall_points.csv"
    )

    image_path = (
        ROOT / "E6C_NMS_image_errors.csv"
    )

    json_path = (
        ROOT / "E6C_NMS_AP_mechanism.json"
    )

    write_csv(summary_path, summary_rows)
    write_csv(pr_path, pr_rows)
    write_csv(image_path, image_rows)

    json_path.write_text(
        json.dumps(
            output_json,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\nE6-C COMPLETED")

    for p in (
        summary_path,
        pr_path,
        image_path,
        json_path
    ):
        print("Saved:", p)


if __name__ == "__main__":
    main()
