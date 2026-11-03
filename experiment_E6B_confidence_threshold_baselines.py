
# -*- coding: utf-8 -*-
"""
E6-B: Confidence Threshold vs Fixed Top-K
Jetson Xavier / YOLOv8n / TensorRT

Required:
  experiment_e1_topk.py
  experiment_e5_sensitivity.py
  eval_precision_1012TEST.py
  1012_fp16.engine
  1012_int8.engine

Output:
  E6B_confidence_threshold_results.json
  E6B_confidence_threshold_results.csv

No torch, ultralytics or pycuda.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

import experiment_e1_topk as e1
import experiment_e5_sensitivity as e5


ROOT = Path(__file__).resolve().parent

CASES = [
    ("Unlimited", "K", None),
    ("TopK40", "K", 40),
    ("TopK50", "K", 50),
    ("Conf0.005", "CONF", 0.005),
    ("Conf0.01", "CONF", 0.01),
    ("Conf0.02", "CONF", 0.02),
    ("Conf0.05", "CONF", 0.05),
    ("Conf0.10", "CONF", 0.10),
    ("Conf0.20", "CONF", 0.20),
]

MAX_DET = 300


def process_one(item, kind, value):
    """
    Copy every raw TensorRT output for fair comparison.

    For confidence-based selection:
    replace low-score class channels with zero.
    Existing E5 postprocess then filters them
    before executing OpenCV Gaussian Soft-NMS.

    Entire processing time includes raw copy,
    confidence filtering, Top-K and Soft-NMS.
    """

    t0 = time.perf_counter()

    raw = np.array(item["raw"], copy=True)

    if kind == "CONF":
        a = raw[0]

        if a.shape[0] == 4 + e1.NC:
            class_scores = a[4:4 + e1.NC, :]
            discard = (
                np.max(class_scores, axis=0) < value
            )
            class_scores[:, discard] = 0.0

        elif a.shape[1] == 4 + e1.NC:
            class_scores = a[:, 4:4 + e1.NC]
            discard = (
                np.max(class_scores, axis=1) < value
            )
            class_scores[discard, :] = 0.0

        else:
            raise RuntimeError(
                "Unexpected TensorRT output: {}".format(
                    a.shape
                )
            )

        topk = None

    elif kind == "K":
        topk = value

    else:
        raise ValueError(kind)

    filter_ms = (
        time.perf_counter() - t0
    ) * 1000.0

    modified_item = dict(item)
    modified_item["raw"] = raw

    pred, detail = e5.postprocess_e5(
        modified_item,
        topk,
        MAX_DET
    )

    total_ms = (
        time.perf_counter() - t0
    ) * 1000.0

    detail = dict(detail)
    detail["filter_ms"] = filter_ms
    detail["total_post_ms"] = total_ms

    return pred, detail


def mean(values):
    return float(np.mean(values))


def p95(values):
    return float(np.percentile(values, 95))


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-images", type=int, default=0
    )
    parser.add_argument(
        "--repeats", type=int, default=5
    )

    args = parser.parse_args()

    if args.repeats < 1 or args.max_images < 0:
        parser.error("Invalid arguments")

    folder, paths = e1.get_image_paths()

    if args.max_images:
        paths = paths[:args.max_images]

    if not paths:
        raise RuntimeError("No validation images")

    cuda = e1.base.Cuda()

    results = {}
    rows = []

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

        # Verify the new wrapper does not change
        # E5 Top-K=50 predictions.
        for item in outputs[:2]:

            expected, _ = e5.postprocess_e5(
                item, 50, MAX_DET
            )

            actual, _ = process_one(
                item, "K", 50
            )

            if (
                expected.shape != actual.shape
                or not np.allclose(
                    expected, actual,
                    rtol=1e-5,
                    atol=1e-4
                )
            ):
                raise RuntimeError(
                    "E6-B TopK50 differs from E5"
                )

        print("PASS: E6-B TopK50 matches E5")

        results[model_name] = {}

        for case_name, kind, value in CASES:

            predictions = None
            latencies = []
            filter_times = []
            nms_times = []
            candidate_counts = []

            for rep in range(args.repeats):

                current = []

                for item in outputs:

                    pred, d = process_one(
                        item, kind, value
                    )

                    current.append(pred)

                    latencies.append(
                        d["total_post_ms"]
                    )
                    filter_times.append(
                        d["filter_ms"]
                    )
                    nms_times.append(
                        d["nms_ms"]
                    )

                    if rep == 0:
                        candidate_counts.append(
                            d["candidates_after"]
                        )

                if predictions is None:
                    predictions = current

            metrics = e1.evaluate_map(
                predictions, gts
            )

            entry = {
                "model": model_name,
                "method": case_name,
                "selection": kind,
                "parameter": (
                    "unlimited"
                    if value is None else value
                ),
                "num_images": len(paths),
                "mAP50": metrics["mAP50"],
                "mAP50_95": metrics["mAP50_95"],
                "avg_candidates": mean(
                    candidate_counts
                ),
                "filter_ms": mean(filter_times),
                "nms_ms": mean(nms_times),
                "post_ms": mean(latencies),
                "post_p95_ms": p95(latencies),
                "per_class": metrics["per_class"],
            }

            results[model_name][case_name] = entry

            rows.append({
                k: v for k, v in entry.items()
                if k != "per_class"
            })

            print(
                "{:<7} {:<12} "
                "mAP50:95={:.4f} "
                "Candidates={:.1f} "
                "Post={:.3f}ms".format(
                    model_name,
                    case_name,
                    entry["mAP50_95"],
                    entry["avg_candidates"],
                    entry["post_ms"]
                ),
                flush=True
            )

    json_path = (
        ROOT / "E6B_confidence_threshold_results.json"
    )

    csv_path = (
        ROOT / "E6B_confidence_threshold_results.csv"
    )

    json_path.write_text(
        json.dumps(
            results, indent=2, ensure_ascii=False
        ),
        encoding="utf-8"
    )

    with open(
        csv_path, "w",
        newline="", encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nE6-B COMPLETED")
    print("Saved:", json_path)
    print("Saved:", csv_path)


if __name__ == "__main__":
    main()
