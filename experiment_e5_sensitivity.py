
# -*- coding: utf-8 -*-
"""
E5: Soft-NMS Top-K and MAX_DET Sensitivity
Jetson Xavier / TensorRT 8.5.x

E5-A: K = unlimited, 50, 100, 150, 200, 300, 500
      MAX_DET = 300

E5-B: K = 200 or unlimited
      MAX_DET = 300 or 1000

Uses:
  experiment_e1_topk.py
  eval_precision_1012TEST.py
  1012_fp16.engine
  1012_int8.engine

No torch / PyCUDA / new model building.

Outputs:
  E5_sensitivity_results.json
  E5_sensitivity_results.csv
"""

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

import experiment_e1_topk as e1


ROOT = Path(__file__).resolve().parent

SIGMA = 0.5
CONF = e1.CONF
NMS_IOU = e1.IOU_NMS

K_VALUES = [10, 20, 30, 40, 50, 75, 100, 200, None]
MAX_DET_VALUES = [300, 1000]


def clock():
    return time.perf_counter()


def postprocess_e5(item, pre_topk, max_det):
    """
    Gaussian OpenCV Soft-NMS.

    pre_topk:
        per-class candidate limit BEFORE Soft-NMS.

    max_det:
        global output limit AFTER all classes.

    OpenCV internal top_k = 0:
        do not impose another hidden per-class cap.
    """

    start = clock()

    raw = item["raw"][0].astype(np.float32)

    if raw.shape == (4 + e1.NC, 8400):
        raw = raw.T
    elif raw.shape != (8400, 4 + e1.NC):
        raise RuntimeError(
            "Unexpected raw output: {}".format(
                raw.shape
            )
        )

    scores = raw[:, 4:].max(axis=1)
    classes = raw[:, 4:].argmax(axis=1)

    valid = scores > CONF

    raw = raw[valid]
    scores = scores[valid]
    classes = classes[valid]

    boxes = np.empty(
        (len(raw), 4),
        dtype=np.float32
    )

    if len(raw):
        boxes[:, :2] = (
            raw[:, :2] - raw[:, 2:4] / 2
        )
        boxes[:, 2:] = (
            raw[:, :2] + raw[:, 2:4] / 2
        )

    decode_ms = (clock() - start) * 1000

    # ======================================
    # Pre-NMS Top-K
    # ======================================

    t = clock()

    groups = []
    candidates_before = len(scores)

    for c in range(e1.NC):

        idx = np.flatnonzero(classes == c)

        if not len(idx):
            continue

        if pre_topk is not None and len(idx) > pre_topk:

            local = np.argpartition(
                -scores[idx],
                pre_topk - 1
            )[:pre_topk]

            idx = idx[local]

        groups.append(idx)

    topk_ms = (clock() - t) * 1000

    candidates_after = sum(
        len(idx) for idx in groups
    )

    # ======================================
    # Compiled Gaussian Soft-NMS
    # ======================================

    t = clock()

    picked_ids = []
    picked_scores = []

    for idx in groups:

        xywh = np.column_stack(
            (
                boxes[idx, :2],
                np.maximum(
                    0,
                    boxes[idx, 2:] - boxes[idx, :2]
                )
            )
        )

        # Compatible with OpenCV Rect interface.
        rects = np.rint(xywh).astype(np.int32)
        rects[:, 2:] = np.maximum(
            rects[:, 2:], 0
        )

        # OpenCV internal top_k=0:
        # no additional internal selection cap.
        updated_scores, keep = cv2.dnn.softNMSBoxes(
            rects.tolist(),
            scores[idx].astype(np.float32).tolist(),
            CONF,
            NMS_IOU,
            0,
            SIGMA
        )

        keep = np.asarray(
            keep,
            dtype=np.int32
        ).reshape(-1)

        updated_scores = np.asarray(
            updated_scores,
            dtype=np.float32
        ).reshape(-1)

        if len(keep) != len(updated_scores):
            raise RuntimeError(
                "Soft-NMS score/index mismatch"
            )

        if len(keep):

            picked_ids.append(idx[keep])
            picked_scores.append(updated_scores)

    nms_ms = (clock() - t) * 1000

    # ======================================
    # Global MAX_DET and restore coordinates
    # ======================================

    t = clock()

    after_nms_count = sum(
        len(x) for x in picked_ids
    )

    if picked_ids:

        picked = np.concatenate(picked_ids)
        adjusted = np.concatenate(picked_scores)

        order = np.argsort(
            -adjusted
        )[:max_det]

        picked = picked[order]
        adjusted = adjusted[order]

        b = boxes[picked].copy()

        b[:, [0, 2]] = np.clip(
            (b[:, [0, 2]] - item["left"])
            / item["ratio"],
            0,
            item["width"]
        )

        b[:, [1, 3]] = np.clip(
            (b[:, [1, 3]] - item["top"])
            / item["ratio"],
            0,
            item["height"]
        )

        pred = np.column_stack(
            (
                b,
                adjusted,
                classes[picked]
            )
        ).astype(np.float32)

    else:

        pred = np.zeros(
            (0, 6),
            np.float32
        )

    restore_ms = (clock() - t) * 1000

    return pred, {
        "decode_ms": decode_ms,
        "topk_ms": topk_ms,
        "nms_ms": nms_ms,
        "restore_ms": restore_ms,
        "post_ms": (clock() - start) * 1000,
        "candidates_before": candidates_before,
        "candidates_after": candidates_after,
        "after_nms": after_nms_count,
        "kept": len(pred),
        "capped": after_nms_count > max_det,
    }


def stats(values):
    x = np.asarray(values, dtype=np.float64)

    return {
        "mean_ms": float(x.mean()),
        "median_ms": float(np.median(x)),
        "p95_ms": float(np.percentile(x, 95)),
    }


def run_case(outputs, gts, pre_topk,
             max_det, repeats):

    predictions = None
    timings = {
        k: [] for k in (
            "decode_ms",
            "topk_ms",
            "nms_ms",
            "restore_ms",
            "post_ms",
        )
    }

    counts = []
    capped = 0

    for rep in range(repeats):

        current = []

        for item in outputs:

            pred, info = postprocess_e5(
                item,
                pre_topk,
                max_det
            )

            current.append(pred)

            for key in timings:
                timings[key].append(info[key])

            if rep == 0:
                counts.append(info)
                capped += int(info["capped"])

        if predictions is None:
            predictions = current

    metrics = e1.evaluate_map(
        predictions,
        gts
    )

    return {
        "topk": pre_topk,
        "max_det": max_det,
        "metrics": metrics,
        "timing": {
            k: stats(v)
            for k, v in timings.items()
        },
        "avg_candidates_before": float(
            np.mean([
                x["candidates_before"]
                for x in counts
            ])
        ),
        "avg_candidates_after": float(
            np.mean([
                x["candidates_after"]
                for x in counts
            ])
        ),
        "avg_after_nms": float(
            np.mean([
                x["after_nms"]
                for x in counts
            ])
        ),
        "avg_kept": float(
            np.mean([
                x["kept"]
                for x in counts
            ])
        ),
        "capped_images": capped,
        "num_images": len(outputs),
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-images",
        type=int,
        default=0
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=3
    )

    parser.add_argument(
        "--only",
        choices=["A", "B", "both"],
        default="both"
    )

    args = parser.parse_args()

    if args.repeats < 1 or args.max_images < 0:
        parser.error("Invalid arguments")

    if not hasattr(cv2.dnn, "softNMSBoxes"):
        raise RuntimeError(
            "OpenCV softNMSBoxes unavailable"
        )

    folder, paths = e1.get_image_paths()

    if args.max_images:
        paths = paths[:args.max_images]

    print("OpenCV:", cv2.__version__)
    print("Images:", len(paths))
    print("Repeats:", args.repeats)
    print("Mode:", args.only)
    print("CONF:", CONF)
    print("NMS_IOU:", NMS_IOU)
    print("SIGMA:", SIGMA)

    # Deduplicated E5 cases.
    cases = []

    if args.only in ("A", "both"):
        for k in K_VALUES:
            cases.append(("E5-A", k, 300))

    if args.only in ("B", "both"):
        for k in (200, None):
            for md in MAX_DET_VALUES:
                case = ("E5-B", k, md)
                if case not in cases:
                    cases.append(case)

    cuda = e1.base.Cuda()

    final = {}
    rows = []

    for model_name, engine_path in e1.MODELS.items():

        print("\n" + "=" * 60)
        print("Collecting TensorRT outputs:", model_name)
        print("=" * 60, flush=True)

        outputs, gts = e1.collect_outputs(
            model_name,
            engine_path,
            folder,
            paths,
            cuda
        )

        final[model_name] = []

        for group, k, md in cases:

            label = "{} K={} MAX_DET={}".format(
                group,
                "unlimited" if k is None else k,
                md
            )

            print("\nRunning", label, flush=True)

            result = run_case(
                outputs,
                gts,
                k,
                md,
                args.repeats
            )

            result["group"] = group
            final[model_name].append(result)

            m = result["metrics"]
            t = result["timing"]

            row = {
                "model": model_name,
                "group": group,
                "topk": (
                    "unlimited"
                    if k is None else str(k)
                ),
                "max_det": md,
                "mAP50": m["mAP50"],
                "mAP50_95": m["mAP50_95"],
                "topk_ms": t["topk_ms"]["mean_ms"],
                "nms_ms": t["nms_ms"]["mean_ms"],
                "post_ms": t["post_ms"]["mean_ms"],
                "post_p95_ms": t["post_ms"]["p95_ms"],
                "candidates_before": result[
                    "avg_candidates_before"
                ],
                "candidates_after": result[
                    "avg_candidates_after"
                ],
                "after_nms": result["avg_after_nms"],
                "kept": result["avg_kept"],
                "capped_images": result[
                    "capped_images"
                ],
            }

            rows.append(row)

            print(
                "mAP50={:.4f} mAP50:95={:.4f} "
                "NMS={:.3f}ms Post={:.3f}ms "
                "Capped={}/{}".format(
                    row["mAP50"],
                    row["mAP50_95"],
                    row["nms_ms"],
                    row["post_ms"],
                    row["capped_images"],
                    len(outputs)
                ),
                flush=True
            )

    output_json = ROOT / "E5_sensitivity_results.json"
    output_csv = ROOT / "E5_sensitivity_results.csv"

    output_json.write_text(
        json.dumps(
            final,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    with open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 72)
    print("E5 FINAL RESULTS")
    print("=" * 72)

    for r in rows:
        print(
            "{:<5} {:<5} K={:<9} MAX_DET={:<4} "
            "mAP50={:.4f} mAP50:95={:.4f} "
            "NMS={:.3f}ms Post={:.3f}ms".format(
                r["model"],
                r["group"],
                r["topk"],
                r["max_det"],
                r["mAP50"],
                r["mAP50_95"],
                r["nms_ms"],
                r["post_ms"]
            )
        )

    print("\nSaved:", output_json)
    print("Saved:", output_csv)


if __name__ == "__main__":
    main()
