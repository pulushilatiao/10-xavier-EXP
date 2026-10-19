
# -*- coding: utf-8 -*-
"""
E2: Soft-NMS experiments on Jetson Xavier

E2-A: Linear Soft-NMS
E2-B: Gaussian Soft-NMS
E2-C: Top-K 200 + Gaussian Soft-NMS

Also evaluates E0 and E1-C references.

Requires:
    experiment_e1_topk.py
    eval_precision_1012TEST.py

No torch, PyCUDA, or Ultralytics.

Outputs:
    E2_softnms_results.json
    E2_softnms_results.csv
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

METHODS = {
    "E0": ("greedy", None),
    "E1-C": ("greedy", 200),
    "E2-A": ("linear", None),
    "E2-B": ("gaussian", None),
    "E2-C": ("gaussian", 200),
}

SIGMA = 0.5
SCORE_THRESHOLD = e1.CONF
NMS_THRESHOLD = e1.IOU_NMS
MAX_DET = e1.MAX_DET


def now():
    return time.perf_counter()


def iou_one_to_many(box, boxes):

    if len(boxes) == 0:
        return np.empty(0, dtype=np.float32)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = (
        np.maximum(x2 - x1, 0)
        * np.maximum(y2 - y1, 0)
    )

    area1 = (
        max(box[2] - box[0], 0)
        * max(box[3] - box[1], 0)
    )

    area2 = (
        np.maximum(boxes[:, 2] - boxes[:, 0], 0)
        * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    )

    union = area1 + area2 - intersection

    return intersection / np.maximum(union, 1e-9)


def soft_nms_single_class(
    boxes,
    scores,
    method,
    max_keep=MAX_DET
):
    """
    Iterative Soft-NMS using NumPy.

    Returns original local indices and updated scores.

    Scores of remaining candidates are updated after
    selecting the current highest-score candidate.
    """

    boxes = np.asarray(boxes, dtype=np.float32)

    working_scores = np.asarray(
        scores, dtype=np.float32
    ).copy()

    active = np.arange(
        len(working_scores),
        dtype=np.int32
    )

    selected_ids = []
    selected_scores = []

    while len(active) and len(selected_ids) < max_keep:

        pos = int(np.argmax(working_scores[active]))

        best_id = int(active[pos])
        best_score = float(working_scores[best_id])

        if best_score <= SCORE_THRESHOLD:
            break

        selected_ids.append(best_id)
        selected_scores.append(best_score)

        # Remove selected box from active candidates.
        active = np.delete(active, pos)

        if not len(active):
            break

        ious = iou_one_to_many(
            boxes[best_id],
            boxes[active]
        )

        if method == "linear":

            weights = np.ones_like(ious)

            mask = ious >= NMS_THRESHOLD

            weights[mask] = 1.0 - ious[mask]

        elif method == "gaussian":

            weights = np.exp(
                -(ious * ious) / SIGMA
            )

        else:
            raise ValueError(
                "Unknown Soft-NMS method: {}".format(
                    method
                )
            )

        working_scores[active] *= weights

        # Discard candidates below the common score
        # threshold after iterative suppression.
        keep_mask = (
            working_scores[active] > SCORE_THRESHOLD
        )

        active = active[keep_mask]

    return (
        np.asarray(selected_ids, dtype=np.int32),
        np.asarray(selected_scores, dtype=np.float32)
    )


def postprocess_soft(
    raw,
    width,
    height,
    ratio,
    left,
    top,
    method,
    topk
):
    """
    Same detection format and coordinate restoration
    as E1, but uses iterative Soft-NMS.
    """

    total_start = now()

    t = now()

    arr = raw[0].astype(np.float32)

    if arr.shape == (4 + e1.NC, 8400):
        arr = arr.T
    elif arr.shape != (8400, 4 + e1.NC):
        raise RuntimeError(
            "Unexpected output shape: {}".format(
                arr.shape
            )
        )

    scores = arr[:, 4:].max(axis=1)
    classes = arr[:, 4:].argmax(axis=1)

    valid = scores > SCORE_THRESHOLD

    arr = arr[valid]
    scores = scores[valid]
    classes = classes[valid]

    boxes = np.empty((len(arr), 4), np.float32)

    if len(arr):

        boxes[:, :2] = (
            arr[:, :2] - arr[:, 2:4] / 2
        )

        boxes[:, 2:] = (
            arr[:, :2] + arr[:, 2:4] / 2
        )

    decode_ms = (now() - t) * 1000

    t = now()

    groups = []

    before_count = len(scores)

    for c in range(e1.NC):

        idx = np.flatnonzero(classes == c)

        if not len(idx):
            continue

        if topk is not None and len(idx) > topk:

            local = np.argpartition(
                -scores[idx],
                topk - 1
            )[:topk]

            idx = idx[local]

        groups.append(idx)

    topk_ms = (now() - t) * 1000

    candidate_count = sum(
        len(idx) for idx in groups
    )

    t = now()

    selected_ids = []
    selected_scores = []

    for idx in groups:

        local_ids, local_scores = soft_nms_single_class(
            boxes[idx],
            scores[idx],
            method
        )

        if len(local_ids):

            selected_ids.append(idx[local_ids])
            selected_scores.append(local_scores)

    nms_ms = (now() - t) * 1000

    t = now()

    if selected_ids:

        chosen = np.concatenate(selected_ids)

        updated_scores = np.concatenate(
            selected_scores
        )

        order = np.argsort(
            -updated_scores
        )[:MAX_DET]

        chosen = chosen[order]
        updated_scores = updated_scores[order]

        final_boxes = boxes[chosen].copy()

        final_boxes[:, [0, 2]] = np.clip(
            (
                final_boxes[:, [0, 2]] - left
            ) / ratio,
            0,
            width
        )

        final_boxes[:, [1, 3]] = np.clip(
            (
                final_boxes[:, [1, 3]] - top
            ) / ratio,
            0,
            height
        )

        pred = np.column_stack(
            (
                final_boxes,
                updated_scores,
                classes[chosen]
            )
        ).astype(np.float32)

    else:
        pred = np.zeros((0, 6), np.float32)

    restore_ms = (now() - t) * 1000

    total_ms = (now() - total_start) * 1000

    return pred, {
        "decode_ms": decode_ms,
        "topk_ms": topk_ms,
        "nms_ms": nms_ms,
        "restore_ms": restore_ms,
        "post_total_ms": total_ms,
        "candidates_before": before_count,
        "candidates_after": candidate_count,
        "kept": len(pred),
    }


def get_predictions(item, algorithm, topk):

    if algorithm == "greedy":

        t = now()

        pred, details = e1.postprocess_topk(
            item["raw"],
            item["width"],
            item["height"],
            item["ratio"],
            item["left"],
            item["top"],
            topk
        )

        details["post_total_ms"] = (
            now() - t
        ) * 1000

        return pred, details

    return postprocess_soft(
        item["raw"],
        item["width"],
        item["height"],
        item["ratio"],
        item["left"],
        item["top"],
        algorithm,
        topk
    )


def summarize(records):

    keys = (
        "decode_ms",
        "topk_ms",
        "nms_ms",
        "restore_ms",
        "post_total_ms",
    )

    result = {}

    for key in keys:

        values = np.array(
            [r[key] for r in records],
            dtype=np.float64
        )

        result[key] = {
            "mean_ms": float(np.mean(values)),
            "median_ms": float(np.median(values)),
            "p95_ms": float(
                np.percentile(values, 95)
            ),
        }

    return result


def run_one_method(
    outputs,
    gts,
    method_name,
    algorithm,
    topk,
    repeats
):

    records = []
    final_preds = None
    counts = []

    started = now()

    for rep in range(repeats):

        current_preds = []

        for item in outputs:

            pred, info = get_predictions(
                item,
                algorithm,
                topk
            )

            current_preds.append(pred)
            records.append(info)

            if rep == 0:
                counts.append(
                    info["candidates_after"]
                )

        if final_preds is None:
            final_preds = current_preds

    metrics = e1.evaluate_map(
        final_preds,
        gts
    )

    timing = summarize(records)

    return {
        "method": method_name,
        "algorithm": algorithm,
        "topk": topk,
        "sigma": (
            SIGMA if algorithm == "gaussian" else None
        ),
        "metrics": metrics,
        "timing": timing,
        "avg_candidates_after": float(
            np.mean(counts)
        ),
        "elapsed_seconds": float(
            now() - started
        ),
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
        "--skip-c",
        action="store_true",
        help="Skip E2-C if desired"
    )

    args = parser.parse_args()

    if args.repeats < 1 or args.max_images < 0:
        parser.error("Invalid parameters")

    folder, paths = e1.get_image_paths()

    if args.max_images:
        paths = paths[:args.max_images]

    print("Validation images:", len(paths))
    print("Postprocess repeats:", args.repeats)
    print("Soft-NMS sigma:", SIGMA)

    cuda = e1.base.Cuda()

    all_results = {}
    csv_rows = []

    for model_name, engine_path in e1.MODELS.items():

        outputs, gts = e1.collect_outputs(
            model_name,
            engine_path,
            folder,
            paths,
            cuda
        )

        all_results[model_name] = {}

        for method_name, (algorithm, topk) in METHODS.items():

            if method_name == "E2-C" and args.skip_c:
                continue

            print(
                "\nRunning {} / {}".format(
                    model_name,
                    method_name
                ),
                flush=True
            )

            result = run_one_method(
                outputs,
                gts,
                method_name,
                algorithm,
                topk,
                args.repeats
            )

            all_results[model_name][method_name] = result

            m = result["metrics"]
            t = result["timing"]

            print(
                "mAP50={:.4f}  "
                "mAP50:95={:.4f}  "
                "NMS={:.3f}ms  "
                "Post={:.3f}ms  "
                "Elapsed={:.1f}s".format(
                    m["mAP50"],
                    m["mAP50_95"],
                    t["nms_ms"]["mean_ms"],
                    t["post_total_ms"]["mean_ms"],
                    result["elapsed_seconds"]
                ),
                flush=True
            )

            csv_rows.append({
                "model": model_name,
                "method": method_name,
                "algorithm": algorithm,
                "topk": (
                    topk if topk is not None else "unlimited"
                ),
                "mAP50": m["mAP50"],
                "mAP50_95": m["mAP50_95"],
                "topk_ms": t["topk_ms"]["mean_ms"],
                "nms_ms": t["nms_ms"]["mean_ms"],
                "post_ms": t["post_total_ms"]["mean_ms"],
                "candidates": result[
                    "avg_candidates_after"
                ],
                "elapsed_seconds": result[
                    "elapsed_seconds"
                ],
            })

    output_json = ROOT / "E2_softnms_results.json"
    output_csv = ROOT / "E2_softnms_results.csv"

    output_json.write_text(
        json.dumps(
            all_results,
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
            fieldnames=csv_rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(csv_rows)

    print("\n" + "=" * 76)
    print("E2 FINAL RESULTS")
    print("=" * 76)

    for row in csv_rows:

        print(
            "{:<5} {:<5} "
            "mAP50={:.4f} "
            "mAP50:95={:.4f} "
            "NMS={:.3f}ms "
            "Post={:.3f}ms "
            "Wall={:.1f}s".format(
                row["model"],
                row["method"],
                row["mAP50"],
                row["mAP50_95"],
                row["nms_ms"],
                row["post_ms"],
                row["elapsed_seconds"]
            )
        )

    print("\nSaved:", output_json)
    print("Saved:", output_csv)


if __name__ == "__main__":
    main()
