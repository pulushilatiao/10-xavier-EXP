
# -*- coding: utf-8 -*-
"""
E4: OpenCV compiled Soft-NMS comparison
Jetson Xavier, no torch / PyCUDA

E1-C: TopK=200 + OpenCV Greedy
E2-C: TopK=200 + Python Gaussian Soft-NMS
E4-A: TopK=200 + OpenCV Gaussian Soft-NMS
E4-B: Unlimited + OpenCV Gaussian Soft-NMS

Reuses E1 TensorRT output collection and AP evaluator.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

import experiment_e1_topk as e1
import experiment_e2_softnms as e2


ROOT = Path(__file__).resolve().parent

TOPK = 200
SIGMA = 0.5

METHODS = {
    "E1-C": ("greedy", TOPK),
    "E2-C": ("python", TOPK),
    "E4-A": ("opencv", TOPK),
    "E4-B": ("opencv", None),
}


def timestamp():
    return time.perf_counter()


def opencv_soft_postprocess(item, topk):
    """
    OpenCV compiled Gaussian Soft-NMS.

    Output:
        predictions [N,6]:
        x1 y1 x2 y2 updated_score class
    """

    start = timestamp()

    arr = item["raw"][0].astype(np.float32)

    if arr.shape == (4 + e1.NC, 8400):
        arr = arr.T
    elif arr.shape != (8400, 4 + e1.NC):
        raise RuntimeError(
            "Unexpected output: {}".format(arr.shape)
        )

    scores = arr[:, 4:].max(axis=1)
    classes = arr[:, 4:].argmax(axis=1)

    valid = scores > e1.CONF

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

    decoded_ms = (
        timestamp() - start
    ) * 1000.0

    t = timestamp()

    groups = []

    for c in range(e1.NC):

        idx = np.flatnonzero(classes == c)

        if len(idx) == 0:
            continue

        if topk is not None and len(idx) > topk:

            chosen = np.argpartition(
                -scores[idx],
                topk - 1
            )[:topk]

            idx = idx[chosen]

        groups.append(idx)

    topk_ms = (timestamp() - t) * 1000.0

    t = timestamp()

    picked_ids = []
    picked_scores = []

    for idx in groups:

        xywh = np.column_stack(
            (
                boxes[idx, :2],
                np.maximum(
                    0.0,
                    boxes[idx, 2:] - boxes[idx, :2]
                )
            )
        )

        # OpenCV softNMSBoxes accepts integer Rect.
        # Round coordinates consistently.
        rects = np.rint(xywh).astype(np.int32)
        rects[:, 2:] = np.maximum(rects[:, 2:], 0)

        # return order: updated_scores, indices
        updated, keep = cv2.dnn.softNMSBoxes(
            rects.tolist(),
            scores[idx].astype(
                np.float32
            ).tolist(),
            e1.CONF,
            e1.IOU_NMS,
            300,       # max selected per class
            SIGMA      # Gaussian default method
        )

        keep = np.asarray(
            keep,
            dtype=np.int32
        ).reshape(-1)

        updated = np.asarray(
            updated,
            dtype=np.float32
        ).reshape(-1)

        if len(keep) != len(updated):
            raise RuntimeError(
                "OpenCV scores/indices mismatch"
            )

        if len(keep):
            picked_ids.append(idx[keep])
            picked_scores.append(updated)

    nms_ms = (timestamp() - t) * 1000.0

    t = timestamp()

    if picked_ids:

        picked = np.concatenate(picked_ids)
        adjusted = np.concatenate(picked_scores)

        order = np.argsort(
            -adjusted
        )[:e1.MAX_DET]

        picked = picked[order]
        adjusted = adjusted[order]

        # Original float boxes are preserved
        # for final coordinate restoration.
        b = boxes[picked].copy()

        b[:, [0, 2]] = np.clip(
            (
                b[:, [0, 2]] - item["left"]
            ) / item["ratio"],
            0,
            item["width"]
        )

        b[:, [1, 3]] = np.clip(
            (
                b[:, [1, 3]] - item["top"]
            ) / item["ratio"],
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
        pred = np.empty((0, 6), np.float32)

    restore_ms = (timestamp() - t) * 1000.0

    return pred, {
        "decode_ms": decoded_ms,
        "topk_ms": topk_ms,
        "nms_ms": nms_ms,
        "restore_ms": restore_ms,
        "post_ms": (
            timestamp() - start
        ) * 1000.0,
        "candidates": sum(
            len(idx) for idx in groups
        ),
    }


def process(item, kind, topk):

    if kind == "opencv":
        return opencv_soft_postprocess(item, topk)

    t = timestamp()

    if kind == "greedy":

        pred, details = e1.postprocess_topk(
            item["raw"],
            item["width"],
            item["height"],
            item["ratio"],
            item["left"],
            item["top"],
            topk
        )

    elif kind == "python":

        pred, details = e2.postprocess_soft(
            item["raw"],
            item["width"],
            item["height"],
            item["ratio"],
            item["left"],
            item["top"],
            "gaussian",
            topk
        )

    else:
        raise RuntimeError(kind)

    details["post_ms"] = (
        timestamp() - t
    ) * 1000.0

    return pred, details


def run_method(outputs, gts, kind, topk, repeats):

    records = []
    predictions = None

    for repeat in range(repeats):

        current = []

        for item in outputs:

            pred, details = process(
                item,
                kind,
                topk
            )

            current.append(pred)
            records.append(details)

        if predictions is None:
            predictions = current

    metrics = e1.evaluate_map(
        predictions,
        gts
    )

    summary = {}

    for key in ("nms_ms", "post_ms"):

        vals = np.asarray(
            [x[key] for x in records],
            dtype=np.float64
        )

        summary[key] = {
            "mean": float(vals.mean()),
            "p95": float(
                np.percentile(vals, 95)
            ),
        }

    return {
        "metrics": metrics,
        "timing": summary,
        "samples": len(outputs),
        "repeats": repeats,
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
        default=5
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
    print("Sigma:", SIGMA)

    cuda = e1.base.Cuda()

    results = {}
    rows = []

    for model_name, engine_path in e1.MODELS.items():

        outputs, gts = e1.collect_outputs(
            model_name,
            engine_path,
            folder,
            paths,
            cuda
        )

        results[model_name] = {}

        for method_name, (kind, topk) in METHODS.items():

            print(
                "\nRunning {} {}".format(
                    model_name,
                    method_name
                ),
                flush=True
            )

            result = run_method(
                outputs,
                gts,
                kind,
                topk,
                args.repeats
            )

            results[model_name][method_name] = result

            m = result["metrics"]
            t = result["timing"]

            row = {
                "model": model_name,
                "method": method_name,
                "topk": (
                    topk if topk is not None else "unlimited"
                ),
                "mAP50": m["mAP50"],
                "mAP50_95": m["mAP50_95"],
                "nms_ms": t["nms_ms"]["mean"],
                "post_ms": t["post_ms"]["mean"],
                "post_p95_ms": t["post_ms"]["p95"],
            }

            rows.append(row)

            print(
                "mAP50={:.4f} "
                "mAP50:95={:.4f} "
                "NMS={:.3f}ms "
                "Post={:.3f}ms".format(
                    row["mAP50"],
                    row["mAP50_95"],
                    row["nms_ms"],
                    row["post_ms"]
                ),
                flush=True
            )

    json_path = ROOT / "E4_opencv_softnms.json"
    csv_path = ROOT / "E4_opencv_softnms.csv"

    json_path.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    with open(
        csv_path,
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

    print("\n" + "=" * 66)
    print("E4 FINAL RESULTS")
    print("=" * 66)

    for r in rows:

        print(
            "{:<5} {:<5} "
            "mAP50:95={:.4f} "
            "NMS={:.3f}ms "
            "Post={:.3f}ms".format(
                r["model"],
                r["method"],
                r["mAP50_95"],
                r["nms_ms"],
                r["post_ms"]
            )
        )

    print("\nSaved:", json_path)
    print("Saved:", csv_path)


if __name__ == "__main__":
    main()
