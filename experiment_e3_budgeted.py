
# -*- coding: utf-8 -*-
"""
E3: Budgeted Gaussian Soft-NMS + Greedy NMS

E3-A: M=30
E3-B: M=60
E3-C: M=100

K=200 candidates per class.

Reuses E1/E2 inference and AP evaluation.
No PyTorch, no new TensorRT engine.
"""

import argparse
import json
import csv
import time
from pathlib import Path

import cv2
import numpy as np

import experiment_e1_topk as e1
import experiment_e2_softnms as e2


ROOT = Path(__file__).resolve().parent

K = 200
SIGMA = 0.5

METHODS = {
    "E1-C": 0,
    "E3-A": 30,
    "E3-B": 60,
    "E3-C": 100,
    "E2-C": 200,
}


def iou_one_to_many(box, boxes):
    return e2.iou_one_to_many(box, boxes)


def hybrid_single_class(boxes, scores, budget):
    """
    Select at most `budget` boxes using Gaussian
    Soft-NMS, then process residual candidates
    using OpenCV Greedy NMS.

    Returns local indices, adjusted scores.
    """

    n = len(scores)

    if n == 0:
        return (
            np.empty(0, np.int32),
            np.empty(0, np.float32),
        )

    working = np.asarray(
        scores, dtype=np.float32
    ).copy()

    active = np.arange(n, dtype=np.int32)

    selected = []
    selected_scores = []

    # Soft-NMS stage
    for _ in range(min(budget, n)):

        if not len(active):
            break

        pos = int(np.argmax(working[active]))
        best = int(active[pos])
        best_score = float(working[best])

        if best_score <= e1.CONF:
            break

        selected.append(best)
        selected_scores.append(best_score)

        active = np.delete(active, pos)

        if not len(active):
            break

        ious = iou_one_to_many(
            boxes[best],
            boxes[active]
        )

        working[active] *= np.exp(
            -(ious * ious) / SIGMA
        )

        active = active[
            working[active] > e1.CONF
        ]

    # Greedy NMS stage on residual candidates.
    if len(active):

        xywh = np.column_stack(
            (
                boxes[active, :2],
                np.maximum(
                    0,
                    boxes[active, 2:] - boxes[active, :2]
                )
            )
        )

        keep = cv2.dnn.NMSBoxes(
            xywh.tolist(),
            working[active].tolist(),
            e1.CONF,
            e1.IOU_NMS
        )

        if keep is not None and len(keep):

            ids = active[
                np.asarray(keep).reshape(-1)
            ]

            selected.extend(ids.tolist())

            selected_scores.extend(
                working[ids].tolist()
            )

    return (
        np.asarray(selected, np.int32),
        np.asarray(selected_scores, np.float32)
    )


def postprocess_hybrid(item, budget):
    """
    Return predictions [N,6] in original image
    coordinates and total postprocessing latency.
    """

    start = time.perf_counter()

    raw = item["raw"]

    arr = raw[0].astype(np.float32)

    if arr.shape == (4 + e1.NC, 8400):
        arr = arr.T
    elif arr.shape != (8400, 4 + e1.NC):
        raise RuntimeError(
            "Unexpected output {}".format(arr.shape)
        )

    scores = arr[:, 4:].max(axis=1)
    classes = arr[:, 4:].argmax(axis=1)

    mask = scores > e1.CONF

    arr = arr[mask]
    scores = scores[mask]
    classes = classes[mask]

    boxes = np.empty((len(arr), 4), np.float32)

    if len(arr):
        boxes[:, :2] = (
            arr[:, :2] - arr[:, 2:4] / 2
        )
        boxes[:, 2:] = (
            arr[:, :2] + arr[:, 2:4] / 2
        )

    selected = []
    adjusted = []

    for c in range(e1.NC):

        idx = np.flatnonzero(classes == c)

        if not len(idx):
            continue

        if len(idx) > K:
            subset = np.argpartition(
                -scores[idx], K-1
            )[:K]
            idx = idx[subset]

        ids, new_scores = hybrid_single_class(
            boxes[idx],
            scores[idx],
            budget
        )

        if len(ids):
            selected.append(idx[ids])
            adjusted.append(new_scores)

    if selected:

        selected = np.concatenate(selected)
        adjusted = np.concatenate(adjusted)

        order = np.argsort(
            -adjusted
        )[:e1.MAX_DET]

        selected = selected[order]
        adjusted = adjusted[order]

        final_boxes = boxes[selected].copy()

        final_boxes[:, [0, 2]] = np.clip(
            (
                final_boxes[:, [0, 2]]
                - item["left"]
            ) / item["ratio"],
            0,
            item["width"]
        )

        final_boxes[:, [1, 3]] = np.clip(
            (
                final_boxes[:, [1, 3]]
                - item["top"]
            ) / item["ratio"],
            0,
            item["height"]
        )

        pred = np.column_stack(
            (
                final_boxes,
                adjusted,
                classes[selected]
            )
        ).astype(np.float32)

    else:
        pred = np.zeros((0, 6), np.float32)

    total_ms = (
        time.perf_counter() - start
    ) * 1000

    return pred, total_ms


def same_predictions(a, b):
    return (
        a.shape == b.shape
        and np.allclose(
            a, b,
            rtol=1e-5,
            atol=1e-4
        )
    )


def validate_boundaries(item):
    """
    M=0 should match E1-C Greedy.
    M=200 should match E2-C Gaussian.
    """

    a, _ = postprocess_hybrid(item, 0)

    b, _ = e1.postprocess_topk(
        item["raw"],
        item["width"],
        item["height"],
        item["ratio"],
        item["left"],
        item["top"],
        K
    )

    if not same_predictions(a, b):
        raise RuntimeError(
            "M=0 does not match E1-C"
        )

    c, _ = postprocess_hybrid(item, 200)

    d, _ = e2.postprocess_soft(
        item["raw"],
        item["width"],
        item["height"],
        item["ratio"],
        item["left"],
        item["top"],
        "gaussian",
        K
    )

    if not same_predictions(c, d):
        raise RuntimeError(
            "M=200 does not match E2-C"
        )


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

    args = parser.parse_args()

    folder, paths = e1.get_image_paths()

    if args.max_images:
        paths = paths[:args.max_images]

    if not paths or args.repeats < 1:
        raise RuntimeError(
            "No images or invalid repeats"
        )

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

        # Check both boundary cases before running E3.
        for item in outputs:
            validate_boundaries(item)

        print(
            "PASS: M=0 equals E1-C; "
            "M=200 equals E2-C",
            flush=True
        )

        results[model_name] = {}

        for method, budget in METHODS.items():

            print(
                "\n{} / {} M={}".format(
                    model_name,
                    method,
                    budget
                ),
                flush=True
            )

            timings = []
            predictions = None

            for rep in range(args.repeats):

                current = []

                for item in outputs:

                    pred, ms = postprocess_hybrid(
                        item,
                        budget
                    )

                    current.append(pred)
                    timings.append(ms)

                if predictions is None:
                    predictions = current

            metrics = e1.evaluate_map(
                predictions,
                gts
            )

            record = {
                "mAP50": metrics["mAP50"],
                "mAP50_95": metrics["mAP50_95"],
                "post_ms_mean": float(
                    np.mean(timings)
                ),
                "post_ms_p95": float(
                    np.percentile(timings, 95)
                ),
                "budget": budget,
            }

            results[model_name][method] = record

            rows.append({
                "model": model_name,
                "method": method,
                **record,
            })

            print(
                "mAP50={:.4f} "
                "mAP50:95={:.4f} "
                "Post={:.3f}ms".format(
                    record["mAP50"],
                    record["mAP50_95"],
                    record["post_ms_mean"]
                ),
                flush=True
            )

    json_path = ROOT / "E3_budgeted_results.json"
    csv_path = ROOT / "E3_budgeted_results.csv"

    json_path.write_text(
        json.dumps(
            results,
            indent=2
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

    print("\n" + "=" * 65)
    print("E3 FINAL RESULTS")
    print("=" * 65)

    for r in rows:
        print(
            "{:<5} {:<5} M={:<3} "
            "mAP50:95={:.4f} "
            "Post={:.3f}ms".format(
                r["model"],
                r["method"],
                r["budget"],
                r["mAP50_95"],
                r["post_ms_mean"]
            )
        )

    print("\nSaved:", json_path)
    print("Saved:", csv_path)


if __name__ == "__main__":
    main()
