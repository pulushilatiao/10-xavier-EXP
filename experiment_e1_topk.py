
# -*- coding: utf-8 -*-
"""
E0 / E1-A / E1-B / E1-C
Class-aware pre-NMS Top-K experiment

Jetson Xavier / TensorRT 8.5.x
No torch / ultralytics / pycuda

E0    : unlimited pre-NMS candidates
E1-A  : Top-K 1000 per class
E1-B  : Top-K 500 per class
E1-C  : Top-K 200 per class

Models:
    1012_fp16.engine
    1012_int8.engine

Outputs:
    E1_topk_results.json
    E1_topk_results.csv
"""

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

import eval_precision_1012TEST as base


# =========================================
# CONFIG
# =========================================

ROOT = Path(__file__).resolve().parent

MODELS = {
    "FP16": ROOT / "1012_fp16.engine",
    "INT8": ROOT / "1012_int8.engine",
}

VAL_IMAGES = ROOT / "val" / "images"
VAL_LABELS = ROOT / "val" / "labels"

CONF = 0.001
IOU_NMS = 0.70
MAX_DET = 300

NC = 3
CLASS_NAMES = ["tomato", "leaf", "stem"]

METHODS = {
    "E0": None,
    "E1-A": 1000,
    "E1-B": 500,
    "E1-C": 200,
}

IOU_THRESHOLDS = np.arange(
    0.50, 0.96, 0.05
)

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# =========================================
# HELPERS
# =========================================

def tick():
    return time.perf_counter()


def elapsed_ms(t):
    return (tick() - t) * 1000.0


def mean(values):
    return float(np.mean(values))


def p95(values):
    return float(np.percentile(values, 95))


def get_image_paths():

    folder = VAL_IMAGES

    if not folder.is_dir():
        folder = ROOT / "val"

    if not folder.is_dir():
        raise RuntimeError("Validation images not found")

    if not VAL_LABELS.is_dir():
        raise RuntimeError(
            "Validation labels not found: {}".format(
                VAL_LABELS
            )
        )

    paths = sorted(
        p for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in EXTS
    )

    if not paths:
        raise RuntimeError("No validation images")

    return folder, paths


def load_gt(path, width, height):
    """
    YOLO TXT:
      class cx cy w h
    Normalized coordinates.

    Returns [N,5]:
      x1 y1 x2 y2 class
    """

    if not path.exists():
        raise FileNotFoundError(path)

    objs = []

    for line in path.read_text().splitlines():
        parts = line.strip().split()

        if not parts:
            continue

        if len(parts) != 5:
            raise RuntimeError(
                "Invalid YOLO label: {}".format(path)
            )

        c = int(parts[0])
        x, y, w, h = map(float, parts[1:])

        if not 0 <= c < NC:
            raise RuntimeError(
                "Invalid class {}: {}".format(c, path)
            )

        x1 = (x - w / 2) * width
        y1 = (y - h / 2) * height
        x2 = (x + w / 2) * width
        y2 = (y + h / 2) * height

        objs.append([x1, y1, x2, y2, c])

    if not objs:
        return np.zeros((0, 5), np.float32)

    return np.asarray(objs, dtype=np.float32)


# =========================================
# TOP-K + NMS
# =========================================

def postprocess_topk(
    raw,
    width,
    height,
    ratio,
    left,
    top,
    topk
):
    """
    raw: [1,7,8400]

    Returns:
      predictions [N,6]
      x1 y1 x2 y2 confidence class
      timing details

    All candidate selection is class-aware.
    """

    t0 = tick()

    a = raw[0].astype(np.float32)

    if a.shape == (4 + NC, 8400):
        a = a.T
    elif a.shape != (8400, 4 + NC):
        raise RuntimeError(
            "Invalid YOLO output: {}".format(a.shape)
        )

    scores = a[:, 4:].max(axis=1)
    classes = a[:, 4:].argmax(axis=1)

    valid = scores > CONF

    a = a[valid]
    scores = scores[valid]
    classes = classes[valid]

    if len(a) == 0:
        return (
            np.zeros((0, 6), np.float32),
            {
                "decode_ms": elapsed_ms(t0),
                "topk_ms": 0.0,
                "nms_ms": 0.0,
                "restore_ms": 0.0,
                "candidates_before": 0,
                "candidates_after": 0,
                "after_nms": 0,
                "kept": 0,
            }
        )

    boxes = np.empty((len(a), 4), np.float32)

    boxes[:, :2] = a[:, :2] - a[:, 2:4] / 2
    boxes[:, 2:] = a[:, :2] + a[:, 2:4] / 2

    decode_ms = elapsed_ms(t0)

    # -------------------------------------
    # Pre-NMS Top-K (per class)
    # -------------------------------------

    t1 = tick()

    groups = []
    before = len(a)

    for c in range(NC):

        idx = np.flatnonzero(classes == c)

        if not len(idx):
            continue

        if topk is not None and len(idx) > topk:

            # Partial selection: O(N) average.
            # Final sorting is not necessary here,
            # because NMSBoxes orders by confidence.
            local = np.argpartition(
                -scores[idx],
                topk - 1
            )[:topk]

            idx = idx[local]

        groups.append(idx)

    topk_ms = elapsed_ms(t1)

    candidates_after = sum(
        len(group) for group in groups
    )

    # -------------------------------------
    # Same OpenCV class-aware Greedy NMS
    # -------------------------------------

    t2 = tick()

    picks = []

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

        keep = cv2.dnn.NMSBoxes(
            xywh.tolist(),
            scores[idx].tolist(),
            CONF,
            IOU_NMS
        )

        if keep is not None and len(keep):

            local_keep = np.asarray(
                keep
            ).reshape(-1)

            picks.extend(
                idx[local_keep].tolist()
            )

    nms_ms = elapsed_ms(t2)

    after_nms = len(picks)

    # -------------------------------------
    # Output limit and coordinate restore
    # -------------------------------------

    t3 = tick()

    if not picks:

        pred = np.zeros((0, 6), np.float32)

    else:

        picks = np.asarray(picks, dtype=int)

        order = np.argsort(
            -scores[picks]
        )[:MAX_DET]

        picks = picks[order]

        b = boxes[picks].copy()

        b[:, [0, 2]] = np.clip(
            (b[:, [0, 2]] - left) / ratio,
            0,
            width
        )

        b[:, [1, 3]] = np.clip(
            (b[:, [1, 3]] - top) / ratio,
            0,
            height
        )

        pred = np.column_stack(
            (
                b,
                scores[picks],
                classes[picks]
            )
        ).astype(np.float32)

    restore_ms = elapsed_ms(t3)

    return pred, {
        "decode_ms": decode_ms,
        "topk_ms": topk_ms,
        "nms_ms": nms_ms,
        "restore_ms": restore_ms,
        "candidates_before": before,
        "candidates_after": candidates_after,
        "after_nms": after_nms,
        "kept": len(pred),
    }


# =========================================
# AP CALCULATION
# =========================================

def box_iou_one_to_many(box, others):

    if len(others) == 0:
        return np.zeros(0, np.float32)

    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])

    inter = (
        np.maximum(0, x2 - x1)
        * np.maximum(0, y2 - y1)
    )

    area1 = max(
        0, box[2]-box[0]
    ) * max(0, box[3]-box[1])

    area2 = (
        np.maximum(0, others[:, 2]-others[:, 0])
        * np.maximum(0, others[:, 3]-others[:, 1])
    )

    return inter / np.maximum(
        area1 + area2 - inter,
        1e-9
    )


def ap101(recalls, precisions):

    # Precision envelope
    mrec = np.concatenate(
        ([0.0], recalls, [1.0])
    )
    mpre = np.concatenate(
        ([1.0], precisions, [0.0])
    )

    mpre = np.maximum.accumulate(
        mpre[::-1]
    )[::-1]

    grid = np.linspace(0, 1, 101)

    return float(
        np.trapz(
            np.interp(grid, mrec, mpre),
            grid
        )
    )


def evaluate_class_iou(
    predictions,
    gts,
    class_id,
    threshold
):

    detections = []
    gt_by_image = {}
    total_gt = 0

    for image_id, (pred, gt) in enumerate(
        zip(predictions, gts)
    ):

        gt_c = gt[gt[:, 4] == class_id]

        gt_by_image[image_id] = gt_c

        total_gt += len(gt_c)

        pred_c = pred[pred[:, 5] == class_id]

        for p in pred_c:
            detections.append(
                (float(p[4]), image_id, p[:4])
            )

    if total_gt == 0:
        return None

    detections.sort(
        key=lambda item: -item[0]
    )

    matched = {
        image_id: np.zeros(
            len(gt),
            dtype=bool
        )
        for image_id, gt in gt_by_image.items()
    }

    tp = np.zeros(len(detections))
    fp = np.zeros(len(detections))

    for i, (_, image_id, box) in enumerate(detections):

        gt = gt_by_image[image_id]

        if len(gt) == 0:
            fp[i] = 1
            continue

        ious = box_iou_one_to_many(
            box, gt[:, :4]
        )

        j = int(np.argmax(ious))

        if (
            ious[j] >= threshold
            and not matched[image_id][j]
        ):
            tp[i] = 1
            matched[image_id][j] = True
        else:
            fp[i] = 1

    if not len(detections):
        return 0.0

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)

    recalls = tp / max(total_gt, 1)
    precisions = tp / np.maximum(tp + fp, 1e-9)

    return ap101(recalls, precisions)


def evaluate_map(predictions, gts):

    ap = np.full(
        (NC, len(IOU_THRESHOLDS)),
        np.nan,
        dtype=np.float64
    )

    for c in range(NC):

        for j, threshold in enumerate(IOU_THRESHOLDS):

            value = evaluate_class_iou(
                predictions,
                gts,
                c,
                float(threshold)
            )

            if value is not None:
                ap[c, j] = value

    if np.all(np.isnan(ap)):
        raise RuntimeError("No GT classes")

    result = {
        "mAP50": float(np.nanmean(ap[:, 0])),
        "mAP50_95": float(np.nanmean(ap)),
        "per_class": {},
    }

    for c, name in enumerate(CLASS_NAMES):

        if np.all(np.isnan(ap[c])):
            result["per_class"][name] = None
        else:
            result["per_class"][name] = {
                "AP50": float(ap[c, 0]),
                "AP50_95": float(
                    np.nanmean(ap[c])
                ),
            }

    return result


# =========================================
# COLLECT INFERENCE OUTPUTS ONCE
# =========================================

def collect_outputs(
    model_name,
    engine_path,
    images_folder,
    paths,
    cuda
):

    print("\nLoading:", model_name, flush=True)

    engine = base.Engine(engine_path, cuda)

    outputs = []
    gts = []

    try:

        if engine.nc != NC:
            raise RuntimeError(
                "Expected {} classes, got {}".format(
                    NC, engine.nc
                )
            )

        first = cv2.imread(str(paths[0]))

        if first is None:
            raise RuntimeError("Cannot read first image")

        warm, _, _, _ = base.preprocess(first)

        for _ in range(10):
            engine.infer(warm)

        for i, path in enumerate(paths, 1):

            img = cv2.imread(str(path))

            if img is None:
                raise RuntimeError(path)

            h, w = img.shape[:2]

            inp, ratio, left, top = base.preprocess(img)

            raw, gpu_ms, _ = engine.infer(inp)

            raw = np.array(raw, copy=True)

            label_rel = (
                path.relative_to(images_folder)
                .with_suffix(".txt")
            )

            label_path = VAL_LABELS / label_rel

            gt = load_gt(label_path, w, h)

            outputs.append({
                "raw": raw,
                "width": w,
                "height": h,
                "ratio": ratio,
                "left": left,
                "top": top,
                "name": path.name,
            })

            gts.append(gt)

            print(
                "Collect {}/{} {} gpu={:.3f}ms".format(
                    i, len(paths),
                    path.name, gpu_ms
                ),
                flush=True
            )

        # Crucial validation: E0 predictions must
        # match the original successful evaluator.
        for item in outputs:

            original = base.postprocess(
                item["raw"],
                (item["width"], item["height"]),
                item["ratio"],
                item["left"],
                item["top"],
                NC
            )

            reference, _ = postprocess_topk(
                item["raw"],
                item["width"],
                item["height"],
                item["ratio"],
                item["left"],
                item["top"],
                None
            )

            if (
                original.shape != reference.shape
                or not np.allclose(
                    original,
                    reference,
                    rtol=1e-5,
                    atol=1e-4
                )
            ):
                raise RuntimeError(
                    "E0 postprocess mismatch: {}".format(
                        item["name"]
                    )
                )

        print(
            "PASS: E0 matches original postprocess "
            "for all images",
            flush=True
        )

        return outputs, gts

    finally:
        engine.close()


# =========================================
# RUN E0/E1 ON CACHED ENGINE OUTPUTS
# =========================================

def run_methods(
    model_name,
    outputs,
    gts,
    repeats
):

    results = {}

    for method, topk in METHODS.items():

        print(
            "\n{} / {} (Top-K={})".format(
                model_name, method, topk
            ),
            flush=True
        )

        all_times = {
            "decode_ms": [],
            "topk_ms": [],
            "nms_ms": [],
            "restore_ms": [],
            "post_total_ms": [],
        }

        predictions = None
        candidate_before = []
        candidate_after = []

        for repeat_idx in range(repeats):

            current_predictions = []

            for item in outputs:

                t = tick()

                pred, detail = postprocess_topk(
                    item["raw"],
                    item["width"],
                    item["height"],
                    item["ratio"],
                    item["left"],
                    item["top"],
                    topk
                )

                total_ms = elapsed_ms(t)

                current_predictions.append(pred)

                for key in (
                    "decode_ms",
                    "topk_ms",
                    "nms_ms",
                    "restore_ms"
                ):
                    all_times[key].append(detail[key])

                all_times["post_total_ms"].append(
                    total_ms
                )

                if repeat_idx == 0:
                    candidate_before.append(
                        detail["candidates_before"]
                    )
                    candidate_after.append(
                        detail["candidates_after"]
                    )

            if predictions is None:
                predictions = current_predictions

        metrics = evaluate_map(
            predictions, gts
        )

        timing_summary = {}

        for key, values in all_times.items():
            timing_summary[key] = {
                "mean_ms": mean(values),
                "p95_ms": p95(values),
            }

        result = {
            "topk": topk,
            "metrics": metrics,
            "timing": timing_summary,
            "avg_candidates_before": mean(
                candidate_before
            ),
            "avg_candidates_after": mean(
                candidate_after
            ),
        }

        results[method] = result

        print(
            "mAP50={:.4f}, mAP50:95={:.4f}, "
            "NMS={:.3f}ms, TopK={:.3f}ms, "
            "Post={:.3f}ms".format(
                metrics["mAP50"],
                metrics["mAP50_95"],
                timing_summary["nms_ms"]["mean_ms"],
                timing_summary["topk_ms"]["mean_ms"],
                timing_summary["post_total_ms"]["mean_ms"]
            ),
            flush=True
        )

    return results


# =========================================
# MAIN
# =========================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repeats",
        type=int,
        default=10
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=0
    )

    args = parser.parse_args()

    if args.repeats < 1 or args.max_images < 0:
        parser.error("Invalid arguments")

    folder, paths = get_image_paths()

    if args.max_images:
        paths = paths[:args.max_images]

    print("Images:", len(paths))
    print("Postprocess repeats:", args.repeats)
    print("Confidence:", CONF)
    print("NMS IoU:", IOU_NMS)
    print("MAX_DET:", MAX_DET)

    cuda = base.Cuda()

    final = {}

    for model_name, engine_path in MODELS.items():

        if not engine_path.is_file():
            raise FileNotFoundError(engine_path)

        outputs, gts = collect_outputs(
            model_name,
            engine_path,
            folder,
            paths,
            cuda
        )

        print(
            "GT objects:",
            sum(len(gt) for gt in gts)
        )

        final[model_name] = run_methods(
            model_name,
            outputs,
            gts,
            args.repeats
        )

    output_json = ROOT / "E1_topk_results.json"
    output_csv = ROOT / "E1_topk_results.csv"

    output_json.write_text(
        json.dumps(final, indent=2),
        encoding="utf-8"
    )

    rows = []

    for model_name, variants in final.items():

        baseline = variants["E0"]

        base_map = baseline["metrics"]["mAP50_95"]
        base_time = baseline["timing"][
            "post_total_ms"
        ]["mean_ms"]

        for method, v in variants.items():

            m = v["metrics"]
            t = v["timing"]

            post_ms = t["post_total_ms"]["mean_ms"]

            rows.append({
                "model": model_name,
                "method": method,
                "topk": (
                    v["topk"]
                    if v["topk"] is not None
                    else "unlimited"
                ),
                "mAP50": m["mAP50"],
                "mAP50_95": m["mAP50_95"],
                "AP_loss_pp": 100 * (
                    base_map - m["mAP50_95"]
                ),
                "topk_ms": t["topk_ms"]["mean_ms"],
                "nms_ms": t["nms_ms"]["mean_ms"],
                "post_total_ms": post_ms,
                "post_speedup": base_time / post_ms,
                "candidates_before": (
                    v["avg_candidates_before"]
                ),
                "candidates_after": (
                    v["avg_candidates_after"]
                ),
            })

    with open(output_csv, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 72)
    print("E0 / E1 RESULTS")
    print("=" * 72)

    for r in rows:

        print(
            "{:<5} {:<5} K={:<9} "
            "mAP50={:.4f} mAP50:95={:.4f} "
            "Loss={:+.3f}pp "
            "NMS={:.3f}ms Post={:.3f}ms "
            "Speedup={:.2f}x".format(
                r["model"],
                r["method"],
                str(r["topk"]),
                r["mAP50"],
                r["mAP50_95"],
                r["AP_loss_pp"],
                r["nms_ms"],
                r["post_total_ms"],
                r["post_speedup"]
            )
        )

    print("\nSaved:", output_json)
    print("Saved:", output_csv)


if __name__ == "__main__":
    main()
