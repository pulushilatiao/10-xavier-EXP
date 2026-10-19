
# -*- coding: utf-8 -*-
"""
Jetson Xavier YOLOv8 FP16 vs INT8 evaluation
TensorRT 8.5.x / NumPy / OpenCV / ctypes

No PyTorch, Ultralytics or PyCUDA required.

Supports YOLOv8 detection raw output:
[1, 4+nc, 8400] or [1, 8400, 4+nc]
"""

import argparse
import ctypes as C
import ctypes.util
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np

# Compatibility fix:
# TensorRT 8.5.x may reference np.bool,
# which was removed in NumPy 1.24.
if "bool" not in np.__dict__:
    np.bool = np.bool_

import tensorrt as trt


# ==========================================
# 1. Configuration
# ==========================================

ROOT = Path(__file__).resolve().parent

MODELS = {
    "FP16": ROOT / "1012_fp16.engine",
    "INT8": ROOT / "1012_int8.engine",
}

IMAGES = ROOT / "val" 

if not IMAGES.is_dir():
    IMAGES = ROOT / "val"

LABELS = ROOT / "val" / "labels"

SIZE = 640
CONF = 0.001
NMS_IOU = 0.70
MAX_DET = 300

THRS = np.arange(0.50, 0.96, 0.05)


# ==========================================
# 2. CUDA Runtime (without PyTorch)
# ==========================================

def check(err, name):
    if err != 0:
        raise RuntimeError(
            "CUDA {} failed with error {}".format(name, err)
        )


class Cuda:

    def __init__(self):

        candidates = [
            ctypes.util.find_library("cudart"),
            "libcudart.so",
            "libcudart.so.11.0"
        ]

        for name in candidates:
            if not name:
                continue

            try:
                self.api = C.CDLL(name)
                break
            except OSError:
                pass
        else:
            raise RuntimeError(
                "CUDA runtime libcudart.so not found"
            )

        A = self.api
        P = C.c_void_p
        I = C.c_int
        SZ = C.c_size_t
        F = C.c_float

        A.cudaSetDevice.argtypes = [I]
        A.cudaSetDevice.restype = I

        A.cudaMalloc.argtypes = [C.POINTER(P), SZ]
        A.cudaMalloc.restype = I

        A.cudaFree.argtypes = [P]
        A.cudaFree.restype = I

        A.cudaMemcpyAsync.argtypes = [
            P, P, SZ, I, P
        ]
        A.cudaMemcpyAsync.restype = I

        A.cudaStreamCreate.argtypes = [
            C.POINTER(P)
        ]
        A.cudaStreamCreate.restype = I

        A.cudaStreamSynchronize.argtypes = [P]
        A.cudaStreamSynchronize.restype = I

        A.cudaStreamDestroy.argtypes = [P]
        A.cudaStreamDestroy.restype = I

        A.cudaEventCreate.argtypes = [
            C.POINTER(P)
        ]
        A.cudaEventCreate.restype = I

        A.cudaEventRecord.argtypes = [P, P]
        A.cudaEventRecord.restype = I

        A.cudaEventElapsedTime.argtypes = [
            C.POINTER(F), P, P
        ]
        A.cudaEventElapsedTime.restype = I

        A.cudaEventDestroy.argtypes = [P]
        A.cudaEventDestroy.restype = I

        check(
            A.cudaSetDevice(0),
            "cudaSetDevice"
        )

    def new(self, fun, *args):

        p = C.c_void_p()

        check(
            getattr(self.api, fun)(
                C.byref(p), *args
            ),
            fun
        )

        return p


# ==========================================
# 3. Load TensorRT Engine
# ==========================================

class Engine:

    def __init__(self, path, cuda):

        self.cuda = cuda

        logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(logger)

        self.engine = (
            self.runtime.deserialize_cuda_engine(
                path.read_bytes()
            )
        )

        if self.engine is None:
            raise RuntimeError(
                "Could not load Engine: {}".format(path)
            )

        self.context = (
            self.engine.create_execution_context()
        )

        if self.context is None:
            raise RuntimeError(
                "Could not create execution context"
            )

        if self.engine.has_implicit_batch_dimension:
            raise RuntimeError(
                "Requires explicit-batch Engine"
            )

        if self.engine.num_bindings != 2:
            raise RuntimeError(
                "Found {} bindings. Only YOLOv8 "
                "detection with one input and one "
                "raw output is supported. "
                "Segmentation needs another evaluator."
                .format(self.engine.num_bindings)
            )

        self.in_id = next(
            (
                i for i in range(2)
                if self.engine.binding_is_input(i)
            ),
            None
        )

        if self.in_id is None:
            raise RuntimeError("No input binding")

        self.out_id = 1 - self.in_id

        shape = tuple(
            self.engine.get_binding_shape(self.in_id)
        )

        if any(v == -1 for v in shape):

            ok = self.context.set_binding_shape(
                self.in_id,
                (1, 3, SIZE, SIZE)
            )

            if not ok:
                raise RuntimeError(
                    "Could not set input shape"
                )

        self.in_shape = tuple(
            self.context.get_binding_shape(self.in_id)
        )

        self.out_shape = tuple(
            self.context.get_binding_shape(self.out_id)
        )

        print("INPUT:", self.in_shape)
        print("OUTPUT:", self.out_shape)

        if self.in_shape != (1, 3, SIZE, SIZE):
            raise RuntimeError(
                "Expected input [1,3,640,640], got {}"
                .format(self.in_shape)
            )

        if (
            len(self.out_shape) != 3
            or self.out_shape[0] != 1
            or 8400 not in self.out_shape
        ):
            raise RuntimeError(
                "Unexpected output shape {}. "
                "Check model task and NMS."
                .format(self.out_shape)
            )

        if self.out_shape[1] == 8400:
            self.nc = self.out_shape[2] - 4
        else:
            self.nc = self.out_shape[1] - 4

        if not 1 <= self.nc <= 1000:
            raise RuntimeError(
                "Invalid class count {}".format(self.nc)
            )

        self.in_type = np.dtype(
            trt.nptype(
                self.engine.get_binding_dtype(
                    self.in_id
                )
            )
        )

        self.out_type = np.dtype(
            trt.nptype(
                self.engine.get_binding_dtype(
                    self.out_id
                )
            )
        )

        supported = (
            np.dtype("float32"),
            np.dtype("float16")
        )

        if (
            self.in_type not in supported
            or self.out_type not in supported
        ):
            raise RuntimeError(
                "This script expects FP32/FP16 I/O. "
                "INT8 internal compute is supported."
            )

        self.buf = [None, None]

        for i in range(2):

            shp = tuple(
                self.context.get_binding_shape(i)
            )

            dt = np.dtype(
                trt.nptype(
                    self.engine.get_binding_dtype(i)
                )
            )

            if any(v <= 0 for v in shp):
                raise RuntimeError(
                    "Invalid binding shape {}".format(shp)
                )

            nbytes = (
                int(np.prod(shp)) * dt.itemsize
            )

            self.buf[i] = cuda.new(
                "cudaMalloc", nbytes
            )

        self.stream = cuda.new(
            "cudaStreamCreate"
        )

        self.ev0 = cuda.new(
            "cudaEventCreate"
        )

        self.ev1 = cuda.new(
            "cudaEventCreate"
        )

        print("CLASSES:", self.nc)
        print("INPUT DTYPE:", self.in_type)
        print("OUTPUT DTYPE:", self.out_type)

    def infer(self, tensor):

        x = np.ascontiguousarray(
            tensor,
            dtype=self.in_type
        )

        if x.shape != self.in_shape:
            raise RuntimeError(
                "Input shape mismatch"
            )

        y = np.empty(
            self.out_shape,
            dtype=self.out_type
        )

        A = self.cuda.api
        st = self.stream

        start = time.perf_counter()

        check(
            A.cudaMemcpyAsync(
                self.buf[self.in_id],
                C.c_void_p(x.ctypes.data),
                x.nbytes,
                1,
                st
            ),
            "H2D"
        )

        check(
            A.cudaEventRecord(self.ev0, st),
            "event start"
        )

        ok = self.context.execute_async_v2(
            [int(p.value) for p in self.buf],
            int(st.value)
        )

        if not ok:
            raise RuntimeError(
                "TensorRT execution failed"
            )

        check(
            A.cudaEventRecord(self.ev1, st),
            "event end"
        )

        check(
            A.cudaMemcpyAsync(
                C.c_void_p(y.ctypes.data),
                self.buf[self.out_id],
                y.nbytes,
                2,
                st
            ),
            "D2H"
        )

        check(
            A.cudaStreamSynchronize(st),
            "stream sync"
        )

        ms = C.c_float()

        check(
            A.cudaEventElapsedTime(
                C.byref(ms),
                self.ev0,
                self.ev1
            ),
            "elapsed"
        )

        total_ms = (
            time.perf_counter() - start
        ) * 1000

        return y, float(ms.value), total_ms

    def close(self):

        A = self.cuda.api

        A.cudaStreamSynchronize(self.stream)

        A.cudaEventDestroy(self.ev0)
        A.cudaEventDestroy(self.ev1)

        A.cudaStreamDestroy(self.stream)

        for ptr in self.buf:
            if ptr:
                A.cudaFree(ptr)


# ==========================================
# 4. YOLOv8 image preprocessing
# ==========================================

def preprocess(im):

    h, w = im.shape[:2]

    r = min(
        SIZE / w,
        SIZE / h
    )

    nw = round(w * r)
    nh = round(h * r)

    resized = cv2.resize(
        im,
        (nw, nh),
        interpolation=cv2.INTER_LINEAR
    )

    left = round((SIZE - nw) / 2 - 0.1)
    top = round((SIZE - nh) / 2 - 0.1)

    right = SIZE - nw - left
    bottom = SIZE - nh - top

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114)
    )

    rgb = cv2.cvtColor(
        padded,
        cv2.COLOR_BGR2RGB
    )

    tensor = np.ascontiguousarray(
        rgb.transpose(2, 0, 1)[None] / 255.0,
        dtype=np.float32
    )

    return tensor, r, left, top


# ==========================================
# 5. YOLOv8 detection postprocessing
# ==========================================

def postprocess(raw, wh, ratio, left, top, nc):

    a = raw[0].astype(np.float32)

    if a.shape[1] == 8400:
        a = a.T

    if (
        a.shape != (8400, 4 + nc)
        or not np.all(np.isfinite(a))
    ):
        raise RuntimeError(
            "Incorrect YOLO output shape "
            "or NaN/Inf: {}".format(a.shape)
        )

    score = a[:, 4:].max(axis=1)
    cls = a[:, 4:].argmax(axis=1)

    valid = score > CONF

    a = a[valid]
    score = score[valid]
    cls = cls[valid]

    if not len(a):
        return np.empty(
            (0, 6),
            dtype=np.float32
        )

    b = np.empty(
        (len(a), 4),
        dtype=np.float32
    )

    b[:, :2] = (
        a[:, :2] - a[:, 2:4] / 2
    )

    b[:, 2:] = (
        a[:, :2] + a[:, 2:4] / 2
    )

    picks = []

    # Class-aware NMS
    for c in np.unique(cls):

        ind = np.flatnonzero(cls == c)

        xywh = np.column_stack(
            (
                b[ind, :2],
                np.maximum(
                    0,
                    b[ind, 2:] - b[ind, :2]
                )
            )
        )

        keep = cv2.dnn.NMSBoxes(
            xywh.tolist(),
            score[ind].tolist(),
            CONF,
            NMS_IOU
        )

        if keep is not None and len(keep):

            picks.extend(
                ind[
                    np.array(keep).reshape(-1)
                ].tolist()
            )

    if not picks:
        return np.empty(
            (0, 6),
            dtype=np.float32
        )

    picks = np.array(
        picks,
        dtype=int
    )

    picks = picks[
        np.argsort(-score[picks])[:MAX_DET]
    ]

    b = b[picks].copy()

    b[:, [0, 2]] = np.clip(
        (b[:, [0, 2]] - left) / ratio,
        0,
        wh[0]
    )

    b[:, [1, 3]] = np.clip(
        (b[:, [1, 3]] - top) / ratio,
        0,
        wh[1]
    )

    return np.column_stack(
        (
            b,
            score[picks],
            cls[picks]
        )
    ).astype(np.float32)


# ==========================================
# 6. Load YOLO detection labels
# ==========================================

def load_gt(txt, width, height):

    result = []

    if not txt.is_file():

        print(
            "[WARN] Missing label, "
            "treated as negative image:",
            txt
        )

        return np.empty(
            (0, 5),
            np.float32
        )

    lines = txt.read_text(
        encoding="utf-8-sig"
    ).splitlines()

    for lineno, line in enumerate(lines, 1):

        if not line.strip():
            continue

        v = line.split()

        if len(v) != 5:
            raise ValueError(
                "{} line {} has {} fields. "
                "Expected detection annotation: "
                "class cx cy w h. "
                "Segmentation is not supported."
                .format(txt, lineno, len(v))
            )

        c = int(v[0])

        cx, cy, w, h = map(
            float,
            v[1:]
        )

        if (
            c < 0
            or not np.isfinite(
                [cx, cy, w, h]
            ).all()
            or not (
                0 <= cx <= 1
                and 0 <= cy <= 1
                and 0 < w <= 1
                and 0 < h <= 1
            )
        ):
            raise ValueError(
                "Invalid label {} line {}"
                .format(txt, lineno)
            )

        result.append(
            [
                max(0, (cx-w/2)*width),
                max(0, (cy-h/2)*height),
                min(width, (cx+w/2)*width),
                min(height, (cy+h/2)*height),
                c
            ]
        )

    return np.asarray(
        result,
        dtype=np.float32
    ).reshape(-1, 5)


# ==========================================
# 7. IoU and AP computation
# ==========================================

def box_iou(b, gt):

    xy1 = np.maximum(
        b[:2],
        gt[:, :2]
    )

    xy2 = np.minimum(
        b[2:4],
        gt[:, 2:4]
    )

    inter = np.maximum(
        xy2 - xy1,
        0
    ).prod(axis=1)

    ba = (
        max(0, b[2]-b[0])
        * max(0, b[3]-b[1])
    )

    ga = np.maximum(
        gt[:, 2:4] - gt[:, :2],
        0
    ).prod(axis=1)

    return inter / (
        ba + ga - inter + 1e-9
    )


def map_scores(preds, gts):

    classes = sorted(
        {
            int(g[4])
            for gt in gts
            for g in gt
        }
    )

    if not classes:
        raise ValueError(
            "No ground-truth objects"
        )

    ap = []

    for c in classes:

        per_img_gt = {
            i: g[g[:, 4] == c, :4]
            for i, g in enumerate(gts)
        }

        n_gt = sum(
            len(g)
            for g in per_img_gt.values()
        )

        detections = sorted(
            (
                (
                    float(p[4]),
                    i,
                    p[:4]
                )
                for i, ps in enumerate(preds)
                for p in ps
                if int(p[5]) == c
            ),
            key=lambda v: -v[0]
        )

        matched = {
            i: np.zeros(
                (len(THRS), len(g)),
                dtype=bool
            )
            for i, g in per_img_gt.items()
        }

        tp = np.zeros(
            (len(detections), len(THRS)),
            dtype=np.float32
        )

        for j, (_, i, box) in enumerate(detections):

            if not len(per_img_gt[i]):
                continue

            overlaps = box_iou(
                box,
                per_img_gt[i]
            )

            for k, thr in enumerate(THRS):

                choices = np.flatnonzero(
                    (overlaps >= thr)
                    & ~matched[i][k]
                )

                if len(choices):

                    gi = choices[
                        np.argmax(overlaps[choices])
                    ]

                    matched[i][k, gi] = True
                    tp[j, k] = 1

        row = []

        for k in range(len(THRS)):

            if not len(detections):
                row.append(0.0)
                continue

            rec = (
                tp[:, k].cumsum() / n_gt
            )

            prec = (
                tp[:, k].cumsum()
                / np.arange(
                    1,
                    len(detections) + 1
                )
            )

            levels = np.linspace(
                0, 1, 101
            )

            values = [
                prec[rec >= r].max()
                if np.any(rec >= r)
                else 0.0
                for r in levels
            ]

            row.append(
                float(np.mean(values))
            )

        ap.append(row)

    ap = np.array(ap)

    return {
        "mAP50": float(
            ap[:, 0].mean()
        ),
        "mAP50_95": float(
            ap.mean()
        ),
        "evaluated_classes": classes
    }


# ==========================================
# 8. Read class count from data.yaml
# ==========================================

def read_nc():

    yaml = ROOT / "data.yaml"

    if not yaml.exists():
        return None

    s = yaml.read_text(
        encoding="utf-8-sig"
    )

    m = re.search(
        r"(?m)^\s*nc\s*:\s*(\d+)",
        s
    )

    if m:
        return int(m.group(1))

    m = re.search(
        r"(?m)^names\s*:\s*\[([^\]]+)\]",
        s
    )

    if m:
        return len(
            m.group(1).split(",")
        )

    return None


# ==========================================
# 9. Evaluate one Engine
# ==========================================

def evaluate(
    name,
    path,
    samples,
    gt,
    cuda,
    expected
):

    print("\n==============================")
    print("Evaluating:", name)
    print("==============================")

    model = Engine(path, cuda)

    try:

        if (
            expected is not None
            and model.nc != expected
        ):
            raise RuntimeError(
                "Engine has {} classes, "
                "but data.yaml says {}. "
                "Check model weights!"
                .format(model.nc, expected)
            )

        for g in gt:
            if np.any(g[:, 4] >= model.nc):
                raise RuntimeError(
                    "Label class ID exceeds "
                    "Engine class count"
                )

        first = cv2.imread(
            str(samples[0])
        )

        if first is None:
            raise RuntimeError(
                "Could not read first image"
            )

        warm, _, _, _ = preprocess(first)

        print("Warmup: 5 iterations")

        for _ in range(5):
            model.infer(warm)

        preds = []
        gpu_times = []
        e2e_times = []

        for i, file in enumerate(samples, 1):

            start = time.perf_counter()

            img = cv2.imread(str(file))

            if img is None:
                raise RuntimeError(
                    "Could not read: {}".format(file)
                )

            h, w = img.shape[:2]

            inp, ratio, left, top = preprocess(
                img
            )

            output, gpu_ms, _ = model.infer(inp)

            p = postprocess(
                output,
                (w, h),
                ratio,
                left,
                top,
                model.nc
            )

            preds.append(p)
            gpu_times.append(gpu_ms)

            elapsed = (
                time.perf_counter() - start
            ) * 1000

            e2e_times.append(elapsed)

            print(
                "{}/{}: {} => {} objects"
                .format(
                    i,
                    len(samples),
                    file.name,
                    len(p)
                )
            )

        result = map_scores(
            preds,
            gt
        )

        result.update(
            images=len(samples),
            classes_in_engine=model.nc,
            gpu_ms_mean=float(
                np.mean(gpu_times)
            ),
            gpu_ms_p95=float(
                np.percentile(
                    gpu_times,
                    95
                )
            ),
            gpu_fps_equivalent=float(
                1000 / np.mean(gpu_times)
            ),
            e2e_ms_mean=float(
                np.mean(e2e_times)
            ),
            e2e_fps=float(
                1000 / np.mean(e2e_times)
            )
        )

        return result

    finally:
        model.close()


# ==========================================
# 10. Main: FP16 vs INT8
# ==========================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help=(
            "5 for smoke test; "
            "0 for all images"
        )
    )

    args = parser.parse_args()

    if args.max_images < 0:
        parser.error(
            "--max-images must be nonnegative"
        )

    print("TensorRT:", trt.__version__)
    print("OpenCV:", cv2.__version__)

    for path in MODELS.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    if (
        not IMAGES.is_dir()
        or not LABELS.is_dir()
    ):
        raise RuntimeError(
            "Missing validation images "
            "or labels directory: {} {}"
            .format(IMAGES, LABELS)
        )

    extensions = {
        ".jpg", ".jpeg", ".png",
        ".bmp", ".webp"
    }

    pics = sorted(
        p for p in IMAGES.rglob("*")
        if (
            p.suffix.lower() in extensions
            and LABELS not in p.parents
        )
    )

    if args.max_images:

        pics = pics[:args.max_images]

        print(
            "[DEBUG] Evaluating only {} images. "
            "Do not publish subset AP."
            .format(len(pics))
        )

    if not pics:
        raise RuntimeError(
            "No validation images found"
        )

    gt = []

    for p in pics:

        img = cv2.imread(str(p))

        if img is None:
            raise RuntimeError(
                "Cannot decode: {}".format(p)
            )

        label = (
            LABELS
            / p.relative_to(IMAGES).with_suffix(".txt")
        )

        gt.append(
            load_gt(
                label,
                img.shape[1],
                img.shape[0]
            )
        )

    total_objects = sum(
        len(g) for g in gt
    )

    print("Validation images:", len(pics))
    print("GT objects:", total_objects)

    if total_objects == 0:
        raise RuntimeError(
            "No GT objects. Check labels."
        )

    expected = read_nc()

    print(
        "Expected nc from data.yaml:",
        expected
    )

    cuda = Cuda()

    results = {}

    for name, path in MODELS.items():

        results[name] = evaluate(
            name,
            path,
            pics,
            gt,
            cuda,
            expected
        )

    if (
        results["FP16"]["classes_in_engine"]
        != results["INT8"]["classes_in_engine"]
    ):
        raise RuntimeError(
            "Class counts differ between Engines"
        )

    print("\n================================")
    print("FP16 vs INT8 RESULTS")
    print("================================")

    print(
        "{:<19} {:>10} {:>10}"
        .format("METRIC", "FP16", "INT8")
    )

    keys = [
        "mAP50",
        "mAP50_95",
        "gpu_ms_mean",
        "gpu_ms_p95",
        "e2e_ms_mean",
        "e2e_fps"
    ]

    for k in keys:

        print(
            "{:<19} {:>10.4f} {:>10.4f}"
            .format(
                k,
                results["FP16"][k],
                results["INT8"][k]
            )
        )

    for k in (
        "mAP50",
        "mAP50_95"
    ):

        a = results["FP16"][k]
        b = results["INT8"][k]

        loss_pp = (a - b) * 100

        print(
            "{} INT8 loss: {:.3f} "
            "percentage points"
            .format(k, loss_pp)
        )

    output = ROOT / (
        "precision_DEBUG.json"
        if args.max_images
        else "precision_comparison.json"
    )

    output.write_text(
        json.dumps(
            results,
            indent=2
        ),
        encoding="utf-8"
    )

    print("Saved:", output)

    print(
        "NOTE: AP uses independent "
        "101-point interpolation."
    )

    print(
        "NOTE: GPU and E2E FPS "
        "are not trtexec throughput."
    )


if __name__ == "__main__":

    print("\n========== ENGINE CHECK ==========")

    for name, path in MODELS.items():

        if not path.is_file():
            raise FileNotFoundError(
                f"{name} Engine not found: {path}"
        )

        print(
        f"{name}: {path.name}, "
        f"{path.stat().st_size / 1024**2:.2f} MB"
    )

    print("==================================\n")

    main()
