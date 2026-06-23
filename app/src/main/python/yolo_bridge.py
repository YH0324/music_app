# -*- coding: utf-8 -*-
"""
yolo_bridge.py — letterbox/decode/NMS 與 yolo_onnx.py 相同,
但「跑 onnx」改由 Kotlin(onnxruntime-android)執行。

複刻 ultralytics 內部兩段（手機端 Kotlin 也照這個算法做）：
  前處理：letterbox 到 1024（長邊縮、補灰邊 114、保持長寬比、置中）
  後處理：解析 (1, 4+nc, N) 輸出 → conf 過濾 → 每類 NMS → 座標還原回原圖

回傳格式與 final_inference.detect_with_yolo_multi 一致：
  [{"box": (x1,y1,x2,y2), "cls_id": int, "conf": float}, ...]
"""

import cv2
import numpy as np


def letterbox(img, new=1024, color=114):
    """回傳 (letterboxed_img, ratio, pad_left, pad_top)。對齊 ultralytics LetterBox。"""
    h, w = img.shape[:2]
    r = min(new / h, new / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)  # ultralytics 用 LINEAR
    dw, dh = (new - nw) / 2.0, (new - nh) / 2.0
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    out = cv2.copyMakeBorder(resized, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(color, color, color))
    return out, r, left, top


def nms_numpy(boxes, scores, iou_thr):
    """單類 NMS。boxes:(N,4) xyxy，scores:(N,)。回傳保留的索引 list。"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


def decode(pred, r, padx, pady, W, H, conf=0.30, iou=0.45, classes=None, max_det=300):
    """
    pred: onnx 原始輸出 (1, 4+nc, Nanchor)。
    回傳 [{"box","cls_id","conf"}, ...]，座標已還原到原圖 (W,H)。
    """
    p = pred[0].T                          # (Nanchor, 4+nc)
    boxes = p[:, :4]                       # cx,cy,w,h（letterbox 1024 空間）
    scores = p[:, 4:]                      # (Nanchor, nc)，已 sigmoid
    cls = scores.argmax(1)
    cf = scores.max(1)

    m = cf > conf                          # ultralytics 用嚴格大於
    if classes is not None:
        m &= np.isin(cls, classes)
    boxes, cls, cf = boxes[m], cls[m], cf[m]
    if boxes.shape[0] == 0:
        return []

    # cx,cy,w,h -> x1,y1,x2,y2
    xy = np.empty_like(boxes)
    xy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    # 每類 NMS（agnostic=False，與 ultralytics 預設一致）
    keep_all = []
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        k = nms_numpy(xy[idx], cf[idx], iou)
        keep_all.extend(idx[k].tolist())
    xy, cls, cf = xy[keep_all], cls[keep_all], cf[keep_all]

    # 全域 max_det 上限（依信心排序）
    if cf.shape[0] > max_det:
        top = cf.argsort()[::-1][:max_det]
        xy, cls, cf = xy[top], cls[top], cf[top]

    # letterbox 空間 -> 原圖
    xy[:, [0, 2]] -= padx
    xy[:, [1, 3]] -= pady
    xy /= r
    xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, W)
    xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, H)

    out = []
    for b, c, s in zip(xy, cls, cf):
        out.append({"box": (int(b[0]), int(b[1]), int(b[2]), int(b[3])),
                    "cls_id": int(c), "conf": float(s)})
    return out


class YoloBridge:
    """
    與 YoloOnnx 介面相同(detect(image, conf, iou, classes)),
    但 onnx 推論交給 Kotlin。kotlin_runner 需提供:
        run(float_list, h, w) -> (flat_output_list, shape_list)
    其中輸入 float_list 是 letterbox 後 [1,3,h,w] 的 CHW 攤平(已 /255、RGB),
    回傳 flat 為攤平的 raw 輸出、shape 例如 [1,26,21504]。
    """
    def __init__(self, kotlin_runner, imgsz=1024):
        self.runner = kotlin_runner
        self.imgsz = imgsz

    def detect(self, image, conf=0.30, iou=0.45, classes=None):
        img = cv2.imread(str(image)) if isinstance(image, str) else image
        H, W = img.shape[:2]
        lb, r, padx, pady = letterbox(img, self.imgsz)
        x = lb[:, :, ::-1].transpose(2, 0, 1)[None]      # BGR->RGB, HWC->CHW, 加 batch
        x = np.ascontiguousarray(x, dtype=np.float32) / 255.0
        h, w = x.shape[2], x.shape[3]

        # === 推論交給 Kotlin(onnxruntime-android)===
        flat, shape = _run_via_kotlin(self.runner, x.reshape(-1), h, w)
        pred = np.asarray(flat, dtype=np.float32).reshape([int(s) for s in shape])

        return decode(pred, r, padx, pady, W, H, conf=conf, iou=iou, classes=classes)


def _run_via_kotlin(runner, flat_chw, h, w):
    """
    呼叫 Kotlin runner.run(...)。Chaquopy 會把回傳的 Java 物件轉成 Python。
    Kotlin 端回傳 Pair(FloatArray, IntArray),這裡轉成 (list, list)。
    為了相容不同回傳形式,做容錯處理。
    """
    import array
    # 把 numpy float32 轉成 Python list 給 Kotlin(Chaquopy 會轉成 float[])
    data = flat_chw.astype(np.float32).tolist()
    res = runner.run(data, int(h), int(w))
    # res 可能是 Pair / list / tuple
    try:
        flat, shape = res
    except (TypeError, ValueError):
        flat, shape = res.getFirst(), res.getSecond()
    flat = list(flat)
    shape = list(shape)
    return flat, shape


# 提供與舊名相容的別名(萬一有程式 import YoloOnnx)
YoloOnnx = YoloBridge