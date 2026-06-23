"""
final_inference.py
===================
新版 OMR pipeline，採用論文 Stave-Aware OMR 的「多階段偵測」架構。

流程：
1. 讀圖 + 解析度標準化（譜線間距 resize 到 ~20 px）
2. 21 類 YOLOv8 偵測：notehead / rest / flag / dot / accidental / key / clef / tie / slur
3. PitchCNN 對 noteheads 判音高
4. OpenCV 偵測 stem / beam（21 類 YOLO 不訓這兩個）
5. 把 notehead + stem + flag/beam + dot 組成完整音符（含時值）
6. 把 rest 直接放進 events
7. 排序 + 輸出 MIDI / debug.png / events.json

替代了之前的 OpenCV-based rest/flag/dot 候選邏輯。
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import re
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

# 加 python 目錄到路徑
sys.path.insert(0, str(Path(__file__).parent))


# YOLO 由外部 runner 執行(PC: yolo_onnx.YoloOnnx;Android: Kotlin onnxruntime)
_YOLO_RUNNER = None
# 模型資料夾(含 dot_cnn.npz/.json);由 recognize() 設定
MODELS_DIR = None

def set_yolo_runner(runner):
    global _YOLO_RUNNER
    _YOLO_RUNNER = runner

from merge_notes_np import (
    detect_staff_lines,
    collect_noteheads_for_staff, collect_symbols_for_staff,
    infer_step_and_pitch_from_lines,
    find_stem_for_note, count_beams_for_stem, flag_level_for_stem,
    infer_duration, merge_notehead_type,
    bbox_center, bbox_width, bbox_height,
)


# ========================================================
# 路徑與參數
# ========================================================
IMAGE_PATH = None   # 由 recognize() 設定

YOLO_MULTI_PATH = Path(r"C:\music\outputs\notehead_detector_v2\weights\best_BACKUP_22class_v1.pt")
# 舊單類 YOLO（如果新的不在則 fallback）
YOLO_SINGLE_PATH = Path(r"C:\music\outputs\notehead_detector\weights\best.pt")

# SymbolCNN：用來做附點二次驗證（YOLO 對 augmentationDot 還是相對弱）
SYMBOL_MODEL_PATH = Path(r"C:\music\outputs\symbol_classifier_v2\best_model.pt")
SYMBOL_CLASSMAP = Path(r"C:\music\outputs\symbol_classifier_v2\class_map.json")
SYMBOL_IMAGE_SIZE = 64

OUT_DIR = Path(r"C:\music\outputs\final")
# 模組級 mkdir 已移除(手機唯讀);實際輸出由 recognize() 用可寫路徑重設並建立

# === [診斷] 只記錄不改判斷：輸出每條 staff 的原始幾何，供離線重放 beam/flag 決策 ===
# 設 False 即關閉，對結果無任何影響。
ENABLE_GEOM_DUMP = True
_GEOM_DUMP = []

# === [診斷] 附點 CNN 二次驗證（dot_classifier）===
# 純診斷：對每個 note 裁「音符+右側」貼片餵 dot_classifier，記錄判斷，
# 不改任何 events。用來在真實譜上驗證這個分類器準不準，確認後才考慮接成 gate。
DOT_CNN_MODEL_PATH = Path(r"C:\music\outputs\dot_classifier\best_model.pt")
DOT_CNN_CLASSMAP = Path(r"C:\music\outputs\dot_classifier\class_map.json")
DOT_CNN_RIGHT_EXT = 1.4   # 與 build_dot_dataset.py 裁窗一致
DOT_CNN_VERT = 1.1
DOT_CNN_LEFT_PAD = 0.2

# === [診斷] 音高：幾何規則 vs PitchCNN 並排 ===
# 純診斷：對每個 note 同時記錄幾何判斷與 PitchCNN 判斷，
# 用來看「低解析/邊界音改信 PitchCNN」是否真的較準，確認後才改判定邏輯。

# === 低解析超解析前處理（Real-ESRGAN）===
# 原始譜線間距 < 門檻才超解析（正常譜不動，避免加假紋理）。
# 對任何低解析譜通用，不是針對特定圖。
ENABLE_SUPERRES = True
SUPERRES_SPACING_THRESHOLD = 10.0   # 原圖譜線間距 < 10px 才啟用
SUPERRES_SCALE = 4
_SR_UPSAMPLER = None                 # 延遲載入（只在需要時初始化一次）

# === beam/flag 時值修正參數 ===
# 條件C（beam 垂直切段合併）已停用：1/6/7.png 實測證實假切段與真 16th 在
#   YOLO box 層級幾何重疊，無法通用區分。此常數保留僅供未來重訓後再評估。
BEAM_SPLIT_MERGE_MIN_XOVL = 0.65   # (停用)
# flag16th 可靠度低：信心低於此值時降級為 flag8th(8th)，避免把 flagged 8th 誤判 16th。
# 此修法零回歸（7 張無真 flag16th 落在門下），保留啟用。
FLAG16_MIN_CONF = 0.5

# 解析度標準化目標
TARGET_STAFF_SPACING = 20.0

# YOLO 信心閾值
YOLO_CONF = 0.30
YOLO_IOU = 0.45

# 拍號（手動指定，Y 方案）
TIME_SIGNATURE_NUM = 4   # 分子
TIME_SIGNATURE_DEN = 4   # 分母

# 21 類定義（跟 train_yolo_multiclass 的 yaml 對齊）
YOLO_CLASS_ID_TO_NAME = {
    0:  "noteheadBlack",
    1:  "noteheadHalf",
    2:  "noteheadWhole",
    3:  "rest8th",
    4:  "rest16th",
    5:  "restQuarter",
    6:  "restHalf",
    7:  "restWhole",
    8:  "flag8thUp",
    9:  "flag8thDown",
    10: "flag16thUp",
    11: "flag16thDown",
    12: "augmentationDot",
    13: "clefG",
    14: "accidentalSharp",
    15: "accidentalFlat",
    16: "accidentalNatural",
    17: "keySharp",
    18: "keyFlat",
    19: "tie",
    20: "slur",
    21: "beam",
}

NOTEHEAD_CLASSES = {"noteheadBlack", "noteheadHalf", "noteheadWhole"}
REST_CLASSES = {"restWhole", "restHalf", "restQuarter", "rest8th", "rest16th"}
FLAG_CLASSES = {"flag8thUp", "flag8thDown", "flag16thUp", "flag16thDown"}


# ========================================================
# 輔助：解析度標準化
# ========================================================
def auto_detect_staff_boxes(gray):
    """回傳每行五線譜的緊貼外框"""
    h, w = gray.shape
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h_proj = np.sum(bw, axis=1) / 255.0
    idx = np.where(h_proj > w * 0.15)[0]
    if len(idx) == 0:
        return []

    staves = []
    start = idx[0]
    for i in range(1, len(idx)):
        if idx[i] - idx[i-1] > 40:
            staves.append((start, idx[i-1]))
            start = idx[i]
    staves.append((start, idx[-1]))

    boxes = []
    for s, e in staves:
        y1 = int(s)
        y2 = int(e)
        boxes.append((0, y1, w, y2))
    return boxes


def estimate_staff_spacing(gray):
    """估計譜線間距。回傳中位數，找不到回 None"""
    boxes = auto_detect_staff_boxes(gray)
    spacings = []
    for sb in boxes:
        ly = detect_staff_lines(gray, sb)
        if ly is not None and len(ly) == 5:
            diffs = np.diff(sorted(ly))
            if len(diffs) > 0:
                md = float(np.median(diffs))
                if md > 0:
                    spacings.append(md)
    if not spacings:
        return None
    return float(np.median(spacings))


def maybe_superres(gray, bgr, log):
    """M2:超解析直通(ESRGAN 在 M3 用 ncnn-Vulkan 於 Kotlin 端處理)。"""
    return gray, bgr


def standardize_resolution(gray, bgr, log):
    """
    把整張圖 resize 到譜線間距 ~ TARGET_STAFF_SPACING，回傳 (gray', bgr', scale)
    """
    spacing = estimate_staff_spacing(gray)
    if spacing is None:
        log("[WARN] 偵測不到譜線間距，跳過解析度標準化")
        return gray, bgr, 1.0
    scale = TARGET_STAFF_SPACING / spacing
    log(f"[INFO] 原始譜線間距: {spacing:.2f} px, "
        f"目標: {TARGET_STAFF_SPACING:.0f} px, 縮放倍率: {scale:.2f}x")
    if 0.77 < scale < 1.3:
        log(f"[INFO] 縮放倍率接近 1，跳過縮放")
        return gray, bgr, 1.0

    new_w = int(round(gray.shape[1] * scale))
    new_h = int(round(gray.shape[0] * scale))
    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    action = "放大" if scale > 1.0 else "縮小"
    log(f"[INFO] {action}圖片: {gray.shape[1]}×{gray.shape[0]} → {new_w}×{new_h}")
    gray2 = cv2.resize(gray, (new_w, new_h), interpolation=interp)
    bgr2 = cv2.resize(bgr, (new_w, new_h), interpolation=interp) if bgr is not None else None
    return gray2, bgr2, scale


# ========================================================
# YOLO 多類偵測
# ========================================================
def detect_with_yolo_multi(yolo_model, image_path):
    """21 類偵測,經由外部 runner。複刻原雙趟(主偵測 + dot 低信心補漏)。"""
    runner = _YOLO_RUNNER
    dets = []
    for d in runner.detect(str(image_path), conf=YOLO_CONF, iou=YOLO_IOU):
        dets.append({"box": d["box"],
                     "class_name": YOLO_CLASS_ID_TO_NAME.get(d["cls_id"], f"class_{d['cls_id']}"),
                     "conf": d["conf"]})
    DOT = 12  # augmentationDot
    existing = [((b["box"][0] + b["box"][2]) / 2, (b["box"][1] + b["box"][3]) / 2)
                for b in dets if b["class_name"] == "augmentationDot"]
    for d in runner.detect(str(image_path), conf=0.15, iou=YOLO_IOU, classes=[DOT]):
        x1, y1, x2, y2 = d["box"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if any(abs(cx - ex) < 10 and abs(cy - ey) < 10 for ex, ey in existing):
            continue
        dets.append({"box": d["box"], "class_name": "augmentationDot", "conf": d["conf"]})
        existing.append((cx, cy))
    return dets


def detect_with_yolo_single(yolo_model, image_path):
    """單類 fallback:全部當 noteheadBlack。"""
    dets = []
    for d in _YOLO_RUNNER.detect(str(image_path), conf=YOLO_CONF, iou=YOLO_IOU):
        dets.append({"box": d["box"], "class_name": "noteheadBlack", "conf": d["conf"]})
    return dets


# ========================================================
# Stem / Beam（OpenCV，21 類 YOLO 不偵測這兩個）
# ========================================================
def detect_stems_in_staff(gray, staff_box, staff_lines_y):
    """偵測這條 staff 範圍內的 stem（用舊版可工作的邏輯）"""
    sx1, sy1, sx2, sy2 = staff_box
    sorted_lines = sorted(staff_lines_y)
    if len(sorted_lines) < 2:
        return []
    d = float(np.median(np.diff(sorted_lines)))
    if d <= 0:
        return []

    top = max(0, int(sy1 - 2.5 * 4 * d))
    bot = min(gray.shape[0], int(sy2 + 2.5 * 4 * d))
    roi = gray[top:bot, sx1:sx2]
    _, bw = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 不擦線（讓 stem 的整條 component 完整）
    vk_h = max(5, int(d * 2.5))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk_h))
    bw_v = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)

    n, _, stats, _ = cv2.connectedComponentsWithStats(bw_v)
    stems = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if h < d * 2.5:
            continue
        if w > d * 0.5:
            continue
        gx1 = int(x + sx1)
        gy1 = int(y + top)
        gx2 = gx1 + int(w)
        gy2 = gy1 + int(h)
        stems.append({"box": (gx1, gy1, gx2, gy2)})
    return stems


def detect_beams_in_staff(gray, staff_box, staff_lines_y, debug_log=None):
    """偵測這條 staff 範圍內的 beam（用舊版可工作的邏輯）"""
    sx1, sy1, sx2, sy2 = staff_box
    sorted_lines = sorted(staff_lines_y)
    if len(sorted_lines) < 2:
        return []
    d = float(np.median(np.diff(sorted_lines)))
    if d <= 0:
        return []

    # ROI 限制在 staff 範圍上下各 4d 內，避免吃到下一個 staff
    top = max(0, int(sy1 - 4 * d))
    bot = min(gray.shape[0], int(sy2 + 4 * d))
    roi = gray[top:bot, sx1:sx2]
    _, bw = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 先擦譜線（避免譜線跟 beam 連在一起）
    line_thick = max(2, int(d * 0.15))
    for ly in sorted_lines:
        ly_local = int(ly - top)
        if 0 <= ly_local < bw.shape[0]:
            bw[max(0, ly_local-line_thick):min(bw.shape[0], ly_local+line_thick+1), :] = 0

    # horizontal open
    hk_w = max(5, int(d * 1.5))   # 寬一點的 kernel 抓 beam
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hk_w, 1))
    bw_h = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)

    n, _, stats, _ = cv2.connectedComponentsWithStats(bw_h)
    beams = []
    n_filtered_w = 0
    n_filtered_h = 0
    n_filtered_line = 0
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if w < d * 1.5:   # 寬度下限稍降
            n_filtered_w += 1
            continue
        if h < d * 0.2 or h > d * 1.2:   # 高度範圍放寬
            n_filtered_h += 1
            continue
        # 排除「就在五線譜本身的線上」的偵測
        global_cy = y + top + h / 2
        # 只擋掉跟 staff line 完全重合的（線厚度 0.15d）
        if min(abs(global_cy - ly) for ly in staff_lines_y) < d * 0.2:
            n_filtered_line += 1
            continue
        gx1 = int(x + sx1)
        gy1 = int(y + top)
        gx2 = gx1 + int(w)
        gy2 = gy1 + int(h)
        beams.append({"box": (gx1, gy1, gx2, gy2)})

    if debug_log is not None:
        debug_log(f"        [BEAM] components={n-1}, filtered "
                  f"w={n_filtered_w} h={n_filtered_h} line={n_filtered_line}, "
                  f"kept={len(beams)}")
    return beams


def detect_barlines_in_staff(gray, staff_box, staff_lines_y):
    """
    偵測這條 staff 的小節線（barline）。
    嚴格條件：barline 必須完整跨越第 1 ~ 第 5 條譜線
    （stem 通常只跨 1~3 條譜線，所以可以區分）
    """
    sx1, sy1, sx2, sy2 = staff_box
    sorted_lines = sorted(staff_lines_y)
    if len(sorted_lines) < 5:
        return []
    d = float(np.median(np.diff(sorted_lines)))
    if d <= 0:
        return []

    staff_top = sorted_lines[0]      # 第 1 條譜線
    staff_bot = sorted_lines[-1]     # 第 5 條譜線
    staff_height = staff_bot - staff_top

    # ROI：剛好涵蓋 staff 五線區域，上下不要拉太多
    roi_top = max(0, int(staff_top - d * 0.3))
    roi_bot = min(gray.shape[0], int(staff_bot + d * 0.3))
    roi = gray[roi_top:roi_bot, sx1:sx2]
    _, bw = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 嚴格垂直 kernel：高度 = staff_height × 0.95
    # 只有「跨越整個 staff」的物件才會被保留
    vk_h = max(int(staff_height * 0.95), 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk_h))
    vertical = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)

    n, _, stats, _ = cv2.connectedComponentsWithStats(vertical)
    barlines = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        # 條件 1：高度必須 ≥ staff 全高 95%（嚴格）
        if h < staff_height * 0.95:
            continue
        # 條件 2：寬度很細（barline 約 2~5 px、stem 也類似但 stem 不會這麼長）
        if w > d * 0.4:
            continue
        # 條件 3：必須真的從第 1 條譜線附近開始、到第 5 條譜線附近結束
        global_y1 = y + roi_top
        global_y2 = y + h + roi_top
        if global_y1 > staff_top + d * 0.5:
            continue  # 起點太低（不是從第 1 條開始）
        if global_y2 < staff_bot - d * 0.5:
            continue  # 終點太高（沒到第 5 條）
        gx1 = int(x + sx1)
        cx = gx1 + w / 2.0
        # 排除 staff 最左/最右
        if cx < sx1 + d * 1.0 or cx > sx2 - d * 0.5:
            continue
        barlines.append({"box": (gx1, int(global_y1), gx1 + int(w), int(global_y2)),
                         "x": cx})

    # 去重（合併鄰近 < 1d 的）
    barlines.sort(key=lambda b: b["x"])
    deduped = []
    for bl in barlines:
        if deduped and abs(bl["x"] - deduped[-1]["x"]) < d * 1.5:
            continue
        deduped.append(bl)
    return deduped


# ========================================================
# 對應關係：給每個 notehead 找 stem / 對應 beam 數
# ========================================================
def find_stem_for_notehead(note_box, stems, d):
    """找最接近的 stem（左/右各 0.6d 內、垂直要重疊）"""
    nx1, ny1, nx2, ny2 = note_box
    ncx, ncy = (nx1 + nx2) / 2, (ny1 + ny2) / 2
    best = None
    best_dist = float("inf")
    for st in stems:
        sx1, sy1, sx2, sy2 = st["box"]
        scx = (sx1 + sx2) / 2
        # x 距離
        dx = abs(scx - ncx)
        if dx > d * 1.0:
            continue
        # y 重疊
        if sy2 < ny1 - d * 0.3 or sy1 > ny2 + d * 0.3:
            continue
        if dx < best_dist:
            best_dist = dx
            best = st
    return best


# ========================================================
# SymbolCNN（附點二次驗證用）
# ========================================================
# SymbolCNN 已移除(SymbolCNN 模型不存在,流程用不到)


def count_beams_at_stem_tip(gray, stem_box, staff_lines_y, d):
    """
    直接從圖片看 stem 末端水平方向有幾條粗黑色橫條 = beam 數。
    這比 OpenCV connected components 可靠很多。

    邏輯：
    1. 找 stem 的「末端」（朝上 stem 取頂端、朝下取底端）
    2. 在末端附近、stem 旁邊 (右側為主) 取一個小 ROI
    3. 在 ROI 內掃描每一行，看哪些行是「黑色密度高」的（粗橫條 = beam）
    4. 連續的高密度行算一條 beam，計算 beam 條數
    """
    sx1, sy1, sx2, sy2 = stem_box
    sorted_lines = sorted(staff_lines_y)
    if not sorted_lines:
        return 0
    staff_top = sorted_lines[0]
    staff_bot = sorted_lines[-1]
    staff_mid = (staff_top + staff_bot) / 2.0

    stem_h = sy2 - sy1
    if stem_h < 10:
        return 0

    # 判斷 stem 朝上還朝下
    # stem 朝上：底端在 staff 中下、頂端伸到 staff 上方
    # stem 朝下：頂端在 staff 中上、底端伸到 staff 下方
    stem_cy = (sy1 + sy2) / 2.0
    if stem_cy < staff_mid:
        # stem 主體在 staff 上半 → 朝上
        # beam 在 stem 頂端
        tip_y_center = sy1
        # ROI：sy1 ± d * 1.0 範圍
        roi_y1 = max(0, int(sy1 - d * 0.3))
        roi_y2 = min(gray.shape[0], int(sy1 + d * 1.5))
    else:
        # stem 朝下，beam 在底端
        tip_y_center = sy2
        roi_y1 = max(0, int(sy2 - d * 1.5))
        roi_y2 = min(gray.shape[0], int(sy2 + d * 0.3))

    # ROI x：以 stem 為中心、左右各 d * 1.5（涵蓋 beam 跨度）
    stem_cx = (sx1 + sx2) / 2.0
    roi_x1 = max(0, int(stem_cx - d * 0.3))
    roi_x2 = min(gray.shape[1], int(stem_cx + d * 1.8))
    if roi_x2 - roi_x1 < 5 or roi_y2 - roi_y1 < 5:
        return 0

    roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
    _, bw = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 對每一行統計黑色像素比例
    row_black = (bw > 0).sum(axis=1) / float(bw.shape[1])

    # 「粗橫條」：黑色比例 > 0.6 的連續行
    # 一條 beam 厚度約 0.3 ~ 0.5 d
    threshold = 0.6
    is_thick = row_black > threshold

    # 找連續區段
    beams = 0
    in_block = False
    block_height = 0
    last_block_end = -100
    for i in range(len(is_thick)):
        if is_thick[i]:
            if not in_block:
                in_block = True
                block_height = 1
            else:
                block_height += 1
        else:
            if in_block:
                # 收 block：必須夠厚（>= 0.15d）才算 beam
                if block_height >= max(2, int(d * 0.15)):
                    # 跟上一個 beam 必須有間隔（避免 staff line 殘留）
                    if i - block_height - last_block_end >= max(2, int(d * 0.1)):
                        beams += 1
                        last_block_end = i - 1
                in_block = False
                block_height = 0
    # 收尾
    if in_block and block_height >= max(2, int(d * 0.15)):
        beams += 1

    return beams


def find_dot_for_note(note_box, dot_dets, d, used_indices, barline_xs=None):
    """
    在 dot_dets 中找最接近這個音符的 dot，回傳 best_idx（-1 表示沒找到）。
    used_indices 是已被前面音符消耗的 dot index set。
    barline_xs（可選）：本 staff 的小節線 x 座標排序 list。
        如果音符右邊界 nx2 與 dot 中心 dcx 之間存在 barline，視為跨小節 → 不配對
        （音樂上：附點屬於其左側音符，不會跨小節線歸屬）
    """
    nx1, ny1, nx2, ny2 = note_box
    ncx, ncy = (nx1 + nx2) / 2, (ny1 + ny2) / 2
    best_idx = -1
    best_dist = float("inf")
    for i, dd in enumerate(dot_dets):
        if i in used_indices:
            continue
        dx1, dy1, dx2, dy2 = dd["box"]
        dcx, dcy = (dx1 + dx2) / 2, (dy1 + dy2) / 2
        gap = dcx - nx2
        if gap < -d * 0.05 or gap > d * 1.0:
            continue
        if abs(dcy - ncy) > d * 0.5:
            continue
        # barline 阻斷：若 nx2 與 dcx 之間有 barline，跳過（附點屬於 barline 右側音符）
        if barline_xs:
            crossed = any(nx2 < bx < dcx for bx in barline_xs)
            if crossed:
                continue
        # 距離評分：x gap 越小越優先
        dist = abs(gap)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx


def count_beams_robust(stem_box, beams, d):
    """
    強化版 beam 計數：用原 count_beams_for_stem 的接觸判定收集所有 touching beam，
    再用 bbox IoU 對重複偵測做去重（避免 YOLO 對同一條 beam 出兩個重疊 bbox）。

    為什麼用 IoU 而非 y 聚類：
      - 真實 16th 雙 beam：兩條 beam 上下平行，y 範圍不重疊 → IoU < 0.2
      - YOLO 重複偵測：同位置兩個 bbox，幾乎完全重合 → IoU > 0.5
      - 用 y 距離閾值無法區分「真實雙 beam 間距 ~6-8 px」與「重複偵測誤差 ~3-5 px」
        在小譜上會誤合併真實的 16th 雙 beam（如 2.png m13 結尾 16C5 16D5）

    為什麼用接觸判定後才做 IoU 去重（而非在 beams 全域去重）：
      - dedupe_beams 已在 beams 全域做過 IoU 去重，這裡是第二道防線
      - 確保「同一個 stem 接觸的 beam」中沒有重複
    """
    sx1, sy1, sx2, sy2 = stem_box

    # 第一步：用原 count_beams_for_stem 的判定收集所有接觸的 beam bbox
    touching_boxes = []
    for beam in beams:
        bx1, by1, bx2, by2 = beam["box"]
        # x interval_gap 判定
        if sx2 < bx1:
            x_gap = bx1 - sx2
        elif bx2 < sx1:
            x_gap = sx1 - bx2
        else:
            x_gap = 0.0
        if x_gap > 5:
            continue
        # y 判定
        y_overlap = max(0, min(sy2, by2) - max(sy1, by1))
        near_top = abs(by1 - sy1)
        near_bottom = abs(by2 - sy2)
        if y_overlap == 0 and min(near_top, near_bottom) > 15:
            continue
        touching_boxes.append((bx1, by1, bx2, by2))

    if not touching_boxes:
        return 0
    if len(touching_boxes) == 1:
        return 1

    # 第二步：去重
    # 條件 A：bbox IoU >= 0.4 → 同位置重複偵測
    # 條件 B：y 範圍幾乎重疊（y_overlap_ratio >= 0.6）且水平相連（x_gap <= 5）
    #         → 同一條 beam 被 YOLO 切成兩段（如 1.png staff 2/6 的情況）
    # 真實雙 beam：y 不重疊（上下平行）→ 兩條件都不會觸發，正確保留
    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / ua if ua > 0 else 0

    def y_overlap_ratio(a, b):
        ay1, ay2 = a[1], a[3]
        by1, by2 = b[1], b[3]
        inter = max(0, min(ay2, by2) - max(ay1, by1))
        union = max(ay2, by2) - min(ay1, by1)
        return inter / union if union > 0 else 0

    def x_overlap_ratio(a, b):
        ax1, ax2 = a[0], a[2]
        bx1, bx2 = b[0], b[2]
        inter = max(0, min(ax2, bx2) - max(ax1, bx1))
        union = max(ax2, bx2) - min(ax1, bx1)
        return inter / union if union > 0 else 0

    def is_duplicate(a, b):
        # A: 重複偵測
        if iou(a, b) >= 0.4:
            return True
        # B: 同條 beam 水平切兩段（y 高度重合 + x 水平相連）
        if y_overlap_ratio(a, b) >= 0.6:
            ax1, ax2 = a[0], a[2]
            bx1, bx2 = b[0], b[2]
            # x 水平間距（負數表示重疊）
            x_gap = max(ax1, bx1) - min(ax2, bx2)
            if x_gap <= 5:  # 重疊或間距 <= 5px 視為同一條
                return True
        # C 已移除：經 1/6/7.png 實測，「單條 beam 被垂直切兩段(假)」與
        #   「真 16th 主beam+beamlet」在 YOLO box 層級的 xOvl/totH 都重疊，
        #   無法用幾何閾值通用區分（7.png 真16th xOvl=0.95、totH/d=1.6，
        #   與 1.png 假切段 xOvl=0.97 幾乎相同）。任何閾值都會在修 1.png 與
        #   壞 7.png 之間二選一，違反「規則須通用」。故不在 inference 端強修，
        #   這 4 個 beam 假切段留待提高輸入解析度或重訓解決。
        return False

    # 按面積由大到小，逐一保留與已保留 box 不重複的
    sorted_b = sorted(touching_boxes,
                      key=lambda b: -((b[2] - b[0]) * (b[3] - b[1])))
    kept = []
    for b in sorted_b:
        if all(not is_duplicate(b, k) for k in kept):
            kept.append(b)
    return len(kept)


def dedupe_beams(beams, iou_thresh=0.3):
    """
    合併 YOLO 對同一條 beam 偵測出的多重 bbox。
    YOLO 對長 beam 偶爾出 2 個重疊框，或一條 beam 被切成 2 段，
    導致 count_beams_for_stem 把同一條 beam 算進 2 次 → stem+2beam 誤判 16th。

    策略：
      - 按 bbox 面積由大到小排序，逐一保留與已保留 beams 的 IoU < iou_thresh 者。
      - 也合併「水平上相連、垂直位置幾乎相同」的兩段 beam（YOLO 切成兩段的情況）。
    """
    if not beams:
        return beams

    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / ua if ua > 0 else 0

    def y_overlap_ratio(a, b):
        ay1, ay2 = a[1], a[3]
        by1, by2 = b[1], b[3]
        inter = max(0, min(ay2, by2) - max(ay1, by1))
        union = max(ay2, by2) - min(ay1, by1)
        return inter / union if union > 0 else 0

    sorted_b = sorted(
        beams,
        key=lambda b: -((b["box"][2] - b["box"][0]) * (b["box"][3] - b["box"][1])),
    )
    kept = []
    for b in sorted_b:
        bb = b["box"]
        is_dup = False
        for k in kept:
            kb = k["box"]
            # 條件 A：IoU 過高，視為同一條 beam
            if iou(bb, kb) >= iou_thresh:
                is_dup = True
                break
            # 條件 B：垂直幾乎重疊 (>=0.7) 且水平相連/重疊（同一條被切兩段）
            if y_overlap_ratio(bb, kb) >= 0.7:
                # 水平相連或交疊
                hx_gap = max(bb[0], kb[0]) - min(bb[2], kb[2])
                # 若 gap <= 0 表示重疊；若 gap 很小（< 10 px）也視為同一條 beam 兩段
                if hx_gap <= 10:
                    is_dup = True
                    break
        if not is_dup:
            kept.append(b)
    return kept


def find_flag_for_stem(stem_box, flag_dets, d):
    """
    找 stem 上是否有對應的 flag 偵測，回傳 flag_level (0/1/2)。

    強化：若同一 stem 找到 >= 2 個 flag bbox（矛盾，正常一條 stem 上 YOLO 只該出 1 個 flag），
    視為偵測噪音，回傳 0（讓上游退為 quarter，再由其他線索修正）。
    這擋掉 staff 4 m1 #1 那種 false flag 連環誤判。

    另外加嚴：x 中心要在 stem x 中心 ±0.8d（原本 ±1.5d 太寬，會被相鄰音符的 flag 誤觸發）
    """
    sx1, sy1, sx2, sy2 = stem_box
    scx = (sx1 + sx2) / 2
    matched = []  # 收集所有匹配到的 flag (cls, conf?)
    for fd in flag_dets:
        fx1, fy1, fx2, fy2 = fd["box"]
        fcx = (fx1 + fx2) / 2
        # x 必須跟 stem 接近（加嚴：1.5d → 0.8d）
        if abs(fcx - scx) > d * 0.8:
            continue
        # y 必須在 stem 範圍內或附近
        if fy2 < sy1 - d * 0.5 or fy1 > sy2 + d * 0.5:
            continue
        cls = fd["class_name"]
        if cls in ("flag16thUp", "flag16thDown"):
            # flag16th 可靠度低：信心不足時降級為 flag8th（8th），
            # 避免把 flagged 8th 因單一低信心 flag16th 誤判成 16th。
            if fd.get("conf", 1.0) < FLAG16_MIN_CONF:
                matched.append(1)
            else:
                matched.append(2)
        elif cls in ("flag8thUp", "flag8thDown"):
            matched.append(1)
    if not matched:
        return 0
    if len(matched) >= 2:
        # 矛盾：一條 stem 不該有多個 flag bbox，當作雜訊
        return 0
    return matched[0]


# ========================================================
# 音高判斷（保留 PitchCNN，但 staff line 規則優先）
# ========================================================
def classify_pitch_for_note(gray, note_box, notehead_class, staff_lines_y,
                            pitch_model, idx_to_class, device, d):
    """
    回傳 pitch 字串（如 "F4"）。
    優先用「staff line + notehead class」規則，失敗才回退 PitchCNN。
    """
    pitch_rule = infer_step_and_pitch_from_lines(
        note_box, notehead_class, staff_lines_y)
    # infer_step_and_pitch_from_lines 回傳 (step_idx, pitch_str) tuple
    if isinstance(pitch_rule, tuple) and len(pitch_rule) >= 2:
        pitch_str = pitch_rule[1]
        if pitch_str and pitch_str != "---":
            return pitch_str
    elif isinstance(pitch_rule, str) and pitch_rule and pitch_rule != "---":
        return pitch_rule

    # 規則失敗 → 不再用 PitchCNN,直接回預設值(音高純靠 OpenCV 譜線幾何)
    return "C4"


# ========================================================
# MIDI 匯出
# ========================================================
def pitch_to_midi(p):
    if not p or p in ("---", "R"):
        return 60
    # 通用解析：大寫 A-G = 音名、'#' = 升、小寫 'b' = 降。
    # 不挑記號與音名的前後順序（'bB' / 'Bb' / '#F' / 'F#' 都吃），
    # 也不挑拼法（不再依賴固定的 'A#'/'Db' key，避免漏拼法 -> 誤判成 C）。
    base = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    try:
        octave = int(p[-1])
        name = p[:-1]
        letter, acc = None, 0
        for ch in name:
            if ch in base:        # 大寫音名
                letter = ch
            elif ch == '#':       # 升記號
                acc += 1
            elif ch == 'b':       # 小寫 b = 降記號（大寫 B 已當作音名）
                acc -= 1
        if letter is None:
            return 60
        return (octave + 1) * 12 + base[letter] + acc
    except Exception:
        return 60


def add_dot_to_duration(duration):
    """把一個 duration 加附點。已經是 dotted 的不變。"""
    if duration is None or duration.startswith("dotted-"):
        return duration
    return f"dotted-{duration}"


def duration_to_beats(duration):
    if not duration:
        return 1.0
    base = {"whole": 4.0, "half": 2.0, "quarter": 1.0, "8th": 0.5, "16th": 0.25}
    if duration.startswith("dotted-"):
        return base.get(duration.replace("dotted-", ""), 1.0) * 1.5
    return base.get(duration, 1.0)


def _vlq(value):
    value = int(value)
    out = [value & 0x7F]
    value >>= 7
    while value > 0:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(out))


def _write_midi(notes, path, tempo=120, program=74, tpqn=480):
    """notes: list of (start_beats, dur_beats, midi_pitch, velocity)。輸出單軌 format-0 MIDI。"""
    evs = []  # (tick, order, data)
    mpqn = int(60_000_000 / tempo)
    evs.append((0, 0, bytes([0xFF, 0x51, 0x03]) + mpqn.to_bytes(3, "big")))  # tempo
    evs.append((0, 0, bytes([0xC0, program & 0x7F])))                        # program change
    for (start, dur, pitch, vel) in notes:
        s = int(round(start * tpqn))
        e = int(round((start + dur) * tpqn))
        if e <= s:
            e = s + 1
        evs.append((s, 2, bytes([0x90, pitch & 0x7F, vel & 0x7F])))  # note on
        evs.append((e, 1, bytes([0x80, pitch & 0x7F, 0])))           # note off(同 tick 先 off)
    evs.sort(key=lambda x: (x[0], x[1]))
    track = bytearray()
    prev = 0
    for tick, _, data in evs:
        track += _vlq(tick - prev) + data
        prev = tick
    track += _vlq(0) + bytes([0xFF, 0x2F, 0x00])  # end of track
    header = b"MThd" + (6).to_bytes(4, "big") + (0).to_bytes(2, "big") + \
             (1).to_bytes(2, "big") + tpqn.to_bytes(2, "big")
    chunk = b"MTrk" + len(track).to_bytes(4, "big") + bytes(track)
    with open(path, "wb") as f:
        f.write(header)
        f.write(chunk)


def events_to_midi(events, path):
    """輸出 MIDI(處理 tie 合併:tied_to_prev 不發音但時間推進;effective_beats 作發音長度)。"""
    notes = []
    t = 0.0
    for ev in events:
        beats = duration_to_beats(ev.get("duration"))
        if ev["type"] == "note":
            if not ev.get("tied_to_prev"):
                play_beats = ev.get("effective_beats", beats)
                notes.append((t, play_beats, pitch_to_midi(ev["pitch"]), 100))
        t += beats
    _write_midi(notes, str(path))


# ========================================================
# Debug 圖
# ========================================================
def draw_debug(bgr, events, staff_lines_all, stems, beams):
    img = bgr.copy() if bgr is not None else None
    if img is None:
        return None
    # 畫 staff lines
    for sl in staff_lines_all:
        for ly in sl:
            cv2.line(img, (0, int(ly)), (img.shape[1], int(ly)), (255, 100, 100), 1)
    # 畫 stems
    for st in stems:
        x1, y1, x2, y2 = st["box"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 200), 1)
    # 畫 beams
    for bm in beams:
        x1, y1, x2, y2 = bm["box"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (200, 0, 200), 1)
    # 畫 events
    for ev in events:
        if "box" not in ev:
            continue
        x1, y1, x2, y2 = ev["box"]
        if ev["type"] == "note":
            color = (255, 0, 0)
            text = f"{ev.get('pitch','?')}{ev.get('duration','q')[:1]}"
        else:
            color = (0, 255, 0)
            text = "R"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        cv2.putText(img, text, (x1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return img


# ========================================================
# 主流程
# ========================================================
def main():
    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(msg)

    # 重置跨張診斷收集器（保險：即使未來改成單程序跑多張也不會累積）
    _GEOM_DUMP.clear()

    log(f"[INFO] 處理：{IMAGE_PATH.name}")

    gray = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_UNCHANGED)
    bgr = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_COLOR)
    if gray is None:
        log("[ERROR] 讀不到圖片")
        return
    if len(gray.shape) == 3:
        ch = gray.shape[2]
        if ch == 4:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGRA2GRAY)
        elif ch == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    # 低解析超解析前處理（通用：間距過低才啟用，正常譜不動）
    gray, bgr = maybe_superres(gray, bgr, log)

    # 解析度標準化
    gray, bgr, scale = standardize_resolution(gray, bgr, log)
    img_h, img_w = gray.shape[:2]

    # YOLO 由外部 runner 執行(對處理後的圖)
    yolo_model = _YOLO_RUNNER
    is_multi = True
    if yolo_model is None:
        log("[ERROR] 未設定 YOLO runner")
        return

    # 不再使用 PitchCNN(音高純靠 OpenCV 譜線幾何)
    pitch_model, idx_to_class = None, {}
    device = "cpu"
    log(f"[INFO] 裝置: {device}(無 CNN,純 YOLO + OpenCV)")

    # SymbolCNN（附點二次驗證用，可選）
    # SymbolCNN / dot_classifier 已移除(附點純靠 YOLO + 幾何)

    # === YOLO 偵測 ===
    log("[INFO] YOLO 偵測 (21 類)...")
    # 因為圖可能被縮放，存暫存檔給 YOLO
    # 一律用處理後的 gray（含超解析/標準化）給 YOLO，避免回讀到原始低解析檔
    tmp_path = Path(tempfile.gettempdir()) / f"_scaled_{IMAGE_PATH.stem}.png"
    cv2.imwrite(str(tmp_path), gray)
    if is_multi:
        all_detections = detect_with_yolo_multi(yolo_model, tmp_path)
    else:
        all_detections = detect_with_yolo_single(yolo_model, tmp_path)
    try:
        tmp_path.unlink()
    except Exception:
        pass
    log(f"[INFO] 偵測到 {len(all_detections)} 個物件")

    # === Rest NMS dedupe ===
    # YOLO 偶爾對同一個休止符出兩個重疊 bbox（如 5.png staff 3 m15 末尾的 quarter rest
    # 被偵測成 8th rest + quarter rest 兩個），會造成事件數比 GT 多 1，後續錯位。
    # 解法：對所有 rest 類做 IoU dedupe（IoU >= 0.4 視為重複）。
    REST_CLASSES = {"rest8th", "rest16th", "restQuarter", "restHalf", "restWhole"}
    def _bbox_iou(a, b):
        ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
        inter = iw * ih
        ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / ua if ua > 0 else 0

    rest_dets = [d_ for d_ in all_detections if d_["class_name"] in REST_CLASSES]
    nonrest_dets = [d_ for d_ in all_detections if d_["class_name"] not in REST_CLASSES]

    # === DEBUG：列出所有 rest 偵測 + 兩兩之間的關係 ===
    # 用於診斷「重複偵測但 IoU 沒到 0.4」的邊界情況
    if rest_dets:
        log(f"[DEBUG-REST] 共 {len(rest_dets)} 個 rest 偵測：")
        # 按 (class, x) 排序方便看
        rest_sorted_for_log = sorted(rest_dets,
                                     key=lambda d_: ((d_["box"][0]+d_["box"][2])/2,
                                                     d_["box"][1]))
        for idx, d_ in enumerate(rest_sorted_for_log):
            x1, y1, x2, y2 = d_["box"]
            cx, cy = (x1+x2)/2, (y1+y2)/2
            w, h = x2-x1, y2-y1
            conf = d_.get("conf", d_.get("confidence", 0.0))
            log(f"  #{idx:2d} {d_['class_name']:12s} "
                f"bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) "
                f"center=({cx:.0f},{cy:.0f}) wxh={w:.0f}x{h:.0f} "
                f"conf={conf:.3f}")
        # 列出「距離很近的 rest 對」（中心距離 < 60px 的就值得看）
        log(f"[DEBUG-REST] 距離較近的 rest 對（中心 < 60px）：")
        n_close_pairs = 0
        for i in range(len(rest_sorted_for_log)):
            for j in range(i+1, len(rest_sorted_for_log)):
                a = rest_sorted_for_log[i]; b = rest_sorted_for_log[j]
                ax1, ay1, ax2, ay2 = a["box"]; bx1, by1, bx2, by2 = b["box"]
                acx, acy = (ax1+ax2)/2, (ay1+ay2)/2
                bcx, bcy = (bx1+bx2)/2, (by1+by2)/2
                dx = bcx - acx; dy = bcy - acy
                center_dist = (dx*dx + dy*dy)**0.5
                if center_dist > 60:
                    continue
                iou = _bbox_iou(a["box"], b["box"])
                log(f"  pair #{i}({a['class_name']}) ↔ #{j}({b['class_name']}): "
                    f"中心距={center_dist:.1f}px (dx={dx:+.0f},dy={dy:+.0f}) IoU={iou:.3f}")
                n_close_pairs += 1
        if n_close_pairs == 0:
            log(f"  （無）")

    # 按 conf 由大到小，保留與已保留 box IoU < 0.4 的（高 conf 優先）
    rest_sorted = sorted(rest_dets,
                         key=lambda d_: -d_.get("conf", d_.get("confidence", 0.5)))
    kept_rests = []
    for d_ in rest_sorted:
        if all(_bbox_iou(d_["box"], k["box"]) < 0.4 for k in kept_rests):
            kept_rests.append(d_)
    n_removed_iou = len(rest_dets) - len(kept_rests)
    if n_removed_iou > 0:
        log(f"[INFO] Rest dedupe (IoU>=0.4): 移除 {n_removed_iou} 個重複偵測 "
            f"({len(rest_dets)} → {len(kept_rests)})")

    # === 低 conf rest 過濾（解決 IoU 抓不到的「同物異類別重複」）===
    # 動機：5.png staff 3 一個 quarter rest 被 YOLO 同時偵測成
    #   rest8th  conf=0.368 在上方位置 (y=1183)
    #   restQuarter conf=0.920 在下方位置 (y=1242)
    # 兩個 bbox 不重疊（IoU=0），但中心 x 距離 88px 在同小節範圍內，
    # 顯然是「同一個 rest 物體 YOLO 出兩種預測」。
    #
    # 規則（兩條件同時成立才移除低 conf 那個）：
    #   1. 自己 conf < LOW_CONF_THRESH (0.5)
    #   2. 中心 1.5d 範圍內存在另一個 conf > HIGH_CONF_THRESH (0.8) 的 rest
    #
    # 這個規則之所以安全：
    #   - 真實「連續兩個 rest」：兩個都會清楚被偵測 → conf 都 > 0.8 → 條件 1 不成立 → 不殺
    #   - 整張譜糊掉導致 rest 都低 conf：找不到 conf > 0.8 對手 → 條件 2 不成立 → 不殺
    #   - 只有「YOLO 對同物體出兩種類別」這種真實「該殺」的情境會觸發
    LOW_CONF_THRESH = 0.5
    HIGH_CONF_THRESH = 0.8
    # NEIGHBOR_RADIUS 設定考量：
    #   - 5.png 案例兩個 bbox 中心距離 106 px (dx=88, dy=59)，要包含進來
    #   - 但範圍太大會誤殺真實「連續兩個 rest」（如 8R 8R）
    #   - 真實連續兩個 rest **兩個 conf 都應該 > 0.8**（清楚的休止符）
    #     → 條件 1 (低 conf<0.5) 不會成立 → 不會誤殺
    #   - 因此 120 px 安全：規則只在「一個 conf 異常低 + 附近有高 conf」觸發
    NEIGHBOR_RADIUS = 120.0

    def _center(box):
        x1, y1, x2, y2 = box
        return (x1+x2)/2, (y1+y2)/2

    def _conf(d_):
        return d_.get("conf", d_.get("confidence", 0.5))

    survivors = []
    n_removed_lowconf = 0
    for d_ in kept_rests:
        c = _conf(d_)
        if c >= LOW_CONF_THRESH:
            survivors.append(d_)
            continue
        # 找鄰近的高 conf rest 對手
        my_cx, my_cy = _center(d_["box"])
        found_neighbor = None
        for other in kept_rests:
            if other is d_:
                continue
            other_c = _conf(other)
            if other_c <= HIGH_CONF_THRESH:
                continue
            ox, oy = _center(other["box"])
            dist = ((ox - my_cx)**2 + (oy - my_cy)**2)**0.5
            if dist <= NEIGHBOR_RADIUS:
                found_neighbor = (other, dist, other_c)
                break
        if found_neighbor is not None:
            other, dist, other_c = found_neighbor
            log(f"[INFO] Rest 低信心過濾: 移除 {d_['class_name']} "
                f"center=({my_cx:.0f},{my_cy:.0f}) conf={c:.3f} "
                f"（鄰近 {NEIGHBOR_RADIUS:.0f}px 有 {other['class_name']} "
                f"conf={other_c:.3f}，距離={dist:.1f}px）")
            n_removed_lowconf += 1
        else:
            survivors.append(d_)

    kept_rests = survivors
    n_removed = n_removed_iou + n_removed_lowconf
    if n_removed_lowconf > 0:
        log(f"[INFO] Rest 低信心過濾共移除 {n_removed_lowconf} 個（"
            f"條件：conf<{LOW_CONF_THRESH} 且鄰近 {NEIGHBOR_RADIUS:.0f}px 有 "
            f"conf>{HIGH_CONF_THRESH} 的 rest）")
    all_detections = nonrest_dets + kept_rests

    # 按類別分組
    by_class = {}
    for d in all_detections:
        by_class.setdefault(d["class_name"], []).append(d)
    log("[INFO] 各類別偵測數：")
    for cn in sorted(by_class.keys()):
        log(f"  {cn}: {len(by_class[cn])}")

    # === 偵測五線譜 ===
    log("[INFO] 偵測五線譜...")
    staff_boxes = auto_detect_staff_boxes(gray)
    staff_lines_all = []
    valid_staff_boxes = []
    for sb in staff_boxes:
        ly = detect_staff_lines(gray, sb)
        if ly is not None and len(ly) == 5:
            staff_lines_all.append(ly)
            valid_staff_boxes.append(sb)
    log(f"[INFO] 共 {len(valid_staff_boxes)} 條 staff")

    # 取參考 spacing
    if staff_lines_all:
        all_diffs = []
        for ly in staff_lines_all:
            all_diffs.extend(np.diff(sorted(ly)))
        ref_d = float(np.median(all_diffs)) if all_diffs else 20.0
    else:
        ref_d = 20.0
    log(f"[INFO] 參考譜線間距 d = {ref_d:.2f} px")

    # === 對每個 staff 處理 ===
    events = []
    all_stems = []
    all_beams = []
    all_barlines = {}        # staff_idx → [x1, x2, ...] sorted
    all_staff_x_ranges = {}  # staff_idx → (sx1, sx2)

    for staff_idx, staff_box in enumerate(valid_staff_boxes):
        sx1, sy1, sx2, sy2 = staff_box
        staff_lines_y = staff_lines_all[staff_idx]
        sorted_lines = sorted(staff_lines_y)
        d = float(np.median(np.diff(sorted_lines))) if len(sorted_lines) >= 2 else ref_d

        # OpenCV 偵測 stem / barline（beam 改用 YOLO）
        stems = detect_stems_in_staff(gray, staff_box, staff_lines_y)
        barlines = detect_barlines_in_staff(gray, staff_box, staff_lines_y)

        # beam 從 YOLO detections 過濾（用新訓練的 22 類模型）
        def in_staff_for_beam(box):
            _, by1, _, by2 = box
            cy = (by1 + by2) / 2
            # beam 在 stem 上端或下端、可能略超出 staff 範圍
            return sy1 - d * 4 <= cy <= sy2 + d * 4
        beams_yolo = [d_ for d_ in all_detections
                      if d_["class_name"] == "beam" and in_staff_for_beam(d_["box"])]
        # 轉成 {"box": (x1,y1,x2,y2)} 格式，跟舊 OpenCV beam 一樣，方便相容下游
        beams_raw = [{"box": b["box"]} for b in beams_yolo]
        # 對 YOLO beam 做 NMS dedupe（解 stem+2beam 誤判 16th 的源頭）
        beams_raw_for_dump = [b["box"] for b in beams_raw]  # [診斷] dedupe 前
        beams = dedupe_beams(beams_raw, iou_thresh=0.3)
        if len(beams) < len(beams_raw):
            log(f"        [BEAM_DEDUPE] staff {staff_idx}: "
                f"{len(beams_raw)} → {len(beams)} (合併 {len(beams_raw) - len(beams)} 條重疊)")

        all_stems.extend(stems)
        all_beams.extend(beams)
        all_barlines.setdefault(staff_idx, [])
        all_barlines[staff_idx] = [bl["x"] for bl in barlines]
        all_staff_x_ranges[staff_idx] = (sx1, sx2)
        log(f"[DEBUG] staff {staff_idx}: 偵測到 {len(barlines)} 條小節線, "
            f"{len(beams)} 條 beam (來自 YOLO)")

        # 篩出這個 staff 範圍內的 YOLO 偵測
        # noteheads 嚴一點：staff 上方 4d、下方 4d 內才算（避免抓到譜頂 ♩=100）
        # rest 寬一點，因為 whole rest 在 staff 上方
        def in_staff_for_note(box):
            _, by1, _, by2 = box
            cy = (by1 + by2) / 2
            return sy1 - d * 3 <= cy <= sy2 + d * 3
        def in_staff_loose(box):
            _, by1, _, by2 = box
            cy = (by1 + by2) / 2
            return sy1 - d * 4 <= cy <= sy2 + d * 4

        staff_noteheads = [d_ for d_ in all_detections
                           if d_["class_name"] in NOTEHEAD_CLASSES and in_staff_for_note(d_["box"])]
        staff_rests = [d_ for d_ in all_detections
                       if d_["class_name"] in REST_CLASSES and in_staff_loose(d_["box"])]
        staff_flags = [d_ for d_ in all_detections
                       if d_["class_name"] in FLAG_CLASSES and in_staff_loose(d_["box"])]
        staff_dots_raw = [d_ for d_ in all_detections
                          if d_["class_name"] == "augmentationDot" and in_staff_loose(d_["box"])]

        # 印出所有候選 dot 的 conf 跟位置（診斷用）
        if staff_dots_raw:
            log(f"        [DOT] staff {staff_idx} candidates: {len(staff_dots_raw)} 個")
            for i, dot in enumerate(staff_dots_raw):
                bx1, by1, bx2, by2 = dot["box"]
                log(f"          #{i} x={(bx1+bx2)/2:.0f} y={(by1+by2)/2:.0f} "
                    f"conf={dot.get('conf', 0):.2f}")

        # 二次驗證：用 SymbolCNN（如果有）
        # 附點直接信 YOLO augmentationDot(不再用 SymbolCNN 二次驗證)
        staff_dots = staff_dots_raw

        log(f"[DEBUG] staff {staff_idx}: {len(staff_noteheads)} notes, "
            f"{len(stems)} stems, {len(beams)} beams, "
            f"{len(staff_rests)} rests, {len(staff_flags)} flags, {len(staff_dots)} dots")

        # 處理音符（先按 x 排序，從左到右消耗 dot）
        sorted_noteheads = sorted(staff_noteheads,
                                  key=lambda n: (n["box"][0] + n["box"][2]) / 2)
        used_dot_indices = set()

        # === [診斷] 記錄這條 staff 的原始幾何（不影響任何判斷）===
        if ENABLE_GEOM_DUMP:
            def _b(x):
                return [int(round(v)) for v in x]
            _GEOM_DUMP.append({
                "staff_idx": int(staff_idx),
                "d": float(d),
                "staff_lines_y": [float(v) for v in staff_lines_y],
                "beams": [_b(bm["box"]) for bm in beams],
                "beams_raw": [_b(b) for b in beams_raw_for_dump],
                "stems": [_b(st["box"]) for st in stems],
                "flags": [{"box": _b(fd["box"]),
                           "cls": fd["class_name"],
                           "conf": float(fd.get("conf", 0.0))} for fd in staff_flags],
                "noteheads": [{"box": _b(n["box"]),
                               "cls": n["class_name"],
                               "conf": float(n.get("conf", 0.0))} for n in staff_noteheads],
            })
        for note in sorted_noteheads:
            note_box = note["box"]
            cx, cy = bbox_center(note_box)
            note_type = note["class_name"]  # noteheadBlack/Half/Whole

            # 音高
            pitch = classify_pitch_for_note(
                gray, note_box, note_type, staff_lines_y, pitch_model,
                idx_to_class, device, d)


            # 時值判斷
            duration_reason = ""
            if note_type == "noteheadWhole":
                duration = "whole"
                duration_reason = "noteheadWhole"
            elif note_type == "noteheadHalf":
                duration = "half"
                duration_reason = "noteheadHalf"
            else:
                # noteheadBlack：用 stem + beam/flag 判斷
                # 修改：beam 優先於 flag（音樂上互斥，且 beam mAP=0.939 > flag16th 的可靠性）
                # 原本是「先看 flag，flag==0 才看 beam」，但這樣 YOLO 若把 flag8th 誤分為
                # flag16th，或一個 flag bbox 被相鄰 stem 共用，就會把 beamed 8th 誤判 16th
                stem_obj = find_stem_for_note(note_box, stems)
                if stem_obj is None:
                    duration = "quarter"
                    duration_reason = "no_stem"
                else:
                    # 改用 count_beams_robust 取代原 count_beams_for_stem
                    # 解 stem+2beam 誤判：原函式把同一條 beam 的兩段或主+副 beam 重疊算成 2
                    beam_count = count_beams_robust(stem_obj["box"], beams, d)
                    if beam_count >= 1:
                        if beam_count >= 2:
                            duration = "16th"
                        else:
                            duration = "8th"
                        duration_reason = f"stem+{beam_count}beam"
                    else:
                        # 沒有 beam 才看 flag
                        flag_level = find_flag_for_stem(stem_obj["box"], staff_flags, d)
                        if flag_level == 1:
                            duration = "8th"
                            duration_reason = "flag1"
                        elif flag_level == 2:
                            duration = "16th"
                            duration_reason = "flag2"
                        else:
                            duration = "quarter"
                            duration_reason = "stem+0beam"

            # 附點：用唯一消耗 dot index，加 barline 阻斷（附點不跨小節）
            staff_barline_xs = [bl["x"] for bl in barlines]
            dot_idx = find_dot_for_note(note_box, staff_dots, d,
                                        used_dot_indices, barline_xs=staff_barline_xs)
            has_dot = dot_idx >= 0
            if has_dot:
                used_dot_indices.add(dot_idx)
                if duration:
                    duration = f"dotted-{duration}"

            events.append({
                "type": "note",
                "pitch": pitch,
                "duration": duration,
                "has_dot": has_dot,
                "duration_reason": duration_reason,
                "box": note_box,
                "x": cx,
                "y": cy,
                "staff_idx": staff_idx,
            })

        # 處理休止符
        for rest in staff_rests:
            rb = rest["box"]
            rcx, rcy = bbox_center(rb)
            cls = rest["class_name"]
            dur_map = {
                "restWhole": "whole", "restHalf": "half",
                "restQuarter": "quarter", "rest8th": "8th", "rest16th": "16th",
            }
            events.append({
                "type": "rest",
                "pitch": "R",
                "duration": dur_map.get(cls, "quarter"),
                "has_dot": False,
                "box": rb,
                "x": rcx,
                "y": rcy,
                "staff_idx": staff_idx,
            })

    # 按 staff_idx 然後 x 排序
    events.sort(key=lambda e: (e["staff_idx"], e["x"]))

    # ===== 後處理 0：Tie（連結線）合併 =====
    # YOLO 偵測到 tie 弧線時，若弧線兩端的音符同音高，後者應「不發音、時長加到前者」。
    # slur（連不同音高）不處理；tie 也只在兩端同 pitch 才合併。
    # 為節省搜尋成本，把全圖 tie 一次取出，按 x 排序
    all_ties = [d_ for d_ in all_detections if d_["class_name"] == "tie"]
    # 按 x 中心排序
    all_ties_sorted = sorted(all_ties,
                             key=lambda t: (t["box"][0] + t["box"][2]) / 2)

    n_tie_merged = 0
    n_tie_skipped_slur = 0  # 同 bbox 兩端音高不同（其實是 slur）
    n_tie_skipped_nopair = 0  # 找不到兩端音符

    def event_center(e):
        if "box" in e:
            x1, _, x2, _ = e["box"]
            return (x1 + x2) / 2
        return e.get("x", 0)

    # 取 tie bbox 兩端最近的 note：左端 = 在 tie 左邊界附近、右端 = 在 tie 右邊界附近
    # 限制條件：兩端 note 必須在「相鄰」位置（沒有別的 note 夾在中間），但允許跨 staff（tie 跨行）
    notes_only = [e for e in events if e["type"] == "note"]

    for t in all_ties_sorted:
        tx1, ty1, tx2, ty2 = t["box"]
        tcx = (tx1 + tx2) / 2
        tcy = (ty1 + ty2) / 2

        # 找最接近 tie 左端的 note：x 中心 < tcx 且 |y - tcy| 不太遠
        # 找最接近 tie 右端的 note：x 中心 > tcx 且 |y - tcy| 不太遠
        # 由於 tie 弧線通常彎到音符上方/下方一點點，y 距離容差設為 d * 3
        Y_TOL = 60  # 約 d * 3，足以容納 tie 弧線到 notehead 的垂直距離
        left_cand = None
        right_cand = None
        left_dist = float("inf")
        right_dist = float("inf")
        for e in notes_only:
            if "box" not in e:
                continue
            ecx = event_center(e)
            ecy = e.get("y", 0)
            if abs(ecy - tcy) > Y_TOL:
                continue
            # 必須有合理的 x 接近度
            if ecx < tcx:
                # 候選左端
                d_left = abs(ecx - tx1)  # 越靠近 tie 左邊界越好
                # 排除距離過遠（>2*tie 寬度）
                if d_left < (tx2 - tx1) * 1.5 and d_left < left_dist:
                    left_dist = d_left
                    left_cand = e
            else:
                # 候選右端
                d_right = abs(ecx - tx2)
                if d_right < (tx2 - tx1) * 1.5 and d_right < right_dist:
                    right_dist = d_right
                    right_cand = e

        if left_cand is None or right_cand is None:
            n_tie_skipped_nopair += 1
            continue

        # 同音高才合併（不同 pitch 視為 slur，跳過）
        if left_cand.get("pitch") != right_cand.get("pitch"):
            n_tie_skipped_slur += 1
            continue

        # 已被合併過的不再合併（避免 tie 偵測重疊造成雙重合併）
        if right_cand.get("tied_to_prev"):
            continue

        # 合併：在 right_cand 上標 tied_to_prev，duration 加到 left_cand
        added_beats = duration_to_beats(right_cand.get("duration"))
        right_cand["tied_to_prev"] = True
        right_cand["tied_duration_added"] = added_beats
        # 也把 left 的 duration 累加（之後 events_to_midi 要看 effective_duration）
        left_cand_beats = duration_to_beats(left_cand.get("duration"))
        # 把 tied 後的總拍數記在 left_cand
        left_cand["effective_beats"] = (
                left_cand.get("effective_beats", left_cand_beats) + added_beats
        )
        n_tie_merged += 1

    if all_ties:
        log(f"[POST] Tie 合併：偵測 {len(all_ties)} 個 tie，"
            f"合併 {n_tie_merged} 個音符對，"
            f"跳過 slur(不同pitch) {n_tie_skipped_slur} 個，"
            f"找不到兩端 {n_tie_skipped_nopair} 個")

    # ===== 後處理 1.6：臨時記號傳遞（accidental propagation）=====
    # 動機：3.png 有 3 個 #F4、5.png 有 4 個 #F4/#G4 被判成 F4/G4，
    # YOLO accidentalSharp/Flat 已偵測到但 inference 沒套用。
    # 規則（音樂理論）：
    #   1. accidentalSharp/Flat/Natural 緊貼某個音符**左邊**（同 y）
    #   2. 它影響「該小節內、它右邊、相同音名（C/D/E/F/G/A/B）」的所有音符
    #   3. accidental 不跨小節線
    #   4. 只處理 accidentalSharp/Flat/Natural（class 14/15/16），
    #      不處理 keySharp/keyFlat（class 17/18 是譜頭調號，依本 case GT 不應用）
    ENABLE_ACCIDENTAL_PROP = True
    if ENABLE_ACCIDENTAL_PROP:
        # 1. 蒐集所有 accidental 偵測
        ACC_MAP = {"accidentalSharp": "#",
                   "accidentalFlat":  "b",
                   "accidentalNatural": ""}
        all_accidentals = [d_ for d_ in all_detections
                           if d_.get("class_name") in ACC_MAP]

        n_acc_applied = 0
        n_acc_unmatched = 0

        # 2. 按 staff 分組處理
        for staff_idx in sorted(set(e["staff_idx"] for e in events)):
            staff_events = [e for e in events if e["staff_idx"] == staff_idx]
            if not staff_events:
                continue
            # staff y 範圍：粗估從 events 推
            staff_ys = [e["y"] for e in staff_events]
            staff_y_min = min(staff_ys) - 3 * ref_d
            staff_y_max = max(staff_ys) + 3 * ref_d
            # 找出屬於這 staff 的 accidental
            staff_accs = []
            for d_ in all_accidentals:
                x1, y1, x2, y2 = d_["box"]
                acy = (y1 + y2) / 2
                if staff_y_min <= acy <= staff_y_max:
                    staff_accs.append(d_)
            if not staff_accs:
                continue

            barline_xs = sorted(all_barlines.get(staff_idx, []))

            def measure_idx_of_x(x):
                """x 在第幾個小節（0-based）"""
                idx = 0
                for bx in barline_xs:
                    if x < bx:
                        return idx
                    idx += 1
                return idx

            # 3. 對每個 accidental，找它「右邊最近、同 y」的音符當「主音」
            for acc in staff_accs:
                ax1, ay1, ax2, ay2 = acc["box"]
                acy = (ay1 + ay2) / 2
                acc_right_x = ax2  # accidental 右邊界
                acc_class = acc["class_name"]
                acc_mark = ACC_MAP[acc_class]
                acc_measure = measure_idx_of_x((ax1 + ax2) / 2)

                # 找「acc 右邊最近、垂直距離 < 0.6d」的 note 當主音
                # 這個 note 用來確定「音名字母」(C/D/E/F/G/A/B)
                # 然後把這個 note 跟後續同小節內、同音名字母的 note 全部加上記號
                candidates = []
                for e in staff_events:
                    if e.get("type") != "note":
                        continue
                    pitch = e.get("pitch")
                    if not pitch or pitch in ("---", "R"):
                        continue
                    # x 在 acc 右邊 0~1.5d 範圍
                    if not (acc_right_x <= e["x"] <= acc_right_x + 1.5 * ref_d):
                        continue
                    # 垂直距離小（同音線/間）
                    if abs(e["y"] - acy) > 0.6 * ref_d:
                        continue
                    candidates.append(e)
                if not candidates:
                    n_acc_unmatched += 1
                    log(f"[ACC-DEBUG] staff {staff_idx} accidental "
                        f"{acc_class} center=({(ax1+ax2)/2:.0f},{acy:.0f}) "
                        f"找不到右側對應的音符")
                    continue
                # 取最近的當主音
                main = min(candidates, key=lambda e: e["x"])
                # 解析主音字母（去除已有的 #/b）
                main_pitch = main["pitch"]
                # pitch 格式 "F4" / "#F4" / "bF4"
                if main_pitch[0] in "#b":
                    letter = main_pitch[1]
                    octave = main_pitch[2:]
                else:
                    letter = main_pitch[0]
                    octave = main_pitch[1:]

                # 4. 找出受影響的音符：同 staff、同 measure、x >= main.x、letter 相同
                main_x = main["x"]
                affected = []
                for e in staff_events:
                    if e.get("type") != "note":
                        continue
                    pitch = e.get("pitch")
                    if not pitch or pitch in ("---", "R"):
                        continue
                    if e["x"] < main_x:
                        continue
                    # 必須在同小節
                    if measure_idx_of_x(e["x"]) != acc_measure:
                        continue
                    # 音名字母相同（不論八度）
                    p = e["pitch"]
                    if p[0] in "#b":
                        ep_letter = p[1]
                    else:
                        ep_letter = p[0]
                    if ep_letter != letter:
                        continue
                    affected.append(e)

                # 5. 套用記號（更新 pitch 字串；不重複加）
                for e in affected:
                    p = e["pitch"]
                    # 移除已有記號（如果有的話）
                    if p[0] in "#b":
                        bare_letter = p[1]
                        bare_octave = p[2:]
                    else:
                        bare_letter = p[0]
                        bare_octave = p[1:]
                    new_pitch = f"{acc_mark}{bare_letter}{bare_octave}"
                    if new_pitch != p:
                        e["pitch"] = new_pitch
                        e["accidental_applied"] = acc_class
                        n_acc_applied += 1
                        log(f"[ACC] staff {staff_idx} m{acc_measure+1} "
                            f"{acc_class}: {p} → {new_pitch} (x={e['x']:.0f})")

        if n_acc_applied > 0 or n_acc_unmatched > 0:
            log(f"[POST] 臨時記號傳遞: 套用 {n_acc_applied} 個音符 "
                f"(無對應主音: {n_acc_unmatched} 個)")

    # ===== 後處理 1.7：調號傳遞（key signature propagation）=====
    # 動機：F 大調譜頭 1 個 flat → 全曲 B 應讀作 Bb；
    # D 大調譜頭 2 個 sharp → 全曲 F 跟 C 應升半音；以此類推。
    # 規則：
    #   1. 蒐集每 staff 的 keySharp / keyFlat 偵測
    #   2. 按譜頭出現的數量決定影響哪些音名：
    #      sharp 順序: F → C → G → D → A → E → B
    #      flat  順序: B → E → A → D → G → C → F
    #   3. 影響整個 staff 內所有未被 accidental 修飾過的同音名音符
    #   4. accidental 優先（已套用 accidental 的不再套 key）
    ENABLE_KEY_PROP = True
    SHARP_ORDER = ["F", "C", "G", "D", "A", "E", "B"]
    FLAT_ORDER  = ["B", "E", "A", "D", "G", "C", "F"]
    if ENABLE_KEY_PROP:
        n_key_applied = 0
        # 按 staff 分組統計譜頭調號
        for staff_idx in sorted(set(e["staff_idx"] for e in events)):
            staff_events = [e for e in events if e["staff_idx"] == staff_idx]
            if not staff_events:
                continue
            staff_ys = [e["y"] for e in staff_events]
            staff_y_min = min(staff_ys) - 3 * ref_d
            staff_y_max = max(staff_ys) + 3 * ref_d
            # 找該 staff 的 keySharp/keyFlat
            staff_keys_sharp = []
            staff_keys_flat = []
            for d_ in all_detections:
                cn = d_.get("class_name")
                if cn not in ("keySharp", "keyFlat"):
                    continue
                x1, y1, x2, y2 = d_["box"]
                kcy = (y1 + y2) / 2
                if not (staff_y_min <= kcy <= staff_y_max):
                    continue
                if cn == "keySharp":
                    staff_keys_sharp.append(d_)
                else:
                    staff_keys_flat.append(d_)
            n_sharp = len(staff_keys_sharp)
            n_flat = len(staff_keys_flat)
            if n_sharp == 0 and n_flat == 0:
                continue
            # 決定哪些音名受影響
            if n_sharp > 0 and n_flat == 0:
                affected_letters = SHARP_ORDER[:n_sharp]
                mark = "#"
            elif n_flat > 0 and n_sharp == 0:
                affected_letters = FLAT_ORDER[:n_flat]
                mark = "b"
            else:
                # 混合罕見，視為無調號跳過
                log(f"[KEY] staff {staff_idx}: 同時偵測到 sharp({n_sharp})+flat({n_flat})，跳過")
                continue
            log(f"[KEY] staff {staff_idx}: 偵測到 {n_sharp+n_flat} 個 key{'Sharp' if mark=='#' else 'Flat'}，"
                f"影響音名 {affected_letters}")
            # 套用到 staff 內所有同音名音符（除已被 accidental 修飾的）
            for e in staff_events:
                if e.get("type") != "note":
                    continue
                pitch = e.get("pitch")
                if not pitch or pitch in ("---", "R"):
                    continue
                # 已經有 accidental 套用 → 跳過（accidental 優先）
                if e.get("accidental_applied"):
                    continue
                # 已經有自帶記號 → 跳過（不重複套用）
                if pitch[0] in "#b":
                    continue
                letter = pitch[0]
                if letter not in affected_letters:
                    continue
                octave = pitch[1:]
                new_pitch = f"{mark}{letter}{octave}"
                e["pitch"] = new_pitch
                e["key_applied"] = mark + letter
                n_key_applied += 1
        if n_key_applied > 0:
            log(f"[POST] 調號傳遞: 套用 {n_key_applied} 個音符")

    # 印前 40 個
    log("\n========== 前 40 個事件 ==========")
    log(f"{'#':>3} {'stf':>3} {'x':>7} {'y':>7} {'pitch':>6} {'type':>13} "
        f"{'dot':>5} {'dur':>16} {'reason':>14}")
    for i, ev in enumerate(events[:40]):
        try:
            si = int(ev.get("staff_idx", 0))
            ex = float(ev.get("x", 0))
            ey = float(ev.get("y", 0))
        except Exception:
            si, ex, ey = 0, 0.0, 0.0
        log(f"{i:>3d} {si:>3d} {ex:>7.1f} {ey:>7.1f} "
            f"{str(ev.get('pitch','?')):>6} {str(ev.get('type','?')):>13} "
            f"{str(ev.get('has_dot',False)):>5} {str(ev.get('duration','?')):>16} "
            f"{str(ev.get('duration_reason','')):>14}")

    # 統計
    total_notes = sum(1 for e in events if e["type"] == "note")
    total_rests = sum(1 for e in events if e["type"] == "rest")
    total_beats = sum(duration_to_beats(e.get("duration")) for e in events)
    log(f"\n[SUCCESS] 偵測事件總數: {len(events)} （{total_notes} 個音符、{total_rests} 個休止符）")
    log(f"[SUCCESS] 總拍數: {total_beats:.2f}")

    # === 輸出 ===
    # 用輸入檔名作前綴（例如 5.png → 5_events.json），不再需要手動複製檔案
    img_stem = IMAGE_PATH.stem  # 例如 "5"
    debug_img = draw_debug(bgr, events, staff_lines_all, all_stems, all_beams)
    if debug_img is not None:
        debug_path = OUT_DIR / f"{img_stem}_debug.png"
        cv2.imwrite(str(debug_path), debug_img)
        log(f"[SUCCESS] Debug 圖: {debug_path}")

    log_path = OUT_DIR / f"{img_stem}_debug_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"[SUCCESS] Debug log: {log_path}")

    # 移除 box 欄位、序列化
    events_for_json = [{k: v for k, v in e.items() if k != "box"} for e in events]
    json_path = OUT_DIR / f"{img_stem}_events.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(events_for_json, f, ensure_ascii=False, indent=2)
    log(f"[SUCCESS] JSON: {json_path}")

    # === [診斷] 輸出原始幾何 dump（供離線重放 beam/flag 決策）===
    if ENABLE_GEOM_DUMP:
        geom_path = OUT_DIR / f"{img_stem}_geom_debug.json"
        with open(geom_path, "w", encoding="utf-8") as f:
            json.dump(_GEOM_DUMP, f, ensure_ascii=False, indent=2)
        log(f"[SUCCESS] Geom dump: {geom_path}")

    # === [診斷] 附點 CNN 二次驗證（純記錄，不改 events）===
    # 對每個 note 裁「音符+右側」貼片餵 dot_classifier，看它判有/無附點，
    # 跟 events 目前的 has_dot 並排，讓我們在真實譜上驗證這個分類器準不準。

    # MIDI（用檔名前綴）
    midi_path = OUT_DIR / f"{img_stem}_output.mid"
    events_to_midi(events, midi_path)
    log(f"[SUCCESS] MIDI: {midi_path}")

    # === 合併到 all_results.json（跨多張的彙整檔）===
    # 每跑一張就把這張的事件 append 到 all_results.json
    # 結構：{ "1": [events...], "2": [events...], ... }
    # 你跑完 7 張，這個檔案就是「所有結果合一」，直接丟給我就好
    summary_path = OUT_DIR / "all_results.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            try:
                all_results = json.load(f)
            except Exception:
                all_results = {}
    else:
        all_results = {}
    all_results[img_stem] = {
        "events": events_for_json,
        "total_events": len(events),
        "total_notes": total_notes,
        "total_rests": total_rests,
        "total_beats": round(total_beats, 2),
        # 以下全部診斷都併進這一個檔，你只要傳 all_results.json 一個即可：
        "geom_debug": list(_GEOM_DUMP),                # 每條 staff 的幾何（staff_lines_y/beams/stems/flags/noteheads）
        "debug_log": list(log_lines),                  # 完整 debug log（staff 偵測、d、pitch 推算等）
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    log(f"[SUCCESS] 已併入彙整檔: {summary_path}  (目前 {len(all_results)} 張)")


if __name__ == "__main__":
    main()

# ========================================================
# 行動端入口
# ========================================================
def recognize(image_path, out_dir, yolo_runner=None, models_dir=None):
    """
    入口:吃一張圖 + YOLO runner,跑完辨識,輸出 MIDI + all_results.json,回傳摘要。
      image_path : 樂譜圖路徑
      out_dir    : 輸出資料夾(放 .mid 與 all_results.json)
      yolo_runner: 物件,需有 detect(path, conf, iou, classes=None) -> [{box,cls_id,conf}]
      models_dir : 模型資料夾(含 pitch_cnn/.npz/.json、dot_cnn/.npz/.json)
    """
    import json as _json
    global IMAGE_PATH, OUT_DIR, MODELS_DIR
    if yolo_runner is not None:
        set_yolo_runner(yolo_runner)
    if models_dir is not None:
        MODELS_DIR = str(models_dir)
        import merge_notes_np as _mn
        _mn.MODELS_DIR = str(models_dir)
    IMAGE_PATH = Path(image_path)
    OUT_DIR = Path(out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    main()

    stem = Path(image_path).stem
    midi_path = OUT_DIR / f"{stem}_output.mid"
    summary = OUT_DIR / "all_results.json"
    info = {}
    if summary.exists():
        try:
            info = _json.load(open(summary, encoding="utf-8")).get(stem, {})
        except Exception:
            info = {}
    return {
        "stem": stem,
        "midi_path": str(midi_path),
        "midi_exists": midi_path.exists(),
        "total_events": info.get("total_events"),
        "total_notes": info.get("total_notes"),
        "events": info.get("events", []),
    }