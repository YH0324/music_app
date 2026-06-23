import csv
import json
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np



# 由 App 入口設定為模型資料夾(含 pitch_cnn.npz / pitch_cnn.json)
MODELS_DIR = None


# =========================
# 先改這個：選哪一個 staff 測試
# =========================
TARGET_STAFF_ROW_INDEX = 0


# =========================
# 路徑設定
# =========================
BASE_PATH = Path(r"C:\music\data\ds2")
IMAGE_PATH = BASE_PATH / "images"
META_DIR = Path(r"C:\music\data\metadata")

INPUT_CSV = META_DIR / "usable_monophonic_treble.csv"

PITCH_DATA_ROOT = Path(r"C:\music\data\pitch")
PITCH_MODEL_DIR = Path(r"C:\music\outputs\pitch_classifier")
PITCH_MODEL_PATH = PITCH_MODEL_DIR / "best_model.pt"

OUT_DIR = Path(r"C:\music\outputs\merge_notes")
# OUT_DIR.mkdir 已移除(手機端不需要)

OUT_NOTE_JSON = OUT_DIR / "note_objects.json"
OUT_EVENTS_JSON = OUT_DIR / "events.json"
OUT_PREVIEW = OUT_DIR / "note_objects_preview.png"


# =========================
# 類別定義
# =========================
NOTEHEAD_CLASSES = {
    "noteheadBlackOnLine",
    "noteheadBlackInSpace",
    "noteheadHalfOnLine",
    "noteheadHalfInSpace",
    "noteheadWholeOnLine",
    "noteheadWholeInSpace",
}

ACCIDENTAL_CLASSES = {
    "accidentalSharp",
    "accidentalFlat",
    "accidentalNatural",
}

REST_CLASSES = {
    "restWhole",
    "restHalf",
    "restQuarter",
    "rest8th",
    "rest16th",
}

# [終極修復] 擴充音域字典，避免極低音或極高音被丟棄
HIGH_PITCHES = {
    "A6", "B6", "C7", "D7", "E7", "F7", "G7",
    "A7", "B7", "C8", "D8", "E8", "F8", "G8"
}

STEP_TO_PITCH = {
    -9: "C3", -8: "D3", -7: "E3", -6: "F3", -5: "G3",
    -4: "A3", -3: "B3", -2: "C4", -1: "D4", 0: "E4", 1: "F4", 2: "G4",
    3: "A4", 4: "B4", 5: "C5", 6: "D5", 7: "E5", 8: "F5", 9: "G5",
    10: "A5", 11: "B5", 12: "C6", 13: "D6", 14: "E6", 15: "F6", 16: "G6",
    17: "A6", 18: "B6", 19: "C7", 20: "D7", 21: "E7", 22: "F7", 23: "G7",
    24: "A7", 25: "B7", 26: "C8", 27: "D8", 28: "E8", 29: "F8", 30: "G8",
}


# =========================
# Pitch model
# =========================
# PitchCNN 已改用 cnn_numpy.NumpyCNN(純 numpy 前向)


# =========================
# 基本工具
# =========================
def normalize_id(value):
    if value is None: return None
    if isinstance(value, list): return None
    if isinstance(value, (int, float)):
        if float(value).is_integer(): return str(int(value))
        return str(value)
    if isinstance(value, str):
        v = value.strip()
        try:
            f = float(v)
            if f.is_integer(): return str(int(f))
        except Exception: pass
        return v
    return str(value)

def load_csv_rows(csv_path: Path):
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader: rows.append(row)
    return rows

def load_json(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_categories_map(categories_data):
    result = {}
    if isinstance(categories_data, list):
        for cat in categories_data:
            if not isinstance(cat, dict): continue
            cat_id = normalize_id(cat.get("id"))
            cat_name = cat.get("name")
            if cat_id is not None and cat_name is not None:
                result[cat_id] = str(cat_name)
    elif isinstance(categories_data, dict):
        for k, v in categories_data.items():
            cat_id = normalize_id(k)
            if isinstance(v, dict): cat_name = v.get("name", f"class_{cat_id}")
            else: cat_name = str(v)
            if cat_id is not None: result[cat_id] = cat_name
    return result

def build_annotations_list(data):
    anns = data.get("annotations")
    if isinstance(anns, list): return anns
    if isinstance(anns, dict): return list(anns.values())
    raise TypeError(f"Unsupported annotations format: {type(anns)}")

def extract_image_id(ann):
    return normalize_id(ann.get("img_id", ann.get("image_id")))

def extract_category_names(ann, categories):
    raw = ann.get("cat_id", ann.get("category_id"))
    if raw is None: return []
    if not isinstance(raw, list): raw = [raw]
    names = []
    for r in raw:
        cat_id = normalize_id(r)
        if cat_id is not None and cat_id in categories:
            names.append(categories[cat_id])
    return names

def extract_a_bbox(ann):
    box = ann.get("a_bbox")
    if not isinstance(box, (list, tuple)) or len(box) != 4: return None
    try: x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    except Exception: return None
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2

def bbox_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0

def bbox_width(box):
    x1, _, x2, _ = box
    return max(1.0, float(x2 - x1))

def bbox_height(box):
    _, y1, _, y2 = box
    return max(1.0, float(y2 - y1))

def point_in_box(x, y, box):
    left, top, right, bottom = box
    return (left <= x <= right) and (top <= y <= bottom)

def count_noteheads_in_crop(crop_box, noteheads):
    cnt = 0
    for note in noteheads:
        cx, cy = bbox_center(note["box"])
        if point_in_box(cx, cy, crop_box): cnt += 1
    return cnt

def interval_gap(a1, a2, b1, b2):
    if a2 < b1: return b1 - a2
    if b2 < a1: return a1 - b2
    return 0.0

def merge_notehead_type(raw_cls):
    if raw_cls.startswith("noteheadBlack"): return "noteheadBlack"
    if raw_cls.startswith("noteheadHalf"): return "noteheadHalf"
    if raw_cls.startswith("noteheadWhole"): return "noteheadWhole"
    return raw_cls

def duration_abbrev(duration):
    mapping = {"whole": "w", "half": "h", "quarter": "q", "8th": "8", "16th": "16", None: "?"}
    if duration and duration.startswith("dotted-"):
        base = duration.split("-")[1]
        return f"{mapping.get(base, base)}."
    return mapping.get(duration, str(duration))


# =========================
# 符號分類
# =========================
def parse_symbol_type(cat_names):
    note_name = None
    for n in cat_names:
        if n in NOTEHEAD_CLASSES:
            note_name = n
            break
    if note_name is not None: return "notehead", note_name
    for n in cat_names:
        if n == "stem" or n.startswith("stem"): return "stem", "stem"
    for n in cat_names:
        if "flag16th" in n: return "flag", "flag16th"
        if "flag8th" in n: return "flag", "flag8th"
    for n in cat_names:
        if "beam" in n: return "beam", "beam"
    for n in cat_names:
        if n in ACCIDENTAL_CLASSES: return "accidental", n
    for n in cat_names:
        if n in REST_CLASSES: return "rest", n
    for n in cat_names:
        if n == "augmentationDot": return "dot", n
    for n in cat_names:
        if n in {"keySharp", "keyFlat"}: return "key_signature", n
    return None, None


# =========================
# staff line 偵測
# =========================
def cluster_consecutive_rows(rows):
    if len(rows) == 0: return []
    clusters = [[rows[0]]]
    for r in rows[1:]:
        if r == clusters[-1][-1] + 1: clusters[-1].append(r)
        else: clusters.append([r])
    return [int(round(np.mean(c))) for c in clusters]

def choose_best_five_lines(line_centers):
    if len(line_centers) < 5: return None
    line_centers = sorted(line_centers)
    best = None
    best_score = 1e18
    for i in range(len(line_centers) - 4):
        cand = line_centers[i:i + 5]
        diffs = np.diff(cand)
        if np.min(diffs) <= 0: continue
        mean_d = float(np.mean(diffs))
        std_d = float(np.std(diffs))
        score = std_d + 0.05 * mean_d
        if score < best_score:
            best_score = score
            best = cand
    return best

def detect_staff_lines(gray, staff_box):
    x1, y1, x2, y2 = map(int, staff_box)
    h, w = gray.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

    # [終極修復] 數學幾何無敵保底，用 bounding box 等分算出完美的 5 條線
    fallback_lines = [int(round(y1 + i * (y2 - y1) / 4.0)) for i in range(5)]

    if x2 <= x1 or y2 <= y1: return fallback_lines
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0: return fallback_lines

    _, bw = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    proj = np.sum(bw > 0, axis=1).astype(np.float32)
    width = roi.shape[1]

    threshold = max(0.4 * width, 0.5 * float(np.max(proj)))

    candidate_rows = np.where(proj >= threshold)[0]
    if len(candidate_rows) == 0: return fallback_lines

    centers = cluster_consecutive_rows(candidate_rows)
    if len(centers) < 5: return fallback_lines

    best_five = choose_best_five_lines(centers)
    if best_five is None: return fallback_lines

    return [int(y1 + c) for c in best_five]


# =========================
# staff 內符號收集
# =========================
def collect_noteheads_for_staff(noteheads, staff_box):
    sx1, sy1, sx2, sy2 = staff_box
    staff_h = sy2 - sy1

    top_bound = sy1 - 2.0 * staff_h
    bottom_bound = sy2 + 1.2 * staff_h

    selected = []
    for note in noteheads:
        cx, cy = bbox_center(note["box"])
        if cx < sx1 or cx > sx2: continue
        if cy < top_bound or cy > bottom_bound: continue
        selected.append(note)
    return selected

def collect_symbols_for_staff(symbols, staff_box):
    sx1, sy1, sx2, sy2 = staff_box
    staff_h = sy2 - sy1

    top_bound = sy1 - 4.0 * staff_h
    bottom_bound = sy2 + 4.0 * staff_h

    selected = []
    for sym in symbols:
        cx, cy = bbox_center(sym["box"])
        if cx < sx1 - 20 or cx > sx2 + 20: continue
        if cy < top_bound or cy > bottom_bound: continue
        selected.append(sym)

    return selected


# =========================
# pitch 規則與 patch
# =========================
def infer_step_and_pitch_from_lines(note_box, notehead_class, staff_lines_y):
    staff_lines_y = sorted(staff_lines_y)
    diffs = np.diff(staff_lines_y)
    if len(diffs) != 4: return None, None
    d = float(np.median(diffs))
    if d <= 0: return None, None
    _, yc = bbox_center(note_box)
    bottom_y = float(staff_lines_y[-1])
    raw_step = (bottom_y - yc) / (d / 2.0)
    step = int(round(raw_step))
    if "OnLine" in notehead_class and (step % 2 != 0):
        candidates = [step - 1, step + 1]
        even_candidates = [c for c in candidates if c % 2 == 0]
        if even_candidates: step = min(even_candidates, key=lambda c: abs(c - raw_step))
    if "InSpace" in notehead_class and (step % 2 == 0):
        candidates = [step - 1, step + 1]
        odd_candidates = [c for c in candidates if c % 2 != 0]
        if odd_candidates: step = min(odd_candidates, key=lambda c: abs(c - raw_step))
    pitch = STEP_TO_PITCH.get(step)
    return step, pitch

def build_crop_box(note_box, staff_lines_y, pitch, img_w, img_h):
    x1, y1, x2, y2 = note_box
    note_w = x2 - x1
    note_h = y2 - y1
    note_cx, note_cy = bbox_center(note_box)
    staff_center_y = float(np.mean(staff_lines_y))
    d = float(np.median(np.diff(sorted(staff_lines_y))))
    staff_h = 4.0 * d
    cx = note_cx
    cy = 0.8 * note_cy + 0.2 * staff_center_y
    patch_w = int(max(2.8 * note_w, 0.75 * staff_h))
    patch_h = int(max(3.6 * staff_h, 7.5 * note_h))
    if pitch in HIGH_PITCHES: patch_h = int(round(patch_h * 1.25))
    left = max(0, int(round(cx - patch_w / 2)))
    right = min(img_w, int(round(cx + patch_w / 2)))
    top = max(0, int(round(cy - patch_h / 2)))
    bottom = min(img_h, int(round(cy + patch_h / 2)))
    if right <= left or bottom <= top: return None
    return left, top, right, bottom

def crop_patch(img, crop_box):
    left, top, right, bottom = crop_box
    patch = img[top:bottom, left:right]
    if patch is None or patch.size == 0: return None
    return patch


# =========================
# duration 推理
# =========================
def find_stem_for_note(note_box, stems):
    x1, y1, x2, y2 = note_box
    note_cx, _ = bbox_center(note_box)
    note_w = bbox_width(note_box)

    best = None
    best_score = 1e18

    for stem in stems:
        sx1, sy1, sx2, sy2 = stem["box"]
        stem_cx, _ = bbox_center(stem["box"])

        h_gap = interval_gap(x1, x2, sx1, sx2)
        v_gap = interval_gap(y1, y2, sy1, sy2)
        center_dx = abs(stem_cx - note_cx)

        if h_gap > 0.5 * note_w:
            continue
        if center_dx > 2.5 * note_w:
            continue
        if v_gap > 10:
            continue

        score = 3.0 * h_gap + 1.0 * v_gap + 0.5 * center_dx
        if score < best_score:
            best_score = score
            best = stem

    return best


def count_beams_for_stem(stem_box, beams):
    sx1, sy1, sx2, sy2 = stem_box
    count = 0
    for beam in beams:
        bx1, by1, bx2, by2 = beam["box"]
        x_gap = interval_gap(sx1, sx2, bx1, bx2)

        if x_gap > 5:
            continue

        y_overlap = max(0, min(sy2, by2) - max(sy1, by1))
        near_top = abs(by1 - sy1)
        near_bottom = abs(by2 - sy2)

        if y_overlap == 0 and min(near_top, near_bottom) > 15:
            continue

        count += 1
    return count


def flag_level_for_stem(stem_box, flags):
    sx1, sy1, sx2, sy2 = stem_box
    stem_len = max(1.0, bbox_height(stem_box))
    best_level = 0

    for flag in flags:
        fx1, fy1, fx2, fy2 = flag["box"]
        _, fcy = bbox_center(flag["box"])

        x_gap = interval_gap(sx1, sx2, fx1, fx2)

        if x_gap > 5:
            continue

        near_top = abs(fcy - sy1)
        near_bottom = abs(fcy - sy2)
        if min(near_top, near_bottom) > 0.6 * stem_len:
            continue

        level = flag.get("level", 0)
        best_level = max(best_level, level)

    return best_level


def infer_duration(notehead_type, stem_obj, beam_count, flag_level):
    if notehead_type == "noteheadWhole": return "whole"
    if notehead_type == "noteheadHalf": return "half"
    if notehead_type == "noteheadBlack":
        level = max(beam_count, flag_level)

        if level >= 2: return "16th"
        if level == 1: return "8th"
        return "quarter"

    return None


# =========================
# 附點與調號推理
# =========================
def find_dot_for_note(note_box, dots):
    x1, y1, x2, y2 = note_box
    note_cx, note_cy = bbox_center(note_box)
    note_w = bbox_width(note_box)
    note_h = bbox_height(note_box)

    best_dot = None
    best_score = 1e18

    for dot in dots:
        dot_cx, dot_cy = bbox_center(dot["box"])
        dx1, dy1, dx2, dy2 = dot["box"]

        if dot_cx <= note_cx: continue

        x_gap = interval_gap(x1, x2, dx1, dx2)
        y_dist = abs(dot_cy - note_cy)

        if x_gap < 5.0 * note_w and y_dist < 3.0 * note_h:
            score = x_gap + y_dist
            if score < best_score:
                best_score = score
                best_dot = dot

    return best_dot


def build_key_signature_map(staff_key_sigs, staff_lines_y):
    key_map = {}
    staff_lines_y = sorted(staff_lines_y)
    diffs = np.diff(staff_lines_y)
    if len(diffs) != 4: return key_map
    d = float(np.median(diffs))
    bottom_y = float(staff_lines_y[-1])

    for ks in staff_key_sigs:
        box = ks["box"]
        cls_name = ks["class_name"]
        _, yc = bbox_center(box)
        raw_step = (bottom_y - yc) / (d / 2.0)
        step = int(round(raw_step))
        pitch = STEP_TO_PITCH.get(step)
        if pitch:
            note_letter = pitch[0]
            key_map[note_letter] = "accidentalSharp" if "Sharp" in cls_name else "accidentalFlat"
    return key_map


# =========================
# accidental 推理
# =========================
def find_accidental_for_note(note_box, accidentals):
    note_cx, note_cy = bbox_center(note_box)
    note_h = bbox_height(note_box)
    best = None
    best_score = 1e18

    for acc in accidentals:
        acc_cx, acc_cy = bbox_center(acc["box"])
        if acc_cx >= note_cx: continue
        dx = note_cx - acc_cx
        dy = abs(note_cy - acc_cy)

        if dx > 80: continue
        if dy > 1.6 * note_h: continue

        score = 1.2 * dx + 2.0 * dy
        if score < best_score:
            best_score = score
            best = acc

    return best


def apply_accidental_to_pitch(pitch_natural, accidental_name):
    if pitch_natural is None: return None
    note_part = pitch_natural[0]
    octave_part = pitch_natural[1:]
    if accidental_name == "accidentalSharp": return f"{note_part}#{octave_part}"
    if accidental_name == "accidentalFlat": return f"{note_part}b{octave_part}"
    if accidental_name == "accidentalNatural": return f"{note_part}{octave_part}"
    return pitch_natural


# =========================
# rest 推理
# =========================
def rest_class_to_duration(rest_class_name):
    mapping = {
        "restWhole": "whole", "restHalf": "half", "restQuarter": "quarter",
        "rest8th": "8th", "rest16th": "16th",
    }
    return mapping.get(rest_class_name)


# =========================
# 預覽圖
# =========================
def draw_preview(img_bgr, note_objects, rest_events, staff_box):
    vis = img_bgr.copy()
    for note in note_objects:
        x1, y1, x2, y2 = note["notehead_bbox"]
        label = f"{note['final_pitch']} {duration_abbrev(note['duration'])}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(vis, label, (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv2.LINE_AA)

    for rest in rest_events:
        x1, y1, x2, y2 = rest["bbox"]
        label = f"R {duration_abbrev(rest['duration'])}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 150, 0), 2)
        cv2.putText(vis, label, (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 0), 1, cv2.LINE_AA)

    sx1, sy1, sx2, sy2 = map(int, staff_box)
    h, w = vis.shape[:2]
    left, right = max(0, sx1 - 40), min(w, sx2 + 40)
    top, bottom = max(0, sy1 - 60), min(h, sy2 + 50)
    return vis[top:bottom, left:right]


# =========================
# 主程式
# =========================