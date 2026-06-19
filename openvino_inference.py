import math
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pyclipper

if not os.access(Path.home(), os.W_OK):
    os.environ["HOME"] = "/tmp"

from openvino import Core

from ocr_config import (
    BOX_THRESH,
    DEBUG_RAW_REC,
    DEFAULT_REC_HEIGHT,
    DET_LIMIT_SIDE_LEN,
    DET_THRESH,
    IMAGE_EXTENSIONS,
    MAX_DYNAMIC_REC_WIDTH,
    OPENVINO_DET_MODEL,
    OPENVINO_DEVICE,
    OPENVINO_REC_MODEL,
    REC_DICT,
    SAVE_CROPS,
    SAVE_DIR,
    SAVE_OUTPUTS,
    TEST_IMAGE_DIR,
    UNCLIP_RATIO,
)

# =========================================================
# CONFIG
# =========================================================

DET_MODEL = OPENVINO_DET_MODEL
REC_MODEL = OPENVINO_REC_MODEL


# =========================================================
# FONT
# =========================================================

FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKkr-Regular.otf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def load_font(size=18):
    from PIL import ImageFont

    for p in FONT_CANDIDATES:
        p = Path(p)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)

    print("WARNING: Korean font not found. Using default font.")
    return ImageFont.load_default()


def contains_korean(text):
    return any("\uac00" <= ch <= "\ud7a3" for ch in text)


# =========================================================
# DICTIONARY
# =========================================================

def load_dict(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dictionary not found: {path}")

    chars = ["blank"]

    with open(path, "r", encoding="utf-8") as f:
        chars += [line.rstrip("\n") for line in f if line.rstrip("\n")]

    if chars[-1] != " ":
        chars.append(" ")

    if SAVE_OUTPUTS:
        print(f"Dictionary classes: {len(chars)}")
    return chars


# =========================================================
# PREPROCESSING
# =========================================================

def normalize_det(img):
    img = img.astype("float32") / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img = (img - mean) / std
    img = img.transpose(2, 0, 1)

    return img[None].astype("float32")


def normalize_rec(img):
    img = img.astype("float32") / 255.0
    img = (img - 0.5) / 0.5
    img = img.transpose(2, 0, 1)

    return img[None].astype("float32")


def resize_det(img, limit_side_len=DET_LIMIT_SIDE_LEN):
    h, w = img.shape[:2]

    ratio = min(limit_side_len / max(h, w), 1.0)

    resize_h = int(round(h * ratio / 32) * 32)
    resize_w = int(round(w * ratio / 32) * 32)

    resize_h = max(32, resize_h)
    resize_w = max(32, resize_w)

    return cv2.resize(img, (resize_w, resize_h))


# =========================================================
# DETECTION POSTPROCESSING
# =========================================================

def order_points(pts):
    pts = np.array(pts, dtype=np.float32)

    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]

    return rect


def get_mini_box(contour):
    box = cv2.boxPoints(cv2.minAreaRect(contour))
    box = order_points(box)

    side1 = np.linalg.norm(box[0] - box[1])
    side2 = np.linalg.norm(box[1] - box[2])

    return box, min(side1, side2)


def box_score(prob, box):
    h, w = prob.shape[:2]

    xmin = max(0, int(np.floor(box[:, 0].min())))
    xmax = min(w - 1, int(np.ceil(box[:, 0].max())))
    ymin = max(0, int(np.floor(box[:, 1].min())))
    ymax = min(h - 1, int(np.ceil(box[:, 1].max())))

    if xmax <= xmin or ymax <= ymin:
        return 0.0

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)

    box2 = box.copy()
    box2[:, 0] -= xmin
    box2[:, 1] -= ymin

    cv2.fillPoly(mask, [box2.astype(np.int32)], 1)

    return cv2.mean(prob[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def unclip(box, ratio=UNCLIP_RATIO):
    area = cv2.contourArea(box.astype(np.float32))
    length = cv2.arcLength(box.astype(np.float32), True)

    if length <= 0:
        return None

    distance = area * ratio / length

    offset = pyclipper.PyclipperOffset()
    offset.AddPath(
        box.astype(np.int32).tolist(),
        pyclipper.JT_ROUND,
        pyclipper.ET_CLOSEDPOLYGON,
    )

    expanded = offset.Execute(distance)

    if not expanded:
        return None

    return np.array(expanded[0], dtype=np.float32)


def detect(det_model, rgb, debug_bitmap_path=None):
    resized = resize_det(rgb)
    inp = normalize_det(resized)

    pred = det_model([inp])[det_model.outputs[0]]
    prob = pred[0, 0]

    bitmap = (prob > DET_THRESH).astype(np.uint8)
    if debug_bitmap_path is not None:
        cv2.imwrite(str(debug_bitmap_path), bitmap * 255)

    contours, _ = cv2.findContours(
        bitmap * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    src_h, src_w = rgb.shape[:2]
    prob_h, prob_w = prob.shape[:2]

    boxes = []

    for contour in contours:
        box, short_side = get_mini_box(contour)

        if short_side < 3:
            continue

        score = box_score(prob, box)

        if score < BOX_THRESH:
            continue

        expanded = unclip(box)

        if expanded is None:
            continue

        box, short_side = get_mini_box(expanded.reshape(-1, 1, 2))

        if short_side < 3:
            continue

        box[:, 0] = np.clip(np.round(box[:, 0] / prob_w * src_w), 0, src_w - 1)
        box[:, 1] = np.clip(np.round(box[:, 1] / prob_h * src_h), 0, src_h - 1)

        box = order_points(box).astype(np.int32)

        boxes.append(
            {
                "box": box,
                "det_score": float(score),
            }
        )

    boxes = sorted(
        boxes,
        key=lambda x: (
            x["box"][:, 1].min(),
            x["box"][:, 0].min(),
        ),
    )

    return boxes


# =========================================================
# CROP
# =========================================================

def crop_text(rgb, box):
    box = order_points(box)

    crop_w = int(
        max(
            np.linalg.norm(box[0] - box[1]),
            np.linalg.norm(box[2] - box[3]),
        )
    )

    crop_h = int(
        max(
            np.linalg.norm(box[0] - box[3]),
            np.linalg.norm(box[1] - box[2]),
        )
    )

    crop_w = max(1, crop_w)
    crop_h = max(1, crop_h)

    dst = np.array(
        [
            [0, 0],
            [crop_w, 0],
            [crop_w, crop_h],
            [0, crop_h],
        ],
        dtype=np.float32,
    )

    m = cv2.getPerspectiveTransform(box.astype(np.float32), dst)

    crop = cv2.warpPerspective(
        rgb,
        m,
        (crop_w, crop_h),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )

    if crop.shape[0] / max(1, crop.shape[1]) >= 1.5:
        crop = np.rot90(crop)

    return crop


# =========================================================
# RECOGNITION
# =========================================================

def get_rec_shape(rec_model):
    pshape = rec_model.input(0).partial_shape

    h_dim = pshape[2]
    w_dim = pshape[3]

    rec_h = h_dim.get_length() if h_dim.is_static else DEFAULT_REC_HEIGHT
    rec_w = w_dim.get_length() if w_dim.is_static else None

    return int(rec_h), rec_w


def resize_rec(crop, rec_h=DEFAULT_REC_HEIGHT, rec_w=None, max_dynamic_w=MAX_DYNAMIC_REC_WIDTH):
    h, w = crop.shape[:2]

    if h <= 0 or w <= 0:
        return None

    ratio = w / float(h)
    new_w = int(math.ceil(rec_h * ratio))

    if rec_w is None:
        final_w = min(max_dynamic_w, max(32, new_w))
    else:
        final_w = int(rec_w)

    new_w = min(new_w, final_w)
    new_w = max(16, new_w)

    resized = cv2.resize(
        crop,
        (new_w, rec_h),
        interpolation=cv2.INTER_LINEAR,
    )

    padded = np.full(
        (rec_h, final_w, 3),
        255,
        dtype=np.uint8,
    )

    padded[:, :new_w, :] = resized

    return padded


def decode_ctc(pred, chars):
    pred = np.asarray(pred)

    if pred.ndim == 3:
        pred = pred[0]

    if pred.shape[0] == len(chars):
        pred = pred.T

    if SAVE_OUTPUTS and pred.shape[-1] != len(chars):
        print(f"WARNING: dict classes={len(chars)}, model classes={pred.shape[-1]}")

    idxs = np.argmax(pred, axis=1)
    vals = np.max(pred, axis=1)

    text = []
    scores = []
    last = None

    for idx, score in zip(idxs, vals):
        idx = int(idx)

        if idx != 0 and idx != last and idx < len(chars):
            text.append(chars[idx])
            scores.append(float(score))

        last = idx

    rec_score = float(np.mean(scores)) if scores else 0.0

    return "".join(text), rec_score


def recognize(rec_model, crop, chars):
    rec_h, rec_w = get_rec_shape(rec_model)

    rec_img = resize_rec(
        crop,
        rec_h=rec_h,
        rec_w=rec_w,
        max_dynamic_w=MAX_DYNAMIC_REC_WIDTH,
    )

    if rec_img is None:
        return "", 0.0

    inp = normalize_rec(rec_img)
    pred = rec_model([inp])[rec_model.outputs[0]]

    return decode_ctc(pred, chars)


# =========================================================
# VISUALIZATION FORMAT
# =========================================================

PALETTE = [
    (202, 236, 121),   # light green
    (125, 230, 196),   # mint
    (185, 161, 226),   # purple
    (134, 203, 255),   # blue
    (246, 180, 102),   # orange
    (152, 245, 190),   # green
    (210, 225, 230),   # grey-blue
    (255, 225, 120),   # yellow
]


def polygon_bbox(box):
    box = np.array(box, dtype=np.int32)

    x1 = int(box[:, 0].min())
    y1 = int(box[:, 1].min())
    x2 = int(box[:, 0].max())
    y2 = int(box[:, 1].max())

    return x1, y1, x2, y2


def draw_filled_rect(draw, rect, color, alpha=120, outline=None, width=1):
    x1, y1, x2, y2 = rect

    fill = color + (alpha,)
    outline_color = outline + (255,) if outline else color + (255,)

    draw.rectangle(
        [x1, y1, x2, y2],
        fill=fill,
        outline=outline_color,
        width=width,
    )


def draw_side_by_side_format(image_path, results, save_path):
    from PIL import Image, ImageDraw

    """
    Creates the same style as the PaddleOCR-style visualization:
    left: original image with colored line regions
    right: OCR text reconstructed with same y-positions and same colors
    """

    bgr = cv2.imread(str(image_path))

    if bgr is None:
        raise FileNotFoundError(image_path)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    left = Image.fromarray(rgb).convert("RGBA")

    img_w, img_h = left.size

    right_w = img_w
    gap = 30

    canvas = Image.new(
        "RGBA",
        (img_w + gap + right_w, img_h),
        (255, 255, 255, 255),
    )

    canvas.paste(left, (0, 0))

    overlay_left = Image.new("RGBA", left.size, (255, 255, 255, 0))
    draw_left = ImageDraw.Draw(overlay_left)

    draw_right = ImageDraw.Draw(canvas)

    font = load_font(18)

    # Right panel origin
    rx = img_w + gap

    # Draw vertical separator
    draw_right.line(
        [(img_w + gap // 2, 0), (img_w + gap // 2, img_h)],
        fill=(220, 220, 220, 255),
        width=1,
    )

    for idx, r in enumerate(results, start=1):
        box = np.array(r["box"], dtype=np.int32)
        x1, y1, x2, y2 = polygon_bbox(box)

        color = PALETTE[(idx - 1) % len(PALETTE)]

        # Slight padding, like PaddleOCR visualization
        pad_x = 2
        pad_y = 2

        left_rect = (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(img_w - 1, x2 + pad_x),
            min(img_h - 1, y2 + pad_y),
        )

        # LEFT: colored background over original text
        draw_filled_rect(
            draw_left,
            left_rect,
            color=color,
            alpha=105,
            outline=color,
            width=1,
        )

        # RIGHT: same y-position and approximate same box height
        text = r["text"]
        if not text.strip():
            continue

        right_x1 = rx + max(0, x1)
        right_y1 = max(0, y1 - pad_y)
        use_crop_for_text = contains_korean(text)

        if use_crop_for_text:
            text_w = max(1, x2 - x1 + 1)
            text_h = max(1, y2 - y1 + 1)
        else:
            text_bbox = draw_right.textbbox((right_x1 + 3, right_y1), text, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]

        right_x2 = min(rx + right_w - 1, right_x1 + text_w + 8)
        right_y2 = min(img_h - 1, max(y2 + pad_y, right_y1 + text_h + 5))

        right_rect = (
            right_x1,
            right_y1,
            right_x2,
            right_y2,
        )

        draw_filled_rect(
            draw_right,
            right_rect,
            color=color,
            alpha=70,
            outline=color,
            width=1,
        )

        if use_crop_for_text:
            crop = rgb[max(0, y1):min(img_h, y2 + 1), max(0, x1):min(img_w, x2 + 1)]
            if crop.size > 0:
                crop_img = Image.fromarray(crop).convert("RGB")
                target_w = max(1, right_x2 - right_x1 - 6)
                target_h = max(1, right_y2 - right_y1 - 4)
                crop_img = crop_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                canvas.paste(crop_img, (right_x1 + 3, right_y1 + 2))
            else:
                draw_right.text(
                    (right_x1 + 4, right_y1),
                    text,
                    font=font,
                    fill=(0, 0, 0, 255),
                )
        else:
            draw_right.text(
                (right_x1 + 4, right_y1),
                text,
                font=font,
                fill=(0, 0, 0, 255),
            )

    # Composite left overlay
    left_colored = Image.alpha_composite(left, overlay_left)
    canvas.paste(left_colored, (0, 0))

    final_rgb = canvas.convert("RGB")
    final_rgb.save(save_path)


# =========================================================
# MAIN
# =========================================================

def get_image_paths(image_dir):
    image_dir = Path(image_dir)

    if image_dir.is_file() and image_dir.suffix.lower() in IMAGE_EXTENSIONS:
        return [image_dir]

    if not image_dir.exists():
        raise FileNotFoundError(f"Image path not found: {image_dir}")

    # Process every supported image in stable filename order.
    return sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def process_image(image_path, det_model, rec_model, chars):
    image_path = Path(image_path)
    image_stem = image_path.stem

    vis_path = SAVE_DIR / f"{image_stem}_openvino.png"
    json_path = SAVE_DIR / f"{image_stem}_openvino.json"
    bitmap_path = SAVE_DIR / f"{image_stem}_openvino_debug_bitmap.png"
    crop_dir = SAVE_DIR / f"{image_stem}_openvino_debug_crops"

    if SAVE_OUTPUTS:
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        if SAVE_CROPS:
            crop_dir.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(str(image_path))

    if bgr is None:
        raise FileNotFoundError(image_path)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    det_items = detect(det_model, rgb, debug_bitmap_path=bitmap_path if SAVE_OUTPUTS else None)

    if SAVE_OUTPUTS:
        print(f"\nImage: {image_path}")
        print(f"Detected boxes: {len(det_items)}")

    results = []

    for idx, item in enumerate(det_items, start=1):
        box = item["box"]
        det_score = item["det_score"]

        crop = crop_text(rgb, box)

        if SAVE_OUTPUTS and SAVE_CROPS:
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(crop_dir / f"crop_{idx:03d}.png"), crop_bgr)

        text, rec_score = recognize(rec_model, crop, chars)

        if SAVE_OUTPUTS and DEBUG_RAW_REC:
            print(
                f"{idx:02d} RAW REC: {repr(text)} | "
                f"rec={rec_score:.4f} | det={det_score:.4f}"
            )

        if text.strip():
            results.append(
                {
                    "text": text,
                    "rec_score": float(rec_score),
                    "det_score": float(det_score),
                    "box": box.tolist(),
                }
            )

    if SAVE_OUTPUTS:
        print("OCR RESULTS")
        print("=" * 80)

        for r in results:
            print(
                f'{r["text"]} | '
                f'rec={r["rec_score"]:.4f} | '
                f'det={r["det_score"]:.4f}'
            )

    image_output = {
        "image": str(image_path),
        "det_model": DET_MODEL,
        "rec_model": REC_MODEL,
        "results": results,
    }

    if SAVE_OUTPUTS:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(image_output, f, ensure_ascii=False, indent=2)

        draw_side_by_side_format(
            image_path=image_path,
            results=results,
            save_path=vis_path,
        )

        print(f"Saved visualization: {vis_path}")
        print(f"Saved JSON: {json_path}")
        if SAVE_CROPS:
            print(f"Saved crops: {crop_dir}")

    return image_output


def main():
    core = Core()

    if SAVE_OUTPUTS:
        print("Loading OpenVINO models...")

    det_model = core.compile_model(DET_MODEL, OPENVINO_DEVICE)
    rec_model = core.compile_model(REC_MODEL, OPENVINO_DEVICE)
    chars = load_dict(REC_DICT)

    if SAVE_OUTPUTS:
        print("Detection input:", det_model.input(0).partial_shape)
        print("Recognition input:", rec_model.input(0).partial_shape)
        print("Recognition output:", rec_model.output(0).partial_shape)

    image_paths = get_image_paths(TEST_IMAGE_DIR)
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {TEST_IMAGE_DIR}")

    all_outputs = []

    for image_path in image_paths:
        all_outputs.append(process_image(image_path, det_model, rec_model, chars))

    terminal_json = {
        "engine": "openvino",
        "image_dir": str(TEST_IMAGE_DIR),
        "save_outputs": bool(SAVE_OUTPUTS),
        "outputs": all_outputs,
    }

    if SAVE_OUTPUTS:
        print("\nJSON OUTPUT")
    print(json.dumps(terminal_json, ensure_ascii=False, indent=2))
    if SAVE_OUTPUTS:
        print("\nDone.")


if __name__ == "__main__":
    main()
