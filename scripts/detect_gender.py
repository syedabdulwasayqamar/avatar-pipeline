import os
import cv2
import numpy as np
import logging
from deepface import DeepFace

logging.getLogger("deepface").setLevel(logging.ERROR)


def sample_skin_tone_direct(crop):
    """
    Sample skin tone directly from pixels — no race classification.
    Focuses on center strip of the crop where skin is most likely.
    """
    if crop is None or crop.size == 0:
        return (220, 185, 160)

    h, w = crop.shape[:2]

    # Sample from center vertical strip — avoids background edges
    x1, x2 = int(w * 0.3), int(w * 0.7)
    y1, y2 = int(h * 0.1), int(h * 0.6)
    region  = crop[y1:y2, x1:x2]

    if region.size == 0:
        return (220, 185, 160)

    # Convert BGR to HSV to filter for skin-colored pixels
    hsv    = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    # Skin hue range: 0-25 degrees, reasonable saturation and value
    mask   = cv2.inRange(hsv,
                         np.array([0,  20,  60], dtype=np.uint8),
                         np.array([25, 200, 255], dtype=np.uint8))

    skin_pixels = region[mask > 0]
    if len(skin_pixels) < 50:
        # Fallback: just use center region average
        avg = region.mean(axis=(0, 1))
    else:
        avg = skin_pixels.mean(axis=0)

    # Return as RGB
    return (int(avg[2]), int(avg[1]), int(avg[0]))


def detect_gender_from_face(head_crop):
    """
    Run DeepFace gender detection on face/head crop only.
    Returns (woman_score, man_score, face_found).
    """
    if head_crop is None or head_crop.size == 0:
        return 0.0, 0.0, False

    # Must be large enough to contain a face
    if head_crop.shape[0] < 30 or head_crop.shape[1] < 30:
        return 0.0, 0.0, False

    try:
        res = DeepFace.analyze(
            head_crop,
            actions=["gender"],
            enforce_detection=True,
            detector_backend="retinaface",
            silent=True,
        )
        if isinstance(res, list):
            res = res[0]
        w = res["gender"].get("Woman", 0.0)
        m = res["gender"].get("Man", 0.0)
        return float(w), float(m), True
    except Exception:
        return 0.0, 0.0, False


def detect_gender_from_body(body_crop):
    """
    Run DeepFace gender detection on body crop with enforce_detection=False.
    Less reliable but works when face is not visible.
    Returns (woman_score, man_score).
    """
    if body_crop is None or body_crop.size == 0:
        return 0.0, 0.0

    try:
        res = DeepFace.analyze(
            body_crop,
            actions=["gender"],
            enforce_detection=False,
            silent=True,
        )
        if isinstance(res, list):
            res = res[0]
        w = res["gender"].get("Woman", 0.0)
        m = res["gender"].get("Man", 0.0)
        return float(w), float(m)
    except Exception:
        return 0.0, 0.0


def estimate_gender_from_silhouette(body_crop):
    """
    Fallback: estimate gender from shoulder-to-hip ratio.
    Women tend to have wider hips relative to shoulders.
    """
    if body_crop is None or body_crop.size == 0:
        return "female", 50.0

    gray = cv2.cvtColor(body_crop, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape

    def row_width(region):
        widths = []
        for row in region:
            nz = np.where(row > 20)[0]
            if len(nz) >= 2:
                widths.append(nz[-1] - nz[0])
        return np.mean(widths) if widths else W * 0.5

    shoulder_w = row_width(gray[:H//3, :])
    hip_w      = row_width(gray[2*H//3:, :])
    ratio      = hip_w / (shoulder_w + 1e-6)

    if ratio > 0.95:
        score = min(100.0, 50.0 + (ratio - 0.95) * 200)
        return "female", score
    else:
        score = min(100.0, 50.0 + (0.95 - ratio) * 200)
        return "male", score


def detect_person_attributes(frame_bgr, person_box):
    """
    Detect gender and skin tone for a single person.
    Uses face (primary) + silhouette (fallback) for gender.
    Uses direct pixel sampling for skin tone.
    """
    x, y, w, h = [int(v) for v in person_box]
    H, W = frame_bgr.shape[:2]

    # ── Extract crops ──────────────────────────────────────────────────────────
    # Full body crop
    body_crop = frame_bgr[max(0,y):min(H,y+h), max(0,x):min(W,x+w)]

    # Upper body crop (top 50%) for better gender detection
    upper_y2   = min(H, y + int(h * 0.5))
    upper_crop = frame_bgr[max(0,y):upper_y2, max(0,x):min(W,x+w)]

    # Head crop — tight around head only (top 30%)
    head_y1   = max(0, y)
    head_y2   = min(H, y + int(h * 0.30))
    head_x1   = max(0, x + int(w * 0.15))
    head_x2   = min(W, x + int(w * 0.85))
    head_crop = frame_bgr[head_y1:head_y2, head_x1:head_x2]

    # ── 1. Face detection (most reliable) ─────────────────────────────────────
    face_w, face_m, face_found = detect_gender_from_face(head_crop)
    print(f"  Face detection — Woman: {face_w:.1f} Man: {face_m:.1f} found: {face_found}")

    # ── 2. Upper body detection (fallback) ────────────────────────────────────
    body_w, body_m = 0.0, 0.0
    if not face_found:
        body_w, body_m = detect_gender_from_body(upper_crop)
        print(f"  Body detection — Woman: {body_w:.1f} Man: {body_m:.1f}")

    # ── 3. Silhouette fallback ─────────────────────────────────────────────────
    sil_gender, sil_score = estimate_gender_from_silhouette(body_crop)
    sil_w = sil_score if sil_gender == "female" else (100.0 - sil_score)
    sil_m = sil_score if sil_gender == "male"   else (100.0 - sil_score)
    print(f"  Silhouette — {sil_gender}: {sil_score:.1f}")

    # ── 4. Combine scores ──────────────────────────────────────────────────────
    if face_found:
        # Face is reliable — weight it heavily
        total_w = face_w * 0.80 + sil_w * 0.20
        total_m = face_m * 0.80 + sil_m * 0.20
    elif body_w > 0 or body_m > 0:
        # Body + silhouette
        total_w = body_w * 0.55 + sil_w * 0.45
        total_m = body_m * 0.55 + sil_m * 0.45
    else:
        # Silhouette only
        total_w = sil_w
        total_m = sil_m

    print(f"  Final scores — Woman: {total_w:.1f} Man: {total_m:.1f}")

    if total_w == 0 and total_m == 0:
        final_gender = "female"
    else:
        final_gender = "female" if total_w >= total_m else "male"

    # ── 5. Skin tone — direct pixel sampling ──────────────────────────────────
    skin_tone = sample_skin_tone_direct(upper_crop)
    print(f"  Skin tone (RGB): {skin_tone} | Gender: {final_gender}")

    return {"gender": final_gender, "skin_tone": skin_tone}