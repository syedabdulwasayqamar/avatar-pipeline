import os
import cv2
import json
import torch
import numpy as np

from detect_gender import detect_person_attributes

print("PROCESS_FRAME VERSION 4 — SMPL only")

ROOT_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR  = os.path.join(ROOT_DIR, "models", "smplx", "smplx_models")
FACES_PATH  = os.path.join(MODELS_DIR, "smplfaces.npy")


# ── Model Loading ──────────────────────────────────────────────────────────────
def load_vibe_model():
    from lib.models.vibe import VIBE_Demo
    device      = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    weight_path = os.path.join(ROOT_DIR, "models", "VIBE", "data", "vibe_data", "vibe_model_w_3dpw.pth.tar")

    model = VIBE_Demo(
        seqlen=16, n_layers=2, hidden_size=1024,
        add_linear=True, use_residual=True,
    ).to(device)

    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["gen_state_dict"], strict=False)
    model.eval()
    return model


def load_smpl_faces():
    """Load SMPL face indices — same for male and female."""
    faces = np.load(FACES_PATH)
    print(f"SMPL faces loaded: {faces.shape}")
    return faces


# ── Helpers ────────────────────────────────────────────────────────────────────
def get_bbox_center_and_side(bbox, scale=1.1):
    x, y, w, h = bbox
    return x + w/2.0, y + h/2.0, max(w, h) * scale


def crop_and_resize(frame, cx, cy, side, target_size=224):
    H, W  = frame.shape[:2]
    x1, y1 = int(cx - side/2), int(cy - side/2)
    x2, y2 = int(cx + side/2), int(cy + side/2)

    pad_t = max(0, -y1);  pad_b = max(0, y2 - H)
    pad_l = max(0, -x1);  pad_r = max(0, x2 - W)

    padded = cv2.copyMakeBorder(frame, pad_t, pad_b, pad_l, pad_r,
                                cv2.BORDER_CONSTANT, value=[0, 0, 0])
    crop   = padded[y1+pad_t:y2+pad_t, x1+pad_l:x2+pad_l]
    return cv2.resize(crop, (target_size, target_size))


def run_vibe_on_crop(crop_rgb, vibe_model):
    device = next(vibe_model.parameters()).device

    t    = torch.from_numpy(crop_rgb).float() / 255.0
    t    = t.permute(2, 0, 1).unsqueeze(0).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
    t    = (t - mean) / std
    t    = t.unsqueeze(1)   # (1, 1, 3, 224, 224)

    with torch.no_grad():
        pred = vibe_model(t)[-1]

    # Use frame 0 of the 16-frame sequence
    return {
        "theta": pred["theta"].cpu().numpy()[0, 0],  # (85,)
        "verts": pred["verts"].cpu().numpy()[0, 0],  # (6890, 3)
        "kp3d":  pred["kp_3d"].cpu().numpy()[0, 0], # (49, 3)
    }


# ── Per-person processing ──────────────────────────────────────────────────────
def process_person_crop(frame_bgr, bbox, W, H, vibe_model):
    cx, cy, side = get_bbox_center_and_side(bbox, scale=1.1)

    # Gender + skin tone
    attributes = detect_person_attributes(frame_bgr, bbox)
    gender     = attributes["gender"]
    skin_tone  = attributes["skin_tone"]

    # Run VIBE — use its verts directly (SMPL space, 6890 verts)
    frame_rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    crop_resized = crop_and_resize(frame_rgb, cx, cy, side, target_size=224)
    vibe_result  = run_vibe_on_crop(crop_resized, vibe_model)

    theta = vibe_result["theta"]   # (85,)
    verts = vibe_result["verts"]   # (6890, 3) — use directly, no SMPLX re-fit
    kp3d  = vibe_result["kp3d"]   # (49, 3)

    # Camera projection
    cam      = theta[:3]
    sx       = cam[0] * (1.0 / (W / side))
    sy       = cam[0] * (1.0 / (H / side))
    tx       = ((cx - W/2) / (W/2) / (sx + 1e-8)) + cam[1]
    ty       = ((cy - H/2) / (H/2) / (sy + 1e-8)) + cam[2]
    orig_cam = np.array([sx, sy, tx, ty])

    return {
        "vertices":    verts,                   # (6890, 3) SMPL verts
        "joints":      kp3d,                    # (49, 3)
        "smpl_theta":  theta[:75],
        "smpl_beta":   theta[69:79],
        "orig_cam":    orig_cam,
        "bbox_center": np.array([cx, cy]),
        "bbox_h":      side,
        "gender":      gender,
        "skin_tone":   skin_tone,
    }


# ── Multi-person frame processing ──────────────────────────────────────────────
def process_frame_multiperson(frame_path, vibe_model, person_det_model, max_people=2):
    frame = cv2.imread(frame_path)
    if frame is None:
        return []

    H, W   = frame.shape[:2]
    device = next(person_det_model.parameters()).device

    img_rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2,0,1).float() / 255.0
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        predictions = person_det_model([img_tensor])[0]

    boxes  = predictions["boxes"].cpu().numpy()
    labels = predictions["labels"].cpu().numpy()
    scores = predictions["scores"].cpu().numpy()

    # Collect person boxes
    person_boxes = [
        [x1, y1, x2-x1, y2-y1]
        for (x1,y1,x2,y2), label, score in zip(boxes, labels, scores)
        if label == 1 and score > 0.7
    ]
    if not person_boxes:
        person_boxes = [
            [x1, y1, x2-x1, y2-y1]
            for (x1,y1,x2,y2), label, score in zip(boxes, labels, scores)
            if label == 1 and score > 0.5
        ]

    person_boxes = sorted(person_boxes, key=lambda b: b[0])

    import traceback
    people_data = []
    for bbox in person_boxes[:max_people]:
        try:
            r = process_person_crop(frame, bbox, W, H, vibe_model)
            people_data.append(r)
        except Exception:
            traceback.print_exc()

    return people_data