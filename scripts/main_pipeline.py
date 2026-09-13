import os
import sys
import cv2
import json
import numpy as np
from tqdm import tqdm
# Add near top of __main__ block
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--interval", type=int, default=1,
                    help="Frame interval (1=every frame for 720p quality)")
parser.add_argument("--max_people", type=int, default=3)
args, _ = parser.parse_known_args()

MAX_PEOPLE     = args.max_people
FRAME_INTERVAL = args.interval


try:
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
except ImportError as e:
    print(f"\n[!] ERROR: {e}")
    sys.exit(1)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(ROOT_DIR, "scripts")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, SCRIPTS_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "models", "VIBE"))

import lib.core.config as vibe_config
vibe_config.VIBE_DATA_DIR = os.path.join(ROOT_DIR, "models", "VIBE", "data", "vibe_data")

from split_video import split_video, pick_video
from extract_frames import extract_frames
from process_frame import process_frame_multiperson, load_vibe_model


# ── Stage 1: Split ─────────────────────────────────────────────────────────────
def stage_split(input_path, clip_length_sec=5):
    video_name = os.path.splitext(os.path.basename(input_path))[0]
    clips_dir  = os.path.join(ROOT_DIR, "temp_clips", video_name)

    if os.path.isdir(clips_dir) and any(f.endswith(".mp4") for f in os.listdir(clips_dir)):
        print(f"Clips already exist — skipping: {clips_dir}")
        return clips_dir

    print(f"\n{'='*50}\nSTAGE 1 — Splitting: {video_name}\n{'='*50}")
    split_video(input_path, clips_dir, clip_length_sec=clip_length_sec)
    return clips_dir


# ── Stage 2: Extract frames ────────────────────────────────────────────────────
def stage_extract(clips_dir, frame_interval=3):
    print(f"\n{'='*50}\nSTAGE 2 — Extracting frames\n{'='*50}")
    extract_frames(clips_dir, frame_interval=frame_interval)
    return os.path.join(ROOT_DIR, "temp_frames", os.path.basename(clips_dir))


# ── Stage 3: Process frames ────────────────────────────────────────────────────
def stage_process(frames_dir, vibe_model, person_det_model, max_people=2):
    print(f"\n{'='*50}\nSTAGE 3 — VIBE + DeepFace (SMPL verts)\n{'='*50}")

    video_name    = os.path.basename(frames_dir)
    output_dir    = os.path.join(ROOT_DIR, "processed_temp_frames", video_name)
    progress_file = os.path.join(frames_dir, "frames_processed.json")
    os.makedirs(output_dir, exist_ok=True)

    frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    if not frame_files:
        print(f"No frames found in: {frames_dir}")
        return output_dir

    existing_npz = set(f for f in os.listdir(output_dir) if f.endswith(".npz"))
    if existing_npz:
        if os.path.isfile(progress_file):
            with open(progress_file) as pf:
                recorded_done = set(json.load(pf))
            done = set(f for f in recorded_done
                       if os.path.splitext(f)[0] + ".npz" in existing_npz)
        else:
            done = set()
        print(f"Resuming — {len(done)}/{len(frame_files)} done.")
    else:
        print("Output empty — processing all frames.")
        done = set()

    for frame_name in tqdm(frame_files, desc="Processing", unit="frame"):
        if frame_name in done:
            continue

        frame_path = os.path.join(frames_dir, frame_name)
        out_path   = os.path.join(output_dir, frame_name.replace(".jpg", ".npz"))

        people = process_frame_multiperson(
            frame_path, vibe_model, person_det_model, max_people=max_people
        )

        if not people:
            print(f"  No people found in {frame_name}, skipping.")
            continue

        np.savez(
            out_path,
            num_people  = np.array(len(people)),
            vertices    = np.stack([p["vertices"]    for p in people]),  # (N, 6890, 3)
            joints      = np.stack([p["joints"]      for p in people]),  # (N, 49, 3)
            smpl_theta  = np.stack([p["smpl_theta"]  for p in people]),
            smpl_beta   = np.stack([p["smpl_beta"]   for p in people]),
            orig_cam    = np.stack([p["orig_cam"]    for p in people]),
            bbox_center = np.stack([p["bbox_center"] for p in people]),
            bbox_h      = np.array([p["bbox_h"]      for p in people]),
            skin_tone   = np.stack([np.array(p["skin_tone"]) for p in people]),
            gender      = np.array([p["gender"]      for p in people]),
        )

        done.add(frame_name)
        with open(progress_file, "w") as pf:
            json.dump(sorted(done), pf, indent=2)

    print(f"  {len(done)} frames processed → {output_dir}")
    return output_dir


# ── Resolution profiles ────────────────────────────────────────────────────────
RESOLUTION_PROFILES = {
    "360p":  {"frame_interval": 3, "max_people": 3, "label": "360p  (640×360  or 360×640)"},
    "720p":  {"frame_interval": 1, "max_people": 3, "label": "720p  (1280×720 or 720×1280)"},
    "1080p": {"frame_interval": 1, "max_people": 3, "label": "1080p (1920×1080 or 1080×1920)"},
}

def detect_resolution_profile(input_path):
    """
    Read first frame of video and return the matching resolution profile.
    """
    cap = cv2.VideoCapture(input_path)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    long_side = max(w, h)

    if long_side <= 480:
        profile_key = "360p"
    elif long_side <= 960:
        profile_key = "360p"
    elif long_side <= 1280:
        profile_key = "720p"
    else:
        profile_key = "1080p"

    profile = RESOLUTION_PROFILES[profile_key]

    print(f"\n{'='*50}")
    print(f"VIDEO RESOLUTION DETECTED")
    print(f"{'='*50}")
    print(f"  Resolution : {w}x{h}")
    print(f"  FPS        : {fps:.3f}")
    print(f"  Profile    : {profile['label']}")
    print(f"  Interval   : every {profile['frame_interval']} frame(s)")
    print(f"  Max people : {profile['max_people']}")
    print(f"{'='*50}\n")

    return profile, w, h, fps


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    input_path = pick_video()
    if not input_path:
        print("No video selected.")
        exit()

    print(f"\nSelected: {input_path}")

    # Auto-detect resolution and select profile
    profile, vid_w, vid_h, vid_fps = detect_resolution_profile(input_path)
    FRAME_INTERVAL = profile["frame_interval"]
    MAX_PEOPLE     = profile["max_people"]

    # Stage 1 — Split
    clips_dir = stage_split(input_path, clip_length_sec=5)

    # Stage 2 — Extract frames
    frames_dir = stage_extract(clips_dir, frame_interval=FRAME_INTERVAL)

    # Stage 3 — Detect gender
    gender = stage_detect_gender(frames_dir) if hasattr(__builtins__, 'stage_detect_gender') else "female"

    # Load models
    print(f"\n{'='*50}")
    print(f"Loading models (VIBE + FasterRCNN)...")
    print(f"{'='*50}")
    vibe_model = load_vibe_model()

    weights          = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    person_det_model = fasterrcnn_resnet50_fpn(weights=weights)
    person_det_model.eval()
    person_det_model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Stage 4 — Process frames
    processed_dir = stage_process(
        frames_dir, vibe_model, person_det_model, MAX_PEOPLE
    )

    print(f"\n{'='*50}")
    print("Pipeline complete!")
    print(f"  Resolution : {vid_w}x{vid_h} @ {vid_fps:.3f} FPS")
    print(f"  Profile    : {profile['label']}")
    print(f"  Clips      → {clips_dir}")
    print(f"  Frames     → {frames_dir}")
    print(f"  Processed  → {processed_dir}")
    print(f"{'='*50}")