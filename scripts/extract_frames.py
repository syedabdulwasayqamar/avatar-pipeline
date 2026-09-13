import os
import cv2
import json
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm


def pick_clips_folder():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.update_idletasks()
    root.update()

    base = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "temp_clips")
    )

    folder = filedialog.askdirectory(
        parent=root,
        title="Select clip subfolder to extract frames from",
        initialdir=base if os.path.isdir(base) else os.path.dirname(__file__),
    )
    root.destroy()
    return folder


def extract_frames(clips_folder, frame_interval=3):
    BASE_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    video_name    = os.path.basename(clips_folder)
    output_dir    = os.path.join(BASE_DIR, "temp_frames", video_name)
    progress_file = os.path.join(clips_folder, "frames_extracted.json")

    os.makedirs(output_dir, exist_ok=True)

    clips = sorted(f for f in os.listdir(clips_folder) if f.endswith(".mp4"))
    if not clips:
        print(f"No .mp4 clips found in: {clips_folder}")
        return

    # Check actual output first
    existing_frames = set(f for f in os.listdir(output_dir) if f.endswith(".jpg"))

    if existing_frames:
        # Load progress JSON to know which clips are done and current frame count
        if os.path.isfile(progress_file):
            with open(progress_file, "r") as pf:
                progress = json.load(pf)
            if isinstance(progress, list):
                done         = set(progress)
                global_frame = len(existing_frames)
            else:
                done         = set(progress.get("done_clips", []))
                global_frame = progress.get("global_frame", len(existing_frames))
        else:
            done         = set()
            global_frame = len(existing_frames)
        print(f"Resuming — {len(done)} clip(s) done, {len(existing_frames)} frames in output.")
    else:
        # Output is empty — start fresh regardless of JSON
        print("Output folder is empty — extracting all clips.")
        done         = set()
        global_frame = 0

    for clip_name in clips:
        if clip_name in done:
            print(f"Skipping (already done): {clip_name}")
            continue

        clip_path = os.path.join(clips_folder, clip_name)
        cap       = cv2.VideoCapture(clip_path)
        total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        saved     = 0

        for i in tqdm(range(total), desc=f"Extracting {clip_name}", unit="frame"):
            ret, frame = cap.read()
            if not ret:
                break
            if i % frame_interval == 0:
                out_path = os.path.join(output_dir, f"frame_{global_frame:05d}.jpg")
                cv2.imwrite(out_path, frame)
                global_frame += 1
                saved += 1

        cap.release()
        print(f"  Saved {saved} frames → {output_dir}")

        done.add(clip_name)
        with open(progress_file, "w") as pf:
            json.dump({"done_clips": sorted(done), "global_frame": global_frame}, pf, indent=2)

    print(f"\nAll done. {global_frame} total frames saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=3,
                        help="Extract every Nth frame (1=all frames, 3=every 3rd, etc)")
    args, _ = parser.parse_known_args()

    clips_folder = pick_clips_folder()
    if not clips_folder:
        print("No folder selected. Exiting.")
        exit()

    extract_frames(clips_folder, frame_interval=args.interval)

