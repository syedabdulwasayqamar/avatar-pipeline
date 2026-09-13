import os
import cv2
import argparse
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog


def pick_video():
    """Stable Tkinter file picker (prevents freeze issues)."""
    root = tk.Tk()
    root.withdraw()

    # force window to front
    root.attributes("-topmost", True)
    root.update_idletasks()
    root.update()

    file_path = filedialog.askopenfilename(
        parent=root,
        title="Select a video file",
        initialdir=os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "input_videos")
        ),
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
    )

    root.destroy()
    return file_path


def split_video(input_path, output_dir, clip_length_sec=5, overlap_sec=0):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    clip_frames = int(clip_length_sec * fps)
    overlap_frames = int(overlap_sec * fps)
    step_frames = clip_frames - overlap_frames

    clip_idx = 0

    for start_frame in tqdm(range(0, total_frames, step_frames), desc="Splitting video"):
        end_frame = min(start_frame + clip_frames, total_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        output_path = os.path.join(output_dir, f"clip_{clip_idx:04d}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        )

        for _ in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        out.release()
        clip_idx += 1

    cap.release()
    print(f"Split {input_path} into {clip_idx} clips in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a video into shorter clips.")
    parser.add_argument("--input", type=str, required=False)
    parser.add_argument("--output_dir", type=str, default="temp_clips")
    parser.add_argument("--clip_length", type=int, default=5)
    parser.add_argument("--overlap", type=int, default=0)
    args = parser.parse_args()

    # -------------------------
    # INPUT VIDEO SELECTION
    # -------------------------
    if not args.input:
        input_path = pick_video()

        if not input_path:
            print("No file selected. Exiting.")
            exit()
    else:
        input_path = args.input

    # -------------------------
    # OUTPUT PATH (FIXED ROOT)
    # -------------------------
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    video_name = os.path.splitext(os.path.basename(input_path))[0]

    output_dir = os.path.normpath(
        os.path.join(BASE_DIR, "..", "temp_clips", video_name)
    )

    # -------------------------
    # RUN
    # -------------------------
    split_video(input_path, output_dir, args.clip_length, args.overlap)