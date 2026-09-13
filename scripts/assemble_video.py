import os
import cv2
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm


def pick_rendered_folder():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.update_idletasks()
    root.update()

    base = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "rendered_frames")
    )

    folder = filedialog.askdirectory(
        parent=root,
        title="Select rendered frames folder to assemble into video",
        initialdir=base if os.path.isdir(base) else os.path.dirname(__file__),
    )
    root.destroy()
    return folder


def get_source_fps(video_name):
    """Read FPS from original clips."""
    ROOT_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    clips_dir  = os.path.join(ROOT_DIR, "temp_clips", video_name)
    if os.path.isdir(clips_dir):
        clips = sorted(f for f in os.listdir(clips_dir) if f.endswith(".mp4"))
        if clips:
            cap = cv2.VideoCapture(os.path.join(clips_dir, clips[0]))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if fps > 0:
                return fps
    return 30.0  # fallback


def assemble_video(rendered_folder, output_fps=None):
    ROOT_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    video_name = os.path.basename(rendered_folder)
    output_dir = os.path.join(ROOT_DIR, "output_videos")
    os.makedirs(output_dir, exist_ok=True)

    # Auto-detect FPS from source if not specified
    if output_fps is None:
        output_fps = get_source_fps(video_name)
        print(f"Auto-detected source FPS: {output_fps:.3f}")

    output_path = os.path.join(output_dir, f"{video_name}.mp4")

    frames = sorted(f for f in os.listdir(rendered_folder) if f.endswith(".png"))
    if not frames:
        print(f"No .png frames found in: {rendered_folder}")
        return

    first = cv2.imread(os.path.join(rendered_folder, frames[0]))
    if first is None:
        print("Could not read first frame.")
        return

    h, w = first.shape[:2]
    long_side = max(w, h)

    # Resolution label for output
    if long_side <= 480:
        res_label = "360p"
    elif long_side <= 960:
        res_label = "360p"
    elif long_side <= 1280:
        res_label = "720p"
    else:
        res_label = "1080p"

    print(f"\n{'='*50}")
    print(f"ASSEMBLING VIDEO")
    print(f"{'='*50}")
    print(f"  Profile    : {res_label}")
    print(f"  Frame size : {w}x{h}")
    print(f"  Frames     : {len(frames)}")
    print(f"  FPS        : {output_fps:.3f}")
    print(f"  Duration   : {len(frames)/output_fps:.1f}s")
    print(f"  Output     : {output_path}")
    print(f"{'='*50}\n")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, output_fps, (w, h))

    for frame_name in tqdm(frames, desc="Assembling video", unit="frame"):
        frame = cv2.imread(os.path.join(rendered_folder, frame_name))
        if frame is None:
            print(f"  Warning: could not read {frame_name}, skipping.")
            continue
        if frame.shape[1] != w or frame.shape[0] != h:
            frame = cv2.resize(frame, (w, h))
        writer.write(frame)

    writer.release()
    print(f"\nDone. Video saved to: {output_path}")


if __name__ == "__main__":
    rendered_folder = pick_rendered_folder()
    if not rendered_folder:
        print("No folder selected. Exiting.")
        exit()

    assemble_video(rendered_folder)  # FPS auto-detected from source
