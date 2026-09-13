import os
import sys
import subprocess
import tkinter as tk
from tkinter import filedialog

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR   = os.path.join(ROOT_DIR, "scripts")
BLENDER_EXE   = r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
RENDER_SCRIPT = os.path.join(SCRIPTS_DIR, "render_avatar.py")

sys.path.insert(0, SCRIPTS_DIR)


# ── Folder picker ──────────────────────────────────────────────────────────────
def pick_processed_folder():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.update_idletasks()
    root.update()

    base = os.path.join(ROOT_DIR, "processed_temp_frames")

    folder = filedialog.askdirectory(
        parent=root,
        title="Select processed frames folder to render",
        initialdir=base if os.path.isdir(base) else ROOT_DIR,
    )
    root.destroy()
    return folder

def pick_render_mode():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.update_idletasks()
    root.update()

    # Simple dialog to choose mode
    mode = tk.simpledialog.askstring(
        "Render Mode",
        "Enter render mode:\n  1 = SMPL mesh (default)\n  2 = FBX avatar",
        initialvalue="1",
        parent=root,
    )
    root.destroy()
    return mode.strip() if mode else "1"


def pick_fbx_file():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.update_idletasks()
    root.update()

    base = os.path.join(ROOT_DIR, "Avatars")
    path = filedialog.askopenfilename(
        parent=root,
        title="Select FBX avatar file",
        initialdir=base if os.path.isdir(base) else ROOT_DIR,
        filetypes=[("FBX files", "*.fbx *.FBX"), ("All files", "*.*")],
    )
    root.destroy()
    return path

# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from tkinter import simpledialog

    processed_folder = pick_processed_folder()
    if not processed_folder:
        print("No folder selected. Exiting.")
        exit()

    mode = pick_render_mode()

    if mode == "2":
        # FBX mode
        fbx_path = pick_fbx_file()
        if not fbx_path:
            print("No FBX selected. Exiting.")
            exit()

        print(f"Selected: {processed_folder}")
        print(f"FBX:      {fbx_path}")
        print(f"Mode:     FBX avatar retargeting")
        print(f"Launching Blender...")

        cmd = [
            BLENDER_EXE,
            "--background",
            "--python", os.path.join(SCRIPTS_DIR, "retarget_fbx.py"),
            "--",
            processed_folder,
            fbx_path,
        ]
    else:
        # Default SMPL mesh mode
        print(f"Selected: {processed_folder}")
        print(f"Mode:     SMPL mesh")
        print(f"Launching Blender...")

        cmd = [
            BLENDER_EXE,
            "--background",
            "--python", RENDER_SCRIPT,
            "--",
            processed_folder,
        ]

    print(f"Running: {' '.join(cmd)}\n")
    subprocess.run(cmd)