import os
import sys
sys.path.append(r"C:\Users\abdul\AppData\Roaming\Python\Python313\site-packages")
sys.path.append(r"C:\Program Files\Blender Foundation\Blender 5.1\5.1\python\lib\site-packages")
import cv2
import json
import math
import numpy as np

try:
    import bpy
except ImportError:
    raise ImportError("Run with: blender --background --python render_avatar.py -- <processed_folder>")

ROOT_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ROOT_DIR, "models", "smplx", "smplx_models")
FACES_PATH = os.path.join(MODELS_DIR, "smplfaces.npy")

# SMPL head vertex indices
SMPL_HEAD_VERTS = (list(range(5461, 5908)) +
                   list(range(412, 468)))


# ── Model ──────────────────────────────────────────────────────────────────────
def load_smpl_faces():
    faces = np.load(FACES_PATH)
    print(f"SMPL faces: {faces.shape}, max index: {faces.max()}")
    return faces


# ── Scene helpers ──────────────────────────────────────────────────────────────
def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def create_smpl_mesh(vertices, faces, name="SMPL_Mesh"):
    mesh = bpy.data.meshes.new(name)
    obj  = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    verts = [(float(v[0]), float(v[1]), float(v[2])) for v in vertices]
    polys = [tuple(int(i) for i in f) for f in faces]
    mesh.from_pydata(verts, [], polys)
    mesh.update()
    return obj


def apply_skin_shader(obj, skin_tone):
    mat           = bpy.data.materials.new(name=f"Skin_{obj.name}")
    mat.use_nodes = True
    nodes         = mat.node_tree.nodes
    links         = mat.node_tree.links
    for n in list(nodes):
        nodes.remove(n)
    bsdf   = nodes.new("ShaderNodeBsdfPrincipled")
    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    r, g, b = [c/255.0 for c in skin_tone]
    bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
    bsdf.inputs["Roughness"].default_value  = 0.4
    for key in ("Subsurface Weight", "Subsurface"):
        if key in bsdf.inputs:
            bsdf.inputs[key].default_value = 0.2
            break
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def setup_camera_orthographic(frame_w, frame_h):
    """1 Blender unit = 1 pixel."""
    bpy.ops.object.camera_add(location=(0, -1000, 0))
    cam = bpy.context.object
    cam.rotation_euler       = (math.radians(90), 0, 0)
    cam.data.type            = "ORTHO"
    cam.data.ortho_scale     = float(max(frame_w, frame_h))
    cam.data.clip_start      = 0.1
    cam.data.clip_end        = 10000.0
    bpy.context.scene.camera = cam
    return cam


def setup_lighting():
    """SUN lights — directional, scale-independent."""
    bpy.ops.object.light_add(type="SUN", location=(0, -500, 500))
    key = bpy.context.object
    key.data.energy    = 3.0
    key.rotation_euler = (math.radians(45), 0, math.radians(-30))

    bpy.ops.object.light_add(type="SUN", location=(200, -500, 200))
    fill = bpy.context.object
    fill.data.energy    = 1.0
    fill.rotation_euler = (math.radians(60), 0, math.radians(30))


def setup_world():
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    if world.node_tree is None:
        world.use_nodes = True
    nodes = world.node_tree.nodes
    bg    = nodes.get("Background") or nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value    = (0.0, 0.0, 0.0, 1.0)
    bg.inputs["Strength"].default_value = 0.0
    out = nodes.get("World Output") or nodes.new("ShaderNodeOutputWorld")
    if not out.inputs["Surface"].is_linked:
        world.node_tree.links.new(bg.outputs["Background"], out.inputs["Surface"])


# ── Projection ─────────────────────────────────────────────────────────────────

def verts_to_blender_ortho(verts, orig_cam, bbox_center, bbox_h, frame_w, frame_h):
    cx, cy = float(bbox_center[0]), float(bbox_center[1])
    h      = float(bbox_h)

    # SMPL Y spans ~[-0.96, 0.73] = total 1.69 units for full person height
    # So bbox_h pixels = 1.69 SMPL units → scale = bbox_h / 1.69
    smpl_height = 1.69
    scale       = h / smpl_height

    # SMPL root (pelvis) is at y≈-0.1, not y=0
    # Shift mesh down slightly so pelvis aligns with bbox center
    smpl_pelvis_offset = -0.1
    cy_adjusted = cy - smpl_pelvis_offset * scale

    x_px = cx          + verts[:, 0] * scale
    y_px = cy_adjusted + verts[:, 1] * scale

    bx =  x_px - frame_w / 2.0
    bz = -(y_px - frame_h / 2.0)
    by =  verts[:, 2] * scale

    result = np.zeros_like(verts)
    result[:, 0] = bx
    result[:, 1] = by
    result[:, 2] = bz
    return result

# ── Head mask ──────────────────────────────────────────────────────────────────
def get_head_mask(verts_bl, frame_w, frame_h):
    """Create soft elliptical mask over SMPL head region."""
    head_verts = verts_bl[SMPL_HEAD_VERTS]
    x_px = head_verts[:, 0] + frame_w / 2
    y_px = -(head_verts[:, 2]) + frame_h / 2

    x_min = int(np.clip(x_px.min() - 20, 0, frame_w))
    x_max = int(np.clip(x_px.max() + 20, 0, frame_w))
    y_min = int(np.clip(y_px.min() - 40, 0, frame_h))
    y_max = int(np.clip(y_px.max() + 20, 0, frame_h))

    mask = np.zeros((frame_h, frame_w), dtype=np.float32)
    cx   = int((x_min + x_max) / 2)
    cy   = int((y_min + y_max) / 2)
    rx   = max(1, (x_max - x_min) // 2)
    ry   = max(1, (y_max - y_min) // 2)

    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (31, 31), 15)
    return mask


# ── Compositing ────────────────────────────────────────────────────────────────

def composite_over_frame(avatar_png, frame_path, output_path, verts_bl_list, frame_w, frame_h):
    """Composite transparent avatar over original frame — no head masking."""
    avatar = cv2.imread(avatar_png, cv2.IMREAD_UNCHANGED)
    bg     = cv2.imread(frame_path)

    if avatar is None or bg is None:
        print(f"  Warning: could not composite")
        if avatar is not None:
            cv2.imwrite(output_path, avatar[:, :, :3])
        return

    if bg.shape[:2] != avatar.shape[:2]:
        bg = cv2.resize(bg, (avatar.shape[1], avatar.shape[0]))

    if avatar.shape[2] == 4:
        alpha      = avatar[:, :, 3:4].astype(np.float32) / 255.0
        avatar_bgr = avatar[:, :, :3].astype(np.float32)
        bg_f       = bg.astype(np.float32)
        composited = (avatar_bgr * alpha + bg_f * (1.0 - alpha)).astype(np.uint8)
    else:
        composited = avatar[:, :, :3]

    cv2.imwrite(output_path, composited)

# ── Frame size ─────────────────────────────────────────────────────────────────
def get_frame_size(processed_folder):
    video_name = os.path.basename(processed_folder)
    frames_dir = os.path.join(ROOT_DIR, "temp_frames", video_name)
    if os.path.isdir(frames_dir):
        jpgs = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
        if jpgs:
            img = cv2.imread(os.path.join(frames_dir, jpgs[0]))
            if img is not None:
                return img.shape[1], img.shape[0]
    return 360, 640


# ── Main render function ───────────────────────────────────────────────────────
def render_frame(npz_path, faces, output_path, frames_dir, frame_w=360, frame_h=640):
    clear_scene()
    setup_world()
    setup_camera_orthographic(frame_w, frame_h)
    setup_lighting()

    scene = bpy.context.scene
    scene.render.film_transparent = True

    raw = np.load(npz_path, allow_pickle=True)

    if "num_people" in raw:
        num_people = int(raw["num_people"])
        vertices   = raw["vertices"]
        orig_cams  = raw["orig_cam"]
        skin_tones = raw["skin_tone"]
    else:
        num_people = 1
        vertices   = raw["vertices"][np.newaxis]
        orig_cams  = np.array([[1.0, 0.0, 0.0, 0.0]])
        skin_tones = np.array(raw["skin_tone"])[np.newaxis]

    if num_people == 0:
        print(f"Skipping empty: {npz_path}")
        return

    verts_bl_list = []

    for i in range(num_people):
        verts       = vertices[i]
        orig_cam    = orig_cams[i]
        skin        = tuple(int(c) for c in skin_tones[i])
        bbox_center = raw["bbox_center"][i] if "bbox_center" in raw else np.array([frame_w/2, frame_h/2])
        bbox_h      = float(raw["bbox_h"][i]) if "bbox_h" in raw else float(frame_h * 0.75)

        verts_bl = verts_to_blender_ortho(
            verts, orig_cam[:3], bbox_center, bbox_h, frame_w, frame_h
        )
        verts_bl_list.append(verts_bl)

        obj = create_smpl_mesh(verts_bl, faces, name=f"Person_{i}")
        apply_skin_shader(obj, skin)

        #FINN 
        #bpy.ops.import_scene.fbx(
        #    filepath=r"D:\Art #Project\avatar_pipeline\Avatars\64-iron_man_mark_44_hulkbuster\Iron_Man_Mark_44_Hulkbuster\Iron_Man_Mark_44_Hulkbuster_fbx.FBX")       
        #obj = bpy.context.selected_objects[0]
        
    

    # Render to temp transparent PNG
    temp_path = output_path.replace(".png", "_tmp.png")
    scene.render.engine                     = "CYCLES"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode  = "RGBA"
    scene.render.filepath                   = temp_path
    scene.cycles.samples                    = 64
    scene.render.resolution_x              = frame_w
    scene.render.resolution_y              = frame_h
    bpy.ops.render.render(write_still=True)

    # Composite over original frame
    frame_name = os.path.splitext(os.path.basename(output_path))[0] + ".jpg"
    frame_path = os.path.join(frames_dir, frame_name)

    if os.path.isfile(temp_path):
        composite_over_frame(temp_path, frame_path, output_path, verts_bl_list, frame_w, frame_h)
        os.remove(temp_path)


# ── Args ───────────────────────────────────────────────────────────────────────
def get_args():
    argv = sys.argv
    if "--" not in argv:
        raise ValueError("Run: blender --background --python render_avatar.py -- <processed_folder>")
    user_args = argv[argv.index("--") + 1:]
    if not user_args:
        raise ValueError("Missing processed_folder argument.")
    return user_args[0]


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    processed_folder = get_args()
    if not os.path.isdir(processed_folder):
        print(f"Folder not found: {processed_folder}")
        exit()

    video_name    = os.path.basename(processed_folder)
    output_dir    = os.path.join(ROOT_DIR, "rendered_frames", video_name)
    frames_dir    = os.path.join(ROOT_DIR, "temp_frames", video_name)
    progress_file = os.path.join(processed_folder, "frames_rendered.json")
    os.makedirs(output_dir, exist_ok=True)

    frame_w, frame_h = get_frame_size(processed_folder)
    print(f"Frame size: {frame_w}x{frame_h}")

    faces     = load_smpl_faces()
    npz_files = sorted(f for f in os.listdir(processed_folder) if f.endswith(".npz"))

    if not npz_files:
        print(f"No .npz files in: {processed_folder}")
        exit()

    existing_png = set(f for f in os.listdir(output_dir) if f.endswith(".png"))
    if existing_png:
        if os.path.isfile(progress_file):
            with open(progress_file) as pf:
                recorded_done = set(json.load(pf))
            done = set(k for k in recorded_done if k.replace(".npz", ".png") in existing_png)
        else:
            done = set()
        print(f"Resuming — {len(done)}/{len(npz_files)} rendered.")
    else:
        print("Output empty — rendering all frames.")
        done = set()

    for idx, npz_name in enumerate(npz_files):
        if npz_name in done:
            continue

        npz_path = os.path.join(processed_folder, npz_name)
        out_name = npz_name.replace(".npz", ".png")
        out_path = os.path.join(output_dir, out_name)

        print(f"[{idx+1}/{len(npz_files)}] Rendering {out_name}...")
        render_frame(npz_path, faces, out_path, frames_dir, frame_w, frame_h)

        done.add(npz_name)
        with open(progress_file, "w") as pf:
            json.dump(sorted(done), pf, indent=2)

    print(f"\nAll done. {len(done)} frames → {output_dir}")