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
    from mathutils import Matrix, Vector, Quaternion
except ImportError:
    raise ImportError("Run with: blender --background --python retarget_fbx.py -- <processed_folder> <fbx_path>")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ── Verified VIBE 49 joint indices ─────────────────────────────────────────────
# Confirmed from Y values: VIBE Y points DOWN (more negative = higher on body)
# pelvis[0] y=-0.84, head[15] y=-0.87, feet[10,11] y=+0.22,+0.58
J_PELVIS       = 0
J_L_HIP        = 1
J_R_HIP        = 2
J_SPINE1       = 3
J_L_KNEE       = 4
J_R_KNEE       = 5
J_SPINE2       = 6
J_L_ANKLE      = 7
J_R_ANKLE      = 8
J_SPINE3       = 9
J_L_FOOT       = 10
J_R_FOOT       = 11
J_NECK         = 12
J_L_COLLAR     = 13
J_R_COLLAR     = 14
J_HEAD         = 15
J_L_SHOULDER   = 16
J_R_SHOULDER   = 17
J_L_ELBOW      = 18
J_R_ELBOW      = 19
J_L_WRIST      = 20
J_R_WRIST      = 21
J_L_HAND       = 22
J_R_HAND       = 23

# FBX bone → (parent_joint, child_joint) for direction vector
# Direction = child_pos - parent_pos defines where bone points
FBX_BONE_TO_VIBE_JOINTS = {
    "Base HumanPelvis"     : (J_PELVIS,     J_SPINE1),
    "Base HumanSpine1"     : (J_SPINE1,     J_SPINE2),
    "Base HumanSpine2"     : (J_SPINE2,     J_SPINE3),
    "Base HumanRibcage"    : (J_SPINE3,     J_NECK),
    "Base HumanNeck1"      : (J_NECK,       J_HEAD),
    "Base HumanLThigh"     : (J_L_HIP,      J_L_KNEE),
    "Base HumanLCalf"      : (J_L_KNEE,     J_L_ANKLE),
    "Base HumanLFoot"      : (J_L_ANKLE,    J_L_FOOT),
    "Base HumanRThigh"     : (J_R_HIP,      J_R_KNEE),
    "Base HumanRCalf"      : (J_R_KNEE,     J_R_ANKLE),
    "Base HumanRFoot"      : (J_R_ANKLE,    J_R_FOOT),
    "Base HumanLCollarbone": (J_L_COLLAR,   J_L_SHOULDER),
    "Base HumanLUpperarm"  : (J_L_SHOULDER, J_L_ELBOW),
    "Base HumanLForearm"   : (J_L_ELBOW,    J_L_WRIST),
    "Base HumanRCollarbone": (J_R_COLLAR,   J_R_SHOULDER),
    "Base HumanRUpperarm"  : (J_R_SHOULDER, J_R_ELBOW),
    "Base HumanRForearm"   : (J_R_ELBOW,    J_R_WRIST),
}


# ── Math helpers ───────────────────────────────────────────────────────────────
def normalize(v):
    v = np.array(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def rotation_from_vectors(v_from, v_to):
    v_from = normalize(v_from)
    v_to   = normalize(v_to)
    dot    = float(np.clip(np.dot(v_from, v_to), -1.0, 1.0))

    if dot > 0.9999:
        return Quaternion()
    if dot < -0.9999:
        perp = np.cross(v_from, [1, 0, 0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(v_from, [0, 1, 0])
        return Quaternion(Vector(normalize(perp)), math.pi)

    axis  = normalize(np.cross(v_from, v_to))
    angle = math.acos(dot)
    return Quaternion(Vector(axis), angle)


# ── Scene helpers ──────────────────────────────────────────────────────────────
def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def get_frame_size(processed_folder):
    video_name = os.path.basename(processed_folder)
    frames_dir = os.path.join(ROOT_DIR, "temp_frames", video_name)
    if os.path.isdir(frames_dir):
        jpgs = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
        if jpgs:
            img = cv2.imread(os.path.join(frames_dir, jpgs[0]))
            if img is not None:
                return img.shape[1], img.shape[0]
    return 1280, 720


def setup_camera_orthographic(frame_w, frame_h):
    bpy.ops.object.camera_add(location=(0, -1000, 0))
    cam = bpy.context.object
    cam.rotation_euler       = (math.radians(90), 0, 0)
    cam.data.type            = "ORTHO"
    cam.data.ortho_scale     = float(max(frame_w, frame_h))
    cam.data.clip_start      = 0.1
    cam.data.clip_end        = 100000.0
    bpy.context.scene.camera = cam
    return cam


def setup_lighting():
    bpy.ops.object.light_add(type="SUN", location=(0, -500, 500))
    key = bpy.context.object
    key.data.energy    = 5.0
    key.rotation_euler = (math.radians(45), 0, math.radians(-30))

    bpy.ops.object.light_add(type="SUN", location=(200, -500, 200))
    fill = bpy.context.object
    fill.data.energy    = 2.0
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


# ── FBX import ────────────────────────────────────────────────────────────────
def import_fbx(fbx_path):
    bpy.ops.import_scene.fbx(filepath=fbx_path)
    arm = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
    if arm is None:
        raise RuntimeError("No armature found in FBX.")
    arm.animation_data_clear()
    return arm


# ── Joint conversion ───────────────────────────────────────────────────────────
def vibe_joints_to_blender(joints, bbox_center, bbox_h, frame_w, frame_h):
    """
    Convert VIBE joints to Blender pixel-space.

    VIBE joint space:
      X = right
      Y = DOWN (confirmed: pelvis y=-0.84, feet y=+0.22 to +0.58)
      Z = forward/depth

    Blender ortho space:
      X = right  (same)
      Y = depth  (VIBE Z)
      Z = UP     (flip VIBE Y: negate and offset)

    Scale: bbox_h pixels = person height in VIBE space
    Person height in VIBE = feet_y - head_y (since Y is down, feet have higher Y)
    feet avg Y ≈ +0.4, head Y ≈ -0.87 → height ≈ 1.27 VIBE units
    But we use bbox_h directly since it matches the detected person size.
    """
    cx, cy = float(bbox_center[0]), float(bbox_center[1])
    h      = float(bbox_h)

    # Person height in VIBE space
    # foot Y - head Y (both in VIBE coords where Y is down)
    foot_y_vibe = (joints[J_L_FOOT, 1] + joints[J_R_FOOT, 1]) / 2.0
    head_y_vibe = joints[J_HEAD, 1]
    vibe_height = foot_y_vibe - head_y_vibe   # positive since feet have higher Y
    if vibe_height < 0.1:
        vibe_height = 1.27  # fallback

    # Scale: bbox_h pixels = vibe_height VIBE units
    scale = h / vibe_height

    # Pelvis position in image pixel space (cx, cy is bbox center ≈ pelvis)
    pelvis_y_vibe = joints[J_PELVIS, 1]

    bl = np.zeros_like(joints)
    for i in range(len(joints)):
        # X: right stays right
        bl[i, 0] = joints[i, 0] * scale + (cx - frame_w / 2)

        # Y: depth (VIBE Z)
        bl[i, 1] = joints[i, 2] * scale

        # Z: up = flip VIBE Y
        # pelvis in image = cy → in Blender Z = -(cy - frame_h/2)
        # other joints offset from pelvis
        delta_y_vibe = joints[i, 1] - pelvis_y_vibe
        pelvis_bl_z  = -(cy - frame_h / 2)
        bl[i, 2]     = pelvis_bl_z - delta_y_vibe * scale

    return bl


# ── Skeleton retargeting ───────────────────────────────────────────────────────
def apply_skeleton_retarget(armature, joints_bl):
    """
    Scale FBX to match VIBE skeleton proportions and drive bones
    by matching bone directions to VIBE joint directions.
    Aspect ratio of FBX is preserved — only uniform scale is applied.
    """
    # ── Uniform scale: match FBX height to VIBE skeleton height ───────────────
    # FBX rest pose Y values (head_local Y = up in FBX space before import rotation)
    fbx_foot_y = 5.29    # avg foot Y in FBX units
    fbx_head_y = 58.21   # head Y in FBX units
    fbx_height = fbx_head_y - fbx_foot_y  # 52.92 FBX units

    vibe_foot_z = (joints_bl[J_L_FOOT, 2] + joints_bl[J_R_FOOT, 2]) / 2.0
    vibe_head_z = joints_bl[J_HEAD, 2]
    vibe_height = abs(vibe_head_z - vibe_foot_z)
    if vibe_height < 1:
        vibe_height = 100

    # Uniform scale — preserves FBX aspect ratio
    final_scale = vibe_height / fbx_height
    print(f"  FBX height: {fbx_height:.2f} | VIBE height px: {vibe_height:.2f} | scale: {final_scale:.4f}")

    # ── Position armature so FBX pelvis aligns with VIBE pelvis ───────────────
    # FBX pelvis rest pos in FBX units
    fbx_pelvis = np.array([-0.5358, 35.8418, -12.9955])  # head_local XYZ

    # After FBX import, rotation is 90° X: FBX Y→BL Z, FBX Z→BL -Y
    # In Blender units after scale:
    fbx_pelvis_bl = np.array([
        fbx_pelvis[0] * final_scale,    # X stays X
       -fbx_pelvis[2] * final_scale,    # FBX Z → BL Y (negated)
        fbx_pelvis[1] * final_scale,    # FBX Y → BL Z
    ])

    vibe_pelvis_bl = joints_bl[J_PELVIS]
    origin = vibe_pelvis_bl - fbx_pelvis_bl

    armature.scale    = Vector((final_scale, final_scale, final_scale))
    armature.location = Vector((float(origin[0]), float(origin[1]), float(origin[2])))

    bpy.context.view_layer.update()

    # ── Apply bone rotations ───────────────────────────────────────────────────
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="POSE")

    # FBX bones rest directions are in FBX local space (Y-up, Z-forward before import)
    # After import 90° X rotation: FBX Y → BL Z, FBX Z → BL -Y
    fbx_to_bl = np.array([
        [1,  0,  0],
        [0,  0, -1],
        [0,  1,  0],
    ], dtype=np.float64)

    for fbx_bone_name, (j_parent, j_child) in FBX_BONE_TO_VIBE_JOINTS.items():
        pose_bone = armature.pose.bones.get(fbx_bone_name)
        if pose_bone is None:
            continue

        # Target direction in Blender world space from VIBE joints
        p1 = joints_bl[j_parent]
        p2 = joints_bl[j_child]
        if np.linalg.norm(p2 - p1) < 1e-6:
            continue
        target_dir = normalize(p2 - p1)

        # Rest direction of bone
        bone         = armature.data.bones[fbx_bone_name]
        rest_head    = np.array(bone.head_local)
        rest_tail    = np.array(bone.tail_local)
        rest_dir_fbx = normalize(rest_tail - rest_head)
        rest_dir_bl  = normalize(fbx_to_bl @ rest_dir_fbx)

        # World rotation from rest → target
        q_world = rotation_from_vectors(rest_dir_bl, target_dir)

        # Convert to bone local space
        if pose_bone.parent:
            parent_mat_inv = pose_bone.parent.matrix.to_3x3().inverted()
            q_local = (parent_mat_inv @ q_world.to_matrix().to_3x3()).to_quaternion()
        else:
            q_local = q_world

        pose_bone.rotation_mode       = "QUATERNION"
        pose_bone.rotation_quaternion = q_local

    bpy.ops.object.mode_set(mode="OBJECT")


# ── Compositing ────────────────────────────────────────────────────────────────
def composite_over_frame(avatar_png, frame_path, output_path, frame_w, frame_h):
    avatar = cv2.imread(avatar_png, cv2.IMREAD_UNCHANGED)
    bg     = cv2.imread(frame_path)

    if avatar is None or bg is None:
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


# ── Main render ────────────────────────────────────────────────────────────────
def render_frame_fbx(npz_path, fbx_path, output_path, frames_dir, frame_w, frame_h):
    clear_scene()
    setup_world()
    setup_camera_orthographic(frame_w, frame_h)
    setup_lighting()

    scene = bpy.context.scene
    scene.render.film_transparent = True

    raw = np.load(npz_path, allow_pickle=True)

    if "num_people" not in raw:
        print(f"Skipping old format: {npz_path}")
        return

    num_people  = int(raw["num_people"])
    joints_all  = raw["joints"]
    bbox_center = raw["bbox_center"]
    bbox_h      = raw["bbox_h"]

    if num_people == 0:
        print(f"Skipping empty: {npz_path}")
        return

    for i in range(num_people):
        h = float(bbox_h[i])
        if h < 50:
            print(f"  Skipping person {i} — bbox_h too small ({h:.1f})")
            continue

        print(f"  Person {i}: bbox_h={h:.1f}")
        joints_bl     = vibe_joints_to_blender(
            joints_all[i], bbox_center[i], h, frame_w, frame_h
        )
        armature      = import_fbx(fbx_path)
        armature.name = f"Avatar_{i}"
        apply_skeleton_retarget(armature, joints_bl)

    temp_path = output_path.replace(".png", "_tmp.png")
    scene.render.engine                     = "CYCLES"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode  = "RGBA"
    scene.render.filepath                   = temp_path
    scene.cycles.samples                    = 64
    scene.render.resolution_x              = frame_w
    scene.render.resolution_y              = frame_h
    bpy.ops.render.render(write_still=True)

    frame_name = os.path.splitext(os.path.basename(output_path))[0] + ".jpg"
    frame_path = os.path.join(frames_dir, frame_name)

    if os.path.isfile(temp_path):
        composite_over_frame(temp_path, frame_path, output_path, frame_w, frame_h)
        os.remove(temp_path)


# ── Args ───────────────────────────────────────────────────────────────────────
def get_args():
    argv = sys.argv
    if "--" not in argv:
        raise ValueError("Run: blender --background --python retarget_fbx.py -- <processed_folder> <fbx_path>")
    user_args = argv[argv.index("--") + 1:]
    if len(user_args) < 2:
        raise ValueError("Need: <processed_folder> <fbx_path>")
    return user_args[0], user_args[1]


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    processed_folder, fbx_path = get_args()

    if not os.path.isdir(processed_folder):
        print(f"Folder not found: {processed_folder}")
        exit()
    if not os.path.isfile(fbx_path):
        print(f"FBX not found: {fbx_path}")
        exit()

    video_name    = os.path.basename(processed_folder)
    output_dir    = os.path.join(ROOT_DIR, "rendered_frames_fbx", video_name)
    frames_dir    = os.path.join(ROOT_DIR, "temp_frames", video_name)
    progress_file = os.path.join(processed_folder, "frames_rendered_fbx.json")
    os.makedirs(output_dir, exist_ok=True)

    frame_w, frame_h = get_frame_size(processed_folder)
    print(f"Frame size: {frame_w}x{frame_h}")
    print(f"FBX: {fbx_path}")

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
        render_frame_fbx(npz_path, fbx_path, out_path, frames_dir, frame_w, frame_h)

        done.add(npz_name)
        with open(progress_file, "w") as pf:
            json.dump(sorted(done), pf, indent=2)

    print(f"\nAll done. {len(done)} frames → {output_dir}")