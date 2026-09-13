import os
import sys
sys.path.append(r"C:\Users\abdul\AppData\Roaming\Python\Python313\site-packages")
sys.path.append(r"C:\Program Files\Blender Foundation\Blender 5.1\5.1\python\lib\site-packages")
import math
import cv2
import numpy as np

import bpy
from mathutils import Vector, Quaternion

FBX_PATH      = r"D:\Art Project\avatar_pipeline\Avatars\X Bot.fbx"
PROCESSED_DIR = r"D:\Art Project\avatar_pipeline\processed_temp_frames\OmgHelayna - Coop"
FRAMES_DIR    = r"D:\Art Project\avatar_pipeline\temp_frames\OmgHelayna - Coop"
OUTPUT_DIR    = r"D:\Art Project\avatar_pipeline\test_multi_frame"

TEST_FRAMES = ["frame_00017.npz", "frame_00020.npz", "frame_00024.npz", "frame_00026.npz"]

BONE_REST = {
    "mixamorig:Hips":         (np.array([0, 104.27, 1.55]),    np.array([0, 114.83, 1.69])),
    "mixamorig:Spine":        (np.array([0, 114.46, 1.69]),    np.array([0, 124.35, 0.22])),
    "mixamorig:Spine1":       (np.array([0, 124.35, 0.22]),    np.array([0, 133.57, -1.16])),
    "mixamorig:Spine2":       (np.array([0, 133.57, -1.16]),   np.array([0, 147.17, -2.82])),
    "mixamorig:Neck":         (np.array([0, 150.31, -3.20]),   np.array([0, 160.00, -4.39])),
    "mixamorig:Head":         (np.array([0, 159.93, -1.52]),   np.array([0, 183.04, -4.35])),
    "mixamorig:LeftUpLeg":    (np.array([8.21, 97.52, -0.05]), np.array([8.21, 53.15, 0.30])),
    "mixamorig:LeftLeg":      (np.array([8.21, 53.15, 0.30]),  np.array([8.21, 8.73, -2.74])),
    "mixamorig:LeftFoot":     (np.array([8.21, 8.73, -2.74]),  np.array([8.21, 0.00, 7.97])),
    "mixamorig:RightUpLeg":   (np.array([-8.21, 97.52, -0.05]),np.array([-8.21, 53.15, 0.30])),
    "mixamorig:RightLeg":     (np.array([-8.21, 53.15, 0.30]), np.array([-8.21, 8.73, -2.74])),
    "mixamorig:RightFoot":    (np.array([-8.21, 8.73, -2.74]), np.array([-8.21, 0.00, 7.97])),
    "mixamorig:LeftShoulder": (np.array([4.57, 144.59, -3.32]),np.array([15.16, 144.06, -5.55])),
    "mixamorig:LeftArm":      (np.array([15.16, 144.06, -5.55]),np.array([43.00, 144.06, -5.55])),
    "mixamorig:LeftForeArm":  (np.array([43.00, 144.06, -5.55]),np.array([71.33, 144.06, -5.55])),
    "mixamorig:RightShoulder":(np.array([-4.57, 144.59, -3.32]),np.array([-15.16, 144.06, -5.55])),
    "mixamorig:RightArm":     (np.array([-15.16, 144.06, -5.55]),np.array([-43.00, 144.06, -5.55])),
    "mixamorig:RightForeArm": (np.array([-43.00, 144.06, -5.55]),np.array([-71.33, 144.06, -5.55])),
}

J_NOSE, J_NECK = 0, 1
J_R_SHOULDER, J_R_ELBOW, J_R_WRIST = 2, 3, 4
J_L_SHOULDER, J_L_ELBOW, J_L_WRIST = 5, 6, 7
J_PELVIS = 8
J_R_HIP, J_R_KNEE, J_R_ANKLE = 9, 10, 11
J_L_HIP, J_L_KNEE, J_L_ANKLE = 12, 13, 14
J_HEAD = 38

BONE_TO_VIBE = {
    "mixamorig:Hips":          (J_PELVIS,     J_NECK),
    "mixamorig:Spine2":        (J_NECK,       J_HEAD),
    "mixamorig:Neck":          (J_NECK,       J_HEAD),
    "mixamorig:LeftUpLeg":     (J_L_HIP,      J_L_KNEE),
    "mixamorig:LeftLeg":       (J_L_KNEE,     J_L_ANKLE),
    "mixamorig:RightUpLeg":    (J_R_HIP,      J_R_KNEE),
    "mixamorig:RightLeg":      (J_R_KNEE,     J_R_ANKLE),
    "mixamorig:LeftArm":       (J_L_SHOULDER, J_L_ELBOW),
    "mixamorig:LeftForeArm":   (J_L_ELBOW,    J_L_WRIST),
    "mixamorig:RightArm":      (J_R_SHOULDER, J_R_ELBOW),
    "mixamorig:RightForeArm":  (J_R_ELBOW,    J_R_WRIST),
}

BONE_ORDER = [
    "mixamorig:Hips", "mixamorig:Spine2", "mixamorig:Neck",
    "mixamorig:LeftUpLeg", "mixamorig:LeftLeg",
    "mixamorig:RightUpLeg", "mixamorig:RightLeg",
    "mixamorig:LeftArm", "mixamorig:LeftForeArm",
    "mixamorig:RightArm", "mixamorig:RightForeArm",
]


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


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def setup_camera_ortho(frame_w, frame_h):
    bpy.ops.object.camera_add(location=(0, -1000, 0))
    cam = bpy.context.object
    cam.rotation_euler   = (math.radians(90), 0, 0)
    cam.data.type        = "ORTHO"

    # Explicitly set sensor_fit based on aspect ratio so ortho_scale maps correctly
    if frame_h >= frame_w:
        cam.data.sensor_fit  = "VERTICAL"
        cam.data.ortho_scale = float(frame_h)
    else:
        cam.data.sensor_fit  = "HORIZONTAL"
        cam.data.ortho_scale = float(frame_w)

    cam.data.clip_start  = 0.1
    cam.data.clip_end    = 100000.0
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


def import_character():
    bpy.ops.import_scene.fbx(filepath=FBX_PATH)
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    arm.animation_data_clear()
    return arm

def apply_vibe_pose_and_position(armature, joints, bbox_center, bbox_h, frame_w, frame_h):
    """Position + pose the armature to match VIBE data in pixel space."""
    J_PELVIS, J_HEAD = 8, 38
    J_L_ANKLE, J_R_ANKLE = 14, 11

    pelvis = joints[J_PELVIS]
    head   = joints[J_HEAD]
    l_ankle, r_ankle = joints[J_L_ANKLE], joints[J_R_ANKLE]
    foot_avg_y  = (l_ankle[1] + r_ankle[1]) / 2.0
    vibe_height = abs(foot_avg_y - head[1])
    if vibe_height < 0.1:
        vibe_height = 1.6

    h = float(bbox_h)
    cx, cy = float(bbox_center[0]), float(bbox_center[1])

    # ── SINGLE scale: pixels per VIBE unit ─────────────────────────────────────
    px_per_vibe_unit = h / vibe_height

    # ── Convert all VIBE joints to Blender pixel-space (relative to pelvis) ───
    def to_bl(joint):
        x = (joint[0] - pelvis[0]) * px_per_vibe_unit + cx - frame_w/2
        y_depth = (joint[2] - pelvis[2]) * px_per_vibe_unit
        z = -(joint[1] - pelvis[1]) * px_per_vibe_unit - (cy - frame_h/2)
        return np.array([x, y_depth, z])

    joints_bl = np.array([to_bl(j) for j in joints])
    pelvis_bl = joints_bl[J_PELVIS]  # should be at (cx-frame_w/2, 0, -(cy-frame_h/2))

    # ── FBX scale: how much to scale FBX (in its native units) to match px_per_vibe_unit ──
    # FBX height (foot to head, FBX units) ≈ 52.92
    # VIBE height (foot to head, VIBE units) ≈ vibe_height
    # We want: fbx_height * fbx_scale = h (pixels)
    fbx_foot_y = 5.29
    fbx_head_y = 58.21
    fbx_height = fbx_head_y - fbx_foot_y  # 52.92

    fbx_scale = h / fbx_height   # pixels per FBX unit, single clean derivation

    # ── FBX pelvis (Hips bone head_local) in Blender axes after import rotation ──
    hips_local = BONE_REST["mixamorig:Hips"][0]  # [0, 104.27, 1.55] FBX units
    fbx_pelvis_bl = np.array([
        hips_local[0] * fbx_scale,
       -hips_local[2] * fbx_scale,
        hips_local[1] * fbx_scale,
    ])

    # ── Position armature so FBX pelvis lands exactly on VIBE pelvis ───────────
    origin = pelvis_bl - fbx_pelvis_bl

    armature.scale    = Vector((fbx_scale, fbx_scale, fbx_scale))
    armature.location = Vector((float(origin[0]), float(origin[1]), float(origin[2])))

    bpy.context.view_layer.update()

    # ── Pose bones (unchanged) ─────────────────────────────────────────────────
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="POSE")

    fbx_to_bl = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0],
    ], dtype=np.float64)

    for bone_name in BONE_ORDER:
        if bone_name not in BONE_TO_VIBE or bone_name not in BONE_REST:
            continue
        j_parent, j_child = BONE_TO_VIBE[bone_name]
        rest_head, rest_tail = BONE_REST[bone_name]

        pose_bone = armature.pose.bones.get(bone_name)
        if pose_bone is None:
            continue

        p1, p2 = joints[j_parent], joints[j_child]
        vibe_dir = np.array([p2[0]-p1[0], p2[2]-p1[2], -(p2[1]-p1[1])])
        if np.linalg.norm(vibe_dir) < 1e-6:
            continue
        target_dir = normalize(vibe_dir)

        rest_dir_fbx = normalize(rest_tail - rest_head)
        rest_dir_bl  = normalize(fbx_to_bl @ rest_dir_fbx)
        q_world = rotation_from_vectors(rest_dir_bl, target_dir)

        if pose_bone.parent:
            bpy.context.view_layer.update()
            parent_inv = pose_bone.parent.matrix.to_3x3().inverted()
            q_local = (parent_inv @ q_world.to_matrix().to_3x3()).to_quaternion()
        else:
            q_local = q_world

        pose_bone.rotation_mode       = "QUATERNION"
        pose_bone.rotation_quaternion = q_local
        bpy.context.view_layer.update()

    bpy.ops.object.mode_set(mode="OBJECT")


def composite_over_frame(avatar_png, frame_path, output_path):
    avatar = cv2.imread(avatar_png, cv2.IMREAD_UNCHANGED)
    bg     = cv2.imread(frame_path)
    if avatar is None or bg is None:
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


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sample = cv2.imread(os.path.join(FRAMES_DIR, "frame_00017.jpg"))
    frame_w, frame_h = sample.shape[1], sample.shape[0]
    print(f"Frame size: {frame_w}x{frame_h}")

    for npz_name in TEST_FRAMES:
        npz_path = os.path.join(PROCESSED_DIR, npz_name)
        if not os.path.isfile(npz_path):
            print(f"Missing: {npz_path}")
            continue

        clear_scene()
        setup_world()
        setup_camera_ortho(frame_w, frame_h)
        setup_lighting()
        bpy.context.scene.render.film_transparent = True

        data = np.load(npz_path, allow_pickle=True)
        joints      = data["joints"][0]
        bbox_center = data["bbox_center"][0]
        bbox_h      = float(data["bbox_h"][0])

        armature = import_character()
        apply_vibe_pose_and_position(armature, joints, bbox_center, bbox_h, frame_w, frame_h)

        temp_path = os.path.join(OUTPUT_DIR, npz_name.replace(".npz", "_avatar.png"))
        scene = bpy.context.scene
        scene.render.engine                     = "CYCLES"
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode  = "RGBA"
        scene.render.filepath                   = temp_path
        scene.cycles.samples                    = 32
        scene.render.resolution_x              = frame_w
        scene.render.resolution_y              = frame_h
        bpy.ops.render.render(write_still=True)

        frame_name = npz_name.replace(".npz", ".jpg")
        frame_path = os.path.join(FRAMES_DIR, frame_name)
        out_path   = os.path.join(OUTPUT_DIR, npz_name.replace(".npz", "_composite.png"))

        composite_over_frame(temp_path, frame_path, out_path)
        print(f"Done: {out_path}")

    print(f"\nAll test frames saved to: {OUTPUT_DIR}")