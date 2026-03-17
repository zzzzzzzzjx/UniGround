import os
import sys
import json
import time
import argparse
import random
import glob
import numpy as np
import open3d as o3d
import re
import math
from PIL import Image, ImageDraw, ImageFont, ImageFile

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Some source candidate PNGs may be truncated; skip/harden instead of crashing a task.
ImageFile.LOAD_TRUNCATED_IMAGES = True

from inference.utils_grounding import (
    parse_response,
    calc_iou,
    encode_img,
    save_to_file,
    load_json,
    load_bboxes,
    load_scene_pcd,
)

from inference.projection_grounding import render_point_cloud_with_pytorch3d_with_objects
from openai import OpenAI


SYSTEM_INFO = "You are a helpful assistant designed to identify objects based on images and descriptions."
COOR_INFO = "The 3D spatial coordinate system is defined as follows: X-axis and Y-axis represent horizontal dimensions, Z-axis represents the vertical dimension."
ASK_INFO = "Please review the provided image(s) and object 3D spatial descriptions, then select the object ID that best matches the given description."
REASONING_INFO = (
    "Think step-by-step to yourself. When reporting, provide a concise but include both appearance cue and spatial cue. "
    "If multiple candidates appear plausible, use elimination based on appearance and spatial relations, in the global renders, each candidate object is labeled with an ID, which helps you to further locate and reason."
)
NAMING_INFO = (
    "Before choosing the target, name every candidate ID using only its stitched candidate image. "
    "Do not use global renders for naming. Use the stitched views (left-to-right) to confirm details. "
    "Avoid 'unknown' unless the object is truly unrecognizable. "
    "Do not omit any response fields."
)
RESPONSE_FORMAT = (
    "Respond in the format:\n"
    "Predicted ID: <ID>\n"
    "Explanation: <concise reasoning>\n"
    "Candidate Names: <ID -> short descriptive phrase (3-8 words: include color/material/part if visible), comma-separated>\n"
    "Target name matches?: <yes/no>\n"
    "Anchor ID: <ID or unknown>\n"
    "Relation used: <relation or none>"
)


def build_objects_info(candidates):
    lines = []
    for obj in candidates:
        bbox = obj.get("bbox_3d")
        if not bbox:
            continue
        score = obj.get("score")
        score_text = f", Score: {score:.3f}" if isinstance(score, (int, float)) else ""
        lines.append(
            "Object ID: {id}{score}, "
            "Center: X {x:.2f}, Y {y:.2f}, Z {z:.2f}, "
            "Size: W {w:.2f}, L {l:.2f}, H {h:.2f}".format(
                id=obj.get("bbox_id"),
                score=score_text,
                x=bbox[0],
                y=bbox[1],
                z=bbox[2],
                w=bbox[3],
                l=bbox[4],
                h=bbox[5],
            )
        )
    return "\n".join(lines)


def _normalize_name_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if isinstance(value, (list, tuple)):
        names = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, str):
                name = item.strip()
                if name:
                    names.append(name)
            elif isinstance(item, (int, float)):
                names.append(str(item))
            elif isinstance(item, dict):
                text = item.get("name") or item.get("anchor") or item.get("target")
                if isinstance(text, str) and text.strip():
                    names.append(text.strip())
        return names
    return [str(value)]


def extract_anchor_names(task):
    for key in ("Anchor", "anchor", "anchors", "anchor_name", "anchor_names"):
        if key in task:
            names = _normalize_name_list(task.get(key))
            if names:
                return names
    parsed = task.get("parsed_query")
    if isinstance(parsed, dict):
        for key in ("Anchor", "anchor", "anchors"):
            if key in parsed:
                names = _normalize_name_list(parsed.get(key))
                if names:
                    return names
    return []


def extract_target_name(task):
    for key in ("target_name", "target", "Target"):
        if key in task and task.get(key) is not None:
            return str(task.get(key))
    for key in ("object_name", "object"):
        if key in task and task.get(key) is not None:
            return str(task.get(key))
    parsed = task.get("parsed_query")
    if isinstance(parsed, dict):
        for key in ("Target", "target"):
            if key in parsed and parsed.get(key) is not None:
                return str(parsed.get(key))
    return None


def _format_anchor_context(target_name, anchor_names, anchor_infos):
    target_text = target_name.strip() if isinstance(target_name, str) and target_name.strip() else "unknown"
    names = []
    if anchor_names:
        names = [n for n in anchor_names if isinstance(n, str) and n.strip()]
    anchor_text = ", ".join(names) if names else "unknown"

    bbox_lines = []
    if anchor_infos:
        for idx, info in enumerate(anchor_infos, 1):
            bbox = info.get("bbox")
            if not bbox or len(bbox) < 6:
                continue
            name = info.get("anchor")
            name_text = name if isinstance(name, str) and name.strip() else f"anchor_{idx}"
            bbox_lines.append(
                "Anchor bbox {name}: Center X {x:.2f}, Y {y:.2f}, Z {z:.2f}, "
                "Size W {w:.2f}, L {l:.2f}, H {h:.2f}".format(
                    name=name_text,
                    x=bbox[0],
                    y=bbox[1],
                    z=bbox[2],
                    w=bbox[3],
                    l=bbox[4],
                    h=bbox[5],
                )
            )
    bbox_text = "\n".join(bbox_lines) if bbox_lines else "Anchor bbox: unknown (no anchor.json provided)."

    return (
        f"Target name (from task): {target_text}\n"
        f"Anchor name(s) (from task): {anchor_text}\n"
        f"{bbox_text}"
    )


def parse_query_key(item, default_idx):
    for key in ("query_id", "ann_id", "annotation_id", "ref_id"):
        if key in item:
            try:
                return int(item[key])
            except (TypeError, ValueError):
                break
    return default_idx


def resolve_task_json(task_dir):
    task_path = os.path.join(task_dir, "task.json")
    if os.path.exists(task_path):
        return task_path
    candidates = []
    for path in sorted(glob.glob(os.path.join(task_dir, "*.json"))):
        name = os.path.basename(path)
        if name in {"gt.json", "ref.json", "anchor.json"}:
            continue
        if name.startswith("anchor_"):
            continue
        candidates.append(path)
    return candidates[0] if candidates else None


def _list_image_files(image_dir):
    if not image_dir or not os.path.isdir(image_dir):
        return []
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
    return sorted([f for f in files if f.lower().endswith(exts)])


def _extract_candidate_images(candidate, image_root, images_per_id):
    items = []
    if "images" in candidate and isinstance(candidate["images"], list):
        for entry in candidate["images"][:images_per_id]:
            if isinstance(entry, str):
                items.append({"path": entry})
            elif isinstance(entry, dict):
                items.append(entry)
        return items

    if "image_paths" in candidate and isinstance(candidate["image_paths"], list):
        for entry in candidate["image_paths"][:images_per_id]:
            items.append({"path": entry})
        return items

    image_dir = candidate.get("image_dir")
    if not image_dir and image_root:
        image_dir = os.path.join(image_root, str(candidate.get("bbox_id")))

    for path in _list_image_files(image_dir)[:images_per_id]:
        items.append({"path": path})
    return items


def _bbox_3d_corners(bbox_3d):
    cx, cy, cz, w, l, h = bbox_3d
    dx = w / 2.0
    dy = l / 2.0
    dz = h / 2.0
    corners = np.array(
        [
            [cx - dx, cy - dy, cz - dz],
            [cx - dx, cy - dy, cz + dz],
            [cx - dx, cy + dy, cz - dz],
            [cx - dx, cy + dy, cz + dz],
            [cx + dx, cy - dy, cz - dz],
            [cx + dx, cy - dy, cz + dz],
            [cx + dx, cy + dy, cz - dz],
            [cx + dx, cy + dy, cz + dz],
        ],
        dtype=np.float32,
    )
    return corners


def _camera_intrinsic(camera):
    if not camera:
        return None
    if "K" in camera:
        return np.array(camera["K"], dtype=np.float32)
    intrinsic = camera.get("intrinsic")
    if isinstance(intrinsic, dict):
        fx = intrinsic.get("fx")
        fy = intrinsic.get("fy")
        cx = intrinsic.get("cx")
        cy = intrinsic.get("cy")
        if None not in (fx, fy, cx, cy):
            return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    if isinstance(intrinsic, (list, tuple)) and len(intrinsic) == 4:
        fx, fy, cx, cy = intrinsic
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    return None


def _camera_extrinsic(camera):
    if not camera:
        return None
    if "extrinsic" in camera:
        extr = np.array(camera["extrinsic"], dtype=np.float32)
        if extr.shape == (4, 4):
            return extr
    if "R" in camera and "T" in camera:
        R = np.array(camera["R"], dtype=np.float32)
        T = np.array(camera["T"], dtype=np.float32).reshape(3)
        extr = np.eye(4, dtype=np.float32)
        extr[:3, :3] = R
        extr[:3, 3] = T
        return extr
    return None


def _project_bbox_to_2d(bbox_3d, camera, image_size=None):
    if not camera:
        return None
    K = _camera_intrinsic(camera)
    extr = _camera_extrinsic(camera)
    if K is None or extr is None:
        return None

    extr_type = camera.get("extrinsic_type", "world_to_camera")
    if extr_type == "camera_to_world":
        extr = np.linalg.inv(extr)

    corners = _bbox_3d_corners(bbox_3d)
    corners_h = np.concatenate([corners, np.ones((8, 1), dtype=np.float32)], axis=1)
    cam_pts = (extr @ corners_h.T).T[:, :3]
    z = cam_pts[:, 2:3]
    valid = z.squeeze(-1) > 1e-6
    if not np.any(valid):
        return None
    cam_pts = cam_pts[valid]
    z = cam_pts[:, 2:3]
    proj = (K @ cam_pts.T).T
    proj_xy = proj[:, :2] / z
    xmin, ymin = proj_xy.min(axis=0)
    xmax, ymax = proj_xy.max(axis=0)

    if image_size:
        w, h = image_size
        xmin = float(np.clip(xmin, 0, w - 1))
        xmax = float(np.clip(xmax, 0, w - 1))
        ymin = float(np.clip(ymin, 0, h - 1))
        ymax = float(np.clip(ymax, 0, h - 1))
    return (xmin, ymin, xmax, ymax)


def _candidate_ply_path(candidate, ply_root):
    for key in ("ply_path", "ply", "pcd_path", "point_cloud_path"):
        path = candidate.get(key)
        if isinstance(path, str) and os.path.exists(path):
            return path

    if not ply_root:
        return None

    bbox_id = candidate.get("bbox_id")
    if bbox_id is None:
        return None

    ply_candidates = [
        os.path.join(ply_root, str(bbox_id), f"{bbox_id}.ply"),
        os.path.join(ply_root, str(bbox_id), "id.ply"),
        os.path.join(ply_root, f"{bbox_id}.ply"),
    ]
    for path in ply_candidates:
        if os.path.exists(path):
            return path
    return None


def _load_anchor_bboxes(task_root):
    if not task_root:
        return []
    anchor_paths = []
    anchor_path = os.path.join(task_root, "anchor.json")
    if os.path.exists(anchor_path):
        anchor_paths.append(anchor_path)
    else:
        for path in sorted(glob.glob(os.path.join(task_root, "anchor_*.json"))):
            anchor_paths.append(path)
    if not anchor_paths:
        return []
    bboxes = []
    for path in anchor_paths:
        try:
            data = load_json(path)
        except Exception:
            continue
        if isinstance(data, dict):
            bbox = data.get("bbox")
            if isinstance(bbox, list) and len(bbox) >= 6:
                bboxes.append(_transform_bbox_to_candidate(bbox[:6]))
    return bboxes


def _load_anchor_infos(task_root):
    if not task_root:
        return []
    anchor_paths = []
    anchor_path = os.path.join(task_root, "anchor.json")
    if os.path.exists(anchor_path):
        anchor_paths.append(anchor_path)
    else:
        for path in sorted(glob.glob(os.path.join(task_root, "anchor_*.json"))):
            anchor_paths.append(path)
    if not anchor_paths:
        return []
    infos = []
    for path in anchor_paths:
        try:
            data = load_json(path)
        except Exception:
            continue
        if isinstance(data, dict):
            bbox = data.get("bbox")
            if isinstance(bbox, list) and len(bbox) >= 6:
                infos.append(
                    {
                        "anchor": data.get("anchor"),
                        "bbox": _transform_bbox_to_candidate(bbox[:6]),
                    }
                )
    return infos


def _bbox_from_ply(ply_path):
    if not ply_path:
        return None
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    if points.size == 0:
        return None

    # Inverse of the Y/Z swap + Z flip applied to candidate object clouds.
    # Forward: (x, y, z) -> (x, z, -y)
    # Inverse: (x', y', z') -> (x', -z', y')
    points = points[:, [0, 2, 1]]
    points[:, 1] *= -1.0

    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)
    center = (min_xyz + max_xyz) / 2.0
    size = max_xyz - min_xyz
    return [
        float(center[0]),
        float(center[1]),
        float(center[2]),
        float(size[0]),
        float(size[1]),
        float(size[2]),
    ]


def _bbox_from_ply_raw(ply_path):
    if not ply_path:
        return None
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    if points.size == 0:
        return None
    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)
    center = (min_xyz + max_xyz) / 2.0
    size = max_xyz - min_xyz
    return [
        float(center[0]),
        float(center[1]),
        float(center[2]),
        float(size[0]),
        float(size[1]),
        float(size[2]),
    ]


def ensure_candidate_bboxes_from_ref(cand_list, scene_root):
    ref_bboxes = _load_ref_bboxes(scene_root)
    if not ref_bboxes:
        raise ValueError(f"Missing ref.json under {scene_root}")
    missing = []
    for cand in cand_list:
        obj_id = cand.get("bbox_id")
        if obj_id in ref_bboxes:
            cand["bbox_3d"] = _transform_bbox_to_candidate(ref_bboxes[obj_id])
            if not cand.get("image_dir"):
                cand["image_dir"] = os.path.join(scene_root, str(obj_id))
        else:
            missing.append(obj_id)
    if missing:
        missing_preview = ", ".join(str(mid) for mid in missing[:10])
        raise ValueError(
            "Missing bbox in ref.json for candidate ids: "
            f"{missing_preview}"
        )


def _load_gt_json(gt_json_path):
    data = load_json(gt_json_path)
    if isinstance(data, dict) and "target_id" in data and "gt_bbox" in data:
        if data["target_id"] is None:
            raise ValueError(f"target_id is null in gt.json: {gt_json_path}")
        return int(data["target_id"]), data["gt_bbox"]
    raise ValueError(f"Unsupported gt.json format in {gt_json_path}")


def _transform_bbox_to_candidate(bbox):
    # (x, y, z) -> (x, -z, y), size: (dx, dy, dz) -> (dx, dz, dy)
    x, y, z, dx, dy, dz = bbox
    return [x, -z, y, dx, dz, dy]


def _update_candidate_blob(raw_candidates, query_key, query_idx, cand_list):
    if isinstance(raw_candidates, dict):
        if query_key in raw_candidates:
            raw_candidates[query_key] = cand_list
        elif str(query_key) in raw_candidates:
            raw_candidates[str(query_key)] = cand_list
        elif query_idx in raw_candidates:
            raw_candidates[query_idx] = cand_list
        else:
            raw_candidates[str(query_idx)] = cand_list
        return raw_candidates

    if isinstance(raw_candidates, list):
        if not raw_candidates:
            return cand_list
        first = raw_candidates[0]
        if isinstance(first, dict) and "candidates" in first:
            updated = False
            for item in raw_candidates:
                item_key = parse_query_key(item, None)
                if item_key == query_key:
                    item["candidates"] = cand_list
                    updated = True
                    break
            if not updated and 0 <= query_idx < len(raw_candidates):
                raw_candidates[query_idx]["candidates"] = cand_list
            return raw_candidates
        if isinstance(first, list):
            if 0 <= query_idx < len(raw_candidates):
                raw_candidates[query_idx] = cand_list
            return raw_candidates
        if isinstance(first, dict) and "bbox_id" in first:
            return cand_list

    return raw_candidates


def _write_candidate_file(candidate_file, raw_candidates):
    if not candidate_file:
        return
    os.makedirs(os.path.dirname(candidate_file), exist_ok=True)
    with open(candidate_file, "w", encoding="utf-8") as f:
        json.dump(raw_candidates, f, indent=4)


def _load_scores(sim_prob_path):
    if not sim_prob_path or not os.path.exists(sim_prob_path):
        return {}
    scores = {}
    pattern = re.compile(r"\((?:[^,]*),\s*([0-9.]+),\s*(\d+)\)")
    with open(sim_prob_path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                score = float(match.group(1))
                obj_id = int(match.group(2))
                scores[obj_id] = score
    return scores


def _load_similarity_json(sim_path):
    if not sim_path or not os.path.exists(sim_path):
        return {}
    data = load_json(sim_path)
    scores = {}
    if isinstance(data, dict):
        for key, value in data.items():
            try:
                obj_id = int(key)
            except (TypeError, ValueError):
                continue
            if isinstance(value, dict) and "similarity" in value:
                scores[obj_id] = float(value["similarity"])
    return scores


def _load_ref_bboxes(scene_root):
    ref_path = os.path.join(scene_root, "ref.json")
    if not os.path.exists(ref_path):
        return {}
    data = load_json(ref_path)
    bboxes = {}
    if isinstance(data, dict):
        for key, value in data.items():
            try:
                obj_id = int(key)
            except (TypeError, ValueError):
                continue
            if isinstance(value, dict):
                bbox = value.get("bbox")
                if isinstance(bbox, list) and len(bbox) >= 6:
                    bboxes[obj_id] = bbox[:6]
    return bboxes


def _build_candidates_from_ref(scene_root):
    ref_bboxes = _load_ref_bboxes(scene_root)
    if not ref_bboxes:
        return []
    scores = _load_scores(os.path.join(scene_root, "sim_prob.txt"))
    if not scores:
        scores = _load_similarity_json(os.path.join(scene_root, "ref.json"))
    candidates = []
    for obj_id in sorted(ref_bboxes):
        image_dir = os.path.join(scene_root, str(obj_id))
        candidates.append(
            {
                "bbox_id": obj_id,
                "bbox_3d": _transform_bbox_to_candidate(ref_bboxes[obj_id]),
                "score": scores.get(obj_id),
                "image_dir": image_dir if os.path.isdir(image_dir) else None,
            }
        )
    return candidates


def _build_candidates_from_id_ply(scene_root):
    return _build_candidates_from_ref(scene_root)


def _find_full_pcd_path(scene_root):
    if not scene_root:
        return None
    candidates = [
        os.path.join(scene_root, "full_pcd.ply"),
        os.path.join(scene_root, "data", "full_pcd.ply"),
        os.path.join(scene_root, "results", "full_pcd.ply"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def load_scene_pcd_from_ply(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    if points.size == 0:
        raise ValueError(f"No points found in {ply_path}")

    colors = np.asarray(pcd.colors)
    if colors.size == 0:
        colors = np.ones_like(points) * 0.5

    scan_pc = np.concatenate((points, colors), axis=1).astype("float32")
    center = np.mean(points, axis=0)
    return scan_pc, center


def _load_label_font(font_size):
    if font_size <= 0:
        return ImageFont.load_default()
    for font_path in (
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, font_size)
    return ImageFont.load_default()


def _label_font_for_image(image):
    min_dim = min(image.size)
    font_size = int(min(max(12, min_dim * 0.06), 48))
    return _load_label_font(font_size)


def annotate_candidate_image(src_path, dst_path, label_text, bbox_2d=None):
    image = Image.open(src_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _label_font_for_image(image)

    if bbox_2d:
        draw.rectangle(bbox_2d, outline=(255, 0, 0), width=2)

    text = f"ID {label_text}"
    text_size = draw.textbbox((0, 0), text, font=font)
    padding = 2
    box = (
        2,
        2,
        2 + (text_size[2] - text_size[0]) + padding * 2,
        2 + (text_size[3] - text_size[1]) + padding * 2,
    )
    draw.rectangle(box, fill=(255, 0, 0))
    draw.text((2 + padding, 2 + padding), text, fill=(255, 255, 255), font=font)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    image.save(dst_path)
    image.close()
    return dst_path


def _draw_bbox_on_image(image, bbox_2d):
    if not bbox_2d:
        return image
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox_2d, outline=(255, 0, 0), width=2)
    return image


def _make_candidate_collage(items, dst_path, label_text, compress=True, quality=85, max_size=None):
    """
    Create a collage of candidate images.

    Args:
        items: List of image items with paths
        dst_path: Destination path for the collage
        label_text: Text label to draw on the collage
        compress: Whether to compress the image (default: True)
        quality: JPEG quality (1-100, default: 85)
        max_size: Maximum dimension (width or height) for the collage (default: None)
    """
    images = []
    try:
        for item in items:
            try:
                img = Image.open(item["path"]).convert("RGB")
            except Exception as exc:
                print(f"[WARN] Skip broken candidate image: {item.get('path')} ({exc})")
                continue
            _draw_bbox_on_image(img, item.get("bbox_2d"))
            images.append(img)
        if not images:
            return None

        max_height = max(img.height for img in images)
        resized = []
        total_width = 0
        for img in images:
            if img.height != max_height:
                scale = max_height / img.height
                new_size = (int(img.width * scale), max_height)
                resized_img = img.resize(new_size, Image.BILINEAR)
                img.close()
                img = resized_img
            resized.append(img)
            total_width += img.width

        images = resized
        collage = Image.new("RGB", (total_width, max_height), (0, 0, 0))
        x_offset = 0
        for img in images:
            collage.paste(img, (x_offset, 0))
            x_offset += img.width

        draw = ImageDraw.Draw(collage)
        font = _label_font_for_image(collage)
        text = f"ID {label_text}"
        text_size = draw.textbbox((0, 0), text, font=font)
        padding = 2
        box = (
            2,
            2,
            2 + (text_size[2] - text_size[0]) + padding * 2,
            2 + (text_size[3] - text_size[1]) + padding * 2,
        )
        draw.rectangle(box, fill=(255, 0, 0))
        draw.text((2 + padding, 2 + padding), text, fill=(255, 255, 255), font=font)

        # Apply size limit if specified
        if max_size and (collage.width > max_size or collage.height > max_size):
            if collage.width > collage.height:
                new_width = max_size
                new_height = int(collage.height * max_size / collage.width)
            else:
                new_height = max_size
                new_width = int(collage.width * max_size / collage.height)
            collage = collage.resize((new_width, new_height), Image.LANCZOS)

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        # Save with compression if enabled
        if compress:
            # Change extension to .jpg for compressed images
            if dst_path.endswith('.png'):
                dst_path = dst_path[:-4] + '.jpg'
            collage.save(dst_path, format='JPEG', quality=quality, optimize=True)
        else:
            collage.save(dst_path)

        collage.close()
        return dst_path
    finally:
        for img in images:
            img.close()


def _make_global_collage(image_paths, dst_path, compress=True, quality=85, max_size=None):
    """
    Create a collage of global view images.

    Args:
        image_paths: List of image paths
        dst_path: Destination path for the collage
        compress: Whether to compress the image (default: True)
        quality: JPEG quality (1-100, default: 85)
        max_size: Maximum dimension (width or height) for the collage (default: None)
    """
    images = []
    try:
        for path in image_paths:
            if not path or not os.path.exists(path):
                continue
            try:
                img = Image.open(path).convert("RGB")
            except Exception as exc:
                print(f"[WARN] Skip broken global image: {path} ({exc})")
                continue
            images.append(img)
        if not images:
            return None

        max_height = max(img.height for img in images)
        resized = []
        total_width = 0
        for img in images:
            if img.height != max_height:
                scale = max_height / img.height
                new_size = (int(img.width * scale), max_height)
                resized_img = img.resize(new_size, Image.BILINEAR)
                img.close()
                img = resized_img
            resized.append(img)
            total_width += img.width

        collage = Image.new("RGB", (total_width, max_height), (0, 0, 0))
        x_offset = 0
        for img in resized:
            collage.paste(img, (x_offset, 0))
            x_offset += img.width

        # Apply size limit if specified
        if max_size and (collage.width > max_size or collage.height > max_size):
            if collage.width > collage.height:
                new_width = max_size
                new_height = int(collage.height * max_size / collage.width)
            else:
                new_height = max_size
                new_width = int(collage.width * max_size / collage.height)
            collage = collage.resize((new_width, new_height), Image.LANCZOS)

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        # Save with compression if enabled
        if compress:
            # Change extension to .jpg for compressed images
            if dst_path.endswith('.png'):
                dst_path = dst_path[:-4] + '.jpg'
            collage.save(dst_path, format='JPEG', quality=quality, optimize=True)
        else:
            collage.save(dst_path)

        collage.close()
        return dst_path
    finally:
        for img in images:
            img.close()


def create_openai_messages(
    query,
    objects_info,
    use_image=False,
    global_images=None,
    candidate_images=None,
    target_name=None,
    anchor_names=None,
    anchor_infos=None,
    use_simple_prompt=False,
):
    """Create OpenAI API messages for grounding task.

    Args:
        use_simple_prompt: If True, use simple prompt for ablation study.
                          If False, use optimized structured prompt (default).
    """

    # 准备 anchor_context (两种模式都可能需要)
    anchor_context = _format_anchor_context(target_name, anchor_names, anchor_infos)

    if use_simple_prompt:
        # ========== 简单 Prompt (用于消融实验) ==========
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Identify the object ID that matches the description."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Query: {query}\n\nCandidate Objects:\n{objects_info}\n\nPlease identify which object ID best matches the query and explain why.",
                    }
                ],
            },
        ]
    else:
        # ========== 优化 Prompt (默认) ==========
        # 当 use_image=False 时，不应该提到图像
        ask_info_no_image = "Please review the object 3D spatial descriptions, then select the object ID that best matches the given description."
        reasoning_info_no_image = (
            "Think step-by-step to yourself. When reporting, provide a concise but include both appearance cue and spatial cue. "
            "If multiple candidates appear plausible, use elimination based on appearance and spatial relations."
        )

        messages = [
            {"role": "system", "content": SYSTEM_INFO},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{COOR_INFO}\n\n{anchor_context}\n\nObject IDs and their positions:\n{objects_info}\n\n{ask_info_no_image}\n{reasoning_info_no_image}\n\n{RESPONSE_FORMAT}\n\nThe given description is: {query}",
                    }
                ],
            },
        ]

    if use_image:
        content = []
        candidate_groups = {}

        if global_images:
            for image_path in global_images:
                image_url = encode_img(image_path)
                content.append({"type": "image_url", "image_url": {"url": image_url}})

        if candidate_images:
            for item in candidate_images:
                if isinstance(item, dict):
                    path = item.get("path")
                    bbox_id = item.get("bbox_id")
                    count = item.get("count", 1)
                else:
                    path = item
                    bbox_id = None
                    count = 1
                if not path:
                    continue
                candidate_groups.setdefault(bbox_id, []).append(
                    {"path": path, "count": count}
                )

            for bbox_id in sorted(candidate_groups, key=lambda v: (v is None, v)):
                entries = candidate_groups[bbox_id]
                label = "unknown" if bbox_id is None else str(bbox_id)
                for entry in entries:
                    view_count = entry.get("count", 1)
                    # 只在优化 Prompt 模式下添加候选图像的文字说明
                    if not use_simple_prompt:
                        note = (
                            f"Candidate ID {label} stitched image (up to {view_count} largest views, "
                            "left-to-right). Use this to name the object."
                        )
                        content.append({"type": "text", "text": note})
                    image_url = encode_img(entry["path"])
                    content.append(
                        {"type": "image_url", "image_url": {"url": image_url}}
                    )

        if content:
            if use_simple_prompt:
                # ========== 简单 Prompt + 图像 ==========
                # 只添加基本的图像说明，不添加复杂的推理指令
                if global_images and candidate_groups:
                    image_note = (
                        "The first image shows the global scene with object IDs. "
                        "Following images show individual candidate objects."
                    )
                elif global_images:
                    image_note = "The image shows the global scene with object IDs."
                else:
                    image_note = "The images show individual candidate objects."

                content.append(
                    {
                        "type": "text",
                        "text": f"{image_note}\n\nObject IDs and their information:\n{objects_info}\n\nThe given description is: {query}",
                    }
                )
            else:
                # ========== 优化 Prompt + 图像 (原有逻辑) ==========
                # 根据是否有全局图/候选图，动态调整 note 内容
                if global_images:
                    anchor_note = (
                        "Global renders may include a red triangle marking the anchor bbox center (no ID). "
                        "If anchor bbox is unknown, there will be no marker. Use it to reason about spatial relations when relevant."
                    )
                    axis_note = (
                        "Axes overlay (if present): X is red, Y is green, Z is blue. "
                        "Arrows originate at the anchor center to visualize the coordinate directions. "
                        "Direction mapping: left = +X (X larger -> more left), right = -X (X smaller -> more right). "
                        "If A is to the right of B, then X_A < X_B. If A is to the left of B, then X_A > X_B. "
                        "Front/back mapping: front = -Y (Y smaller -> more front), back = +Y (Y larger -> more back). "
                        "If A is in front of B, then Y_A < Y_B. If A is behind B, then Y_A > Y_B. "
                        "Up/down mapping: up = +Z (Z larger -> more up), down = -Z (Z smaller -> more down). "
                        "If A is above B, then Z_A > Z_B. If A is below B, then Z_A < Z_B."
                    )
                else:
                    anchor_note = ""
                    axis_note = ""

                if candidate_groups:
                    crop_note = (
                        "Candidate stitched images include a red box around the object in each view. "
                        "Each stitched image places up to three different views left-to-right."
                    )
                else:
                    crop_note = ""
                if global_images and candidate_groups:
                    image_note = (
                        "The first image is a stitched global render (multiple views side-by-side) "
                        "with object IDs and box for spatial context. Then follow stitched candidate images "
                        "grouped by ID."
                    )
                elif global_images:
                    image_note = (
                        "The image is a stitched global render (multiple views side-by-side) "
                        "with object IDs for spatial context. "
                        "Use the 3D coordinates to reason about relative position and size."
                    )
                else:
                    image_note = (
                        "The images are stitched candidate views grouped by ID."
                    )

                if candidate_groups:
                    group_lines = []
                    id_list = []
                    for bbox_id in sorted(candidate_groups, key=lambda v: (v is None, v)):
                        label = "unknown" if bbox_id is None else str(bbox_id)
                        id_list.append(label)
                        total_views = sum(
                            entry.get("count", 1) for entry in candidate_groups[bbox_id]
                        )
                        line = (
                            f"Candidate ID {label}: stitched image with up to {total_views} views "
                            "(left-to-right)."
                        )
                        group_lines.append(line)
                    group_text = "\n".join(group_lines)

                    # 根据是否有全局图，调整 decision_note 和 fallback_note
                    if global_images:
                        decision_note = (
                            "Decision protocol: identify each candidate by visual attributes "
                            "(color/material/shape) from stitched candidate images first, then verify spatial relations "
                            "with global renders and 3D coordinates. If there is conflict, prefer the visual "
                            "match when the description is appearance-based."
                        )
                        fallback_note = (
                            "Fallback rule: if the target name is missing from Candidate Names, or appears for "
                            "multiple IDs, extract the anchor object from the query and use spatial relations "
                            "(e.g., 'next to', 'left of') to choose the target. Locate the anchor ID via Candidate "
                            "Names, then pick the candidate whose 3D bbox best matches the relation; confirm with "
                            "the global renders."
                        )
                    else:
                        decision_note = (
                            "Decision protocol: identify each candidate by visual attributes "
                            "(color/material/shape) from stitched candidate images first, then verify spatial relations "
                            "with 3D coordinates. If there is conflict, prefer the visual "
                            "match when the description is appearance-based."
                        )
                        fallback_note = (
                            "Fallback rule: if the target name is missing from Candidate Names, or appears for "
                            "multiple IDs, extract the anchor object from the query and use spatial relations "
                            "(e.g., 'next to', 'left of') to choose the target. Locate the anchor ID via Candidate "
                            "Names, then pick the candidate whose 3D bbox best matches the relation; confirm with "
                            "the 3D coordinates."
                        )

                    anchor_name_note = (
                        "Anchor hint: use the Anchor name(s) above to identify the anchor ID if possible. "
                        "If the anchor is a room/area rather than a single object, use it only as a coarse spatial cue."
                    )
                    naming_note = (
                        "You must provide Candidate Names for every ID in this list: "
                        + ", ".join(id_list)
                        + ". Use 'unknown' if unclear."
                    )
                else:
                    group_text = ""
                    decision_note = ""
                    naming_note = ""
                    anchor_name_note = ""
                    fallback_note = ""

                # 根据图像情况动态调整常量内容
                if candidate_groups:
                    naming_info_text = NAMING_INFO
                else:
                    naming_info_text = ""

                if global_images:
                    reasoning_info_text = REASONING_INFO
                else:
                    # 无全局图时，移除关于 global renders 的说明
                    reasoning_info_text = (
                        "Think step-by-step to yourself. When reporting, provide a concise but include both appearance cue and spatial cue. "
                        "If multiple candidates appear plausible, use elimination based on appearance and spatial relations."
                    )

                if global_images or candidate_groups:
                    ask_info_text = ASK_INFO
                else:
                    # 无图像时，不提 "image(s)"
                    ask_info_text = "Please review the object 3D spatial descriptions, then select the object ID that best matches the given description."

                content.append(
                    {
                        "type": "text",
                        "text": f"{image_note}\n{anchor_note}\n{axis_note}\n{crop_note}\n{group_text}\n{decision_note}\n{fallback_note}\n{anchor_name_note}\n{naming_info_text}\n{naming_note}\n\n{anchor_context}\n\nObject IDs and their 3D spatial information are as follows:\n{objects_info}\n\n{COOR_INFO}\n\n{ask_info_text}\n{reasoning_info_text}\n\n{RESPONSE_FORMAT}\n\nThe given description is: {query}",
                    }
                )
            # 只有当 content 不为空时才替换消息内容
            messages[1]["content"] = content

    return messages


def process_query(
    query,
    objects_info,
    openai_api_key,
    openai_api_base,
    use_image=False,
    global_images=None,
    candidate_images=None,
    model_name="Qwen2-VL-72B-Instruct",
    target_name=None,
    anchor_names=None,
    anchor_infos=None,
    log_file=None,
    max_retries=5,
    base_backoff=2,
    use_simple_prompt=False,
):
    assert objects_info is not None
    assert query is not None

    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base, timeout=300.0)
    messages = create_openai_messages(
        query,
        objects_info,
        use_image=use_image,
        global_images=global_images,
        candidate_images=candidate_images,
        target_name=target_name,
        anchor_names=anchor_names,
        anchor_infos=anchor_infos,
        use_simple_prompt=use_simple_prompt,
    )

    if log_file and not os.path.exists(log_file):
        save_to_file(log_file, json.dumps(messages, ensure_ascii=False, indent=2))

    for attempt in range(max_retries):
        try:
            print(f"Sending API request (attempt {attempt + 1}/{max_retries})...")
            chat_response = client.chat.completions.create(
                model=model_name, messages=messages, timeout=300.0
            )
            result = chat_response.choices[0].message.content
            if log_file:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"\n\n=== API RESPONSE ===\n{result}\n")
            print(f"API request successful, response length: {len(result)} chars")
            return result.replace("\\n", "\n")
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            is_retryable = status in {429, 500, 502, 503, 504} or (
                status is None
                and any(code in str(exc) for code in ("429", "500", "502", "503", "504"))
            )
            if not is_retryable or attempt == max_retries - 1:
                raise
            sleep_s = (base_backoff ** attempt) + random.random()
            print(f"OpenAI request failed (status {status}), retrying in {sleep_s:.1f}s...")
            time.sleep(sleep_s)


def load_candidates(cand_file):
    data = load_json(cand_file)

    if isinstance(data, dict):
        return {int(k): v for k, v in data.items()}

    if isinstance(data, list):
        if not data:
            return {}
        first = data[0]
        if isinstance(first, dict) and "candidates" in first:
            return {
                int(item.get("query_id", idx)): item["candidates"]
                for idx, item in enumerate(data)
            }
        if isinstance(first, list):
            return {idx: item for idx, item in enumerate(data)}
        if isinstance(first, dict) and "bbox_id" in first:
            return {0: data}

    raise ValueError(f"Unsupported candidate format in {cand_file}")


def prepare_candidate_images(
    cand_list,
    image_root,
    output_dir,
    room,
    query_idx,
    images_per_id,
    max_total,
    use_projection=False,
    collage=False,
    primary_smallest=False,
    primary_only=False,
    compress_collage=True,
    collage_quality=85,
    collage_max_size=None,
):
    labeled_paths = []
    if images_per_id <= 0:
        return labeled_paths

    for cand in cand_list:
        image_items = _extract_candidate_images(cand, image_root, images_per_id)
        if not image_items:
            continue
        processed_items = []
        for item in image_items:
            src_path = item.get("path")
            if not src_path or not os.path.exists(src_path):
                continue
            try:
                with Image.open(src_path) as im:
                    width, height = im.size
            except Exception:
                continue

            bbox_2d = None
            if use_projection and cand.get("bbox_3d") and item.get("camera"):
                bbox_2d = _project_bbox_to_2d(
                    cand["bbox_3d"],
                    item["camera"],
                    image_size=item.get("image_size"),
                )
            processed_items.append(
                {
                    "path": src_path,
                    "bbox_2d": bbox_2d,
                    "size": (width, height),
                    "area": width * height,
                }
            )

        if not processed_items:
            continue

        if collage:
            out_path = os.path.join(
                output_dir,
                "render",
                room,
                str(query_idx),
                "candidates",
                f"{cand.get('bbox_id')}_collage.png",
            )
            collage_items = sorted(
                processed_items, key=lambda x: x.get("area", 0), reverse=True
            )[:3]
            collage_path = _make_candidate_collage(
                collage_items,
                out_path,
                cand.get("bbox_id"),
                compress=compress_collage,
                quality=collage_quality,
                max_size=collage_max_size,
            )
            if collage_path:
                labeled_paths.append(
                    {
                        "path": collage_path,
                        "bbox_id": cand.get("bbox_id"),
                        "count": len(collage_items) if collage_items else 1,
                        "role": "primary",
                    }
                )
        else:
            primary_item = None
            if primary_smallest or primary_only:
                primary_item = min(processed_items, key=lambda x: x["area"])
                processed_items = [item for item in processed_items if item is not primary_item]
            if primary_only and primary_item is None and processed_items:
                primary_item = processed_items.pop(0)

            if primary_item is not None:
                out_path = os.path.join(
                    output_dir,
                    "render",
                    room,
                    str(query_idx),
                    "candidates",
                    f"{cand.get('bbox_id')}_primary.png",
                )
                labeled_paths.append(
                    {
                        "path": annotate_candidate_image(
                            primary_item["path"],
                            out_path,
                            cand.get("bbox_id"),
                            primary_item.get("bbox_2d"),
                        ),
                        "bbox_id": cand.get("bbox_id"),
                        "count": 1,
                        "role": "primary",
                    }
                )
                if max_total > 0 and len(labeled_paths) >= max_total:
                    return labeled_paths
                if primary_only:
                    continue

            for img_idx, item in enumerate(processed_items):
                out_path = os.path.join(
                    output_dir,
                    "render",
                    room,
                    str(query_idx),
                    "candidates",
                    f"{cand.get('bbox_id')}_{img_idx}.png",
                )
                labeled_paths.append(
                    {
                        "path": annotate_candidate_image(
                            item["path"],
                            out_path,
                            cand.get("bbox_id"),
                            item.get("bbox_2d"),
                        ),
                        "bbox_id": cand.get("bbox_id"),
                        "count": 1,
                        "role": "aux" if primary_item is not None else "primary",
                    }
                )

                if max_total > 0 and len(labeled_paths) >= max_total:
                    return labeled_paths

        if max_total > 0 and len(labeled_paths) >= max_total:
            return labeled_paths
    return labeled_paths


def process_room(
    dataset,
    room,
    pcd_dir,
    output_dir,
    scene_json_path,
    candidate_file,
    gt_json_path,
    openai_api_key,
    openai_api_base,
    use_image=False,
    model_name=None,
    image_size=680,
    request_delay=0.0,
    max_candidates=0,
    sort_by_score=False,
    anchor_mode="mean",
    num_global_views=1,
    global_view_azims=None,
    candidate_image_root=None,
    candidate_image_per_id=0,
    max_candidate_images=0,
    candidate_image_use_projection=False,
    candidate_image_collage=False,
    candidate_image_primary_smallest=False,
    candidate_image_primary_only=False,
    camera_distance_factor=1.0,
    camera_lift=1.5,
    camera_elev=0.0,
    min_camera_distance=0.0,
    min_camera_height=None,
    fixed_camera_center=None,
    fixed_focal_length=None,
    fixed_principal_point=None,
    fixed_camera_params=None,
    adaptive_point_radius=False,
    save_camera_params=False,
    top_down_view=True,
    top_down_tilt=15.0,
    ensure_all_visible=True,
    write_candidate_bboxes=True,
    use_simple_prompt=False,
    filter_ceiling_points=False,
    ceiling_percentile=95.0,
    compress_collage=True,
    collage_quality=85,
    collage_max_size=0,
):
    scene_data = load_json(scene_json_path)
    if isinstance(scene_data, dict):
        scene_data = [scene_data]
    if not scene_data:
        raise ValueError("scene_json is empty.")
    first_scene = scene_data[0]
    room = (
        first_scene.get("scene_id")
        or first_scene.get("scan_id")
        or room
    )
    queries = [
        {
            "scan_id": item.get("scene_id") or item.get("scan_id") or room,
            "caption": item.get("description") or item.get("caption", ""),
            "target_name": extract_target_name(item),
            "anchor_names": extract_anchor_names(item),
        }
        for item in scene_data
        if (item.get("scene_id") or item.get("scan_id")) == room
    ]
    if not gt_json_path:
        raise ValueError("gt_json is required.")
    try:
        gt_target_id, gt_bbox = _load_gt_json(gt_json_path)
    except ValueError as exc:
        print(str(exc))
        return
    gt_bbox = _transform_bbox_to_candidate(gt_bbox)

    if not candidate_file or not os.path.exists(candidate_file):
        scene_root = candidate_image_root or os.path.dirname(candidate_file)
        cand_list = _build_candidates_from_ref(scene_root)
        if not cand_list:
            print(f"Missing ref.json under {scene_root}, skipping")
            return
        raw_candidates = cand_list
        candidates_by_query = {0: cand_list}
    else:
        raw_candidates = load_json(candidate_file)
        candidates_by_query = load_candidates(candidate_file)

    output_file = os.path.join(output_dir, "pred", f"{room}.json")
    if os.path.exists(output_file):
        print(f"File {output_file} already exists, skipping")
        return
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    log_file = os.path.join(output_dir, "room_info", f"{room}.txt")
    print(f"Saved objects_info to {log_file}")

    correct_25 = 0
    correct_50 = 0
    total_predictions = 0
    acc_25 = 0.0
    acc_50 = 0.0
    results = []

    queries = sorted(queries, key=lambda x: x.get("scan_id", ""))

    view_azims = []
    if global_view_azims:
        view_azims = [float(v) for v in global_view_azims.split(",") if v.strip()]
    elif num_global_views > 1:
        step = 360.0 / num_global_views
        view_azims = [i * step for i in range(num_global_views)]
    elif num_global_views == 1:
        view_azims = [0.0]
    else:
        # num_global_views == 0: 不生成全局图像
        view_azims = []

    for i, d in enumerate(queries):
        query = d["caption"]
        target_name = d.get("target_name")
        anchor_names = d.get("anchor_names")
        gt_id = gt_target_id

        query_key = parse_query_key(d, i)
        cand_list = candidates_by_query.get(query_key) or candidates_by_query.get(i)
        if cand_list is None:
            print(f"No candidates for {room} query {i}, skipping")
            continue

        if sort_by_score:
            cand_list = sorted(
                cand_list, key=lambda x: x.get("score", 0.0), reverse=True
            )

        if max_candidates > 0:
            cand_list = cand_list[:max_candidates]

        if not cand_list:
            print(f"Empty candidate list for {room} query {i}, skipping")
            continue

        scene_root = candidate_image_root or os.path.dirname(candidate_file)
        ensure_candidate_bboxes_from_ref(cand_list, scene_root)
        raw_candidates = _update_candidate_blob(raw_candidates, query_key, i, cand_list)

        objects_info = build_objects_info(cand_list)

        if anchor_mode == "top1":
            anchors = cand_list[:1]
        else:
            anchors = cand_list

        targets = cand_list

        camera_distance_factor_local = camera_distance_factor
        camera_lift_local = camera_lift
        camera_elev_local = camera_elev
        min_camera_distance_local = min_camera_distance
        min_camera_height_local = min_camera_height
        top_down_view_local = top_down_view
        top_down_tilt_local = top_down_tilt
        ensure_all_visible_local = ensure_all_visible
        fixed_camera_center_local = fixed_camera_center
        fixed_focal_length_local = fixed_focal_length
        fixed_principal_point_local = fixed_principal_point
        save_camera_params_local = save_camera_params

        if fixed_camera_params:
            if "camera_center" in fixed_camera_params:
                fixed_center = fixed_camera_params["camera_center"]
                fixed_camera_center_local = (
                    [float(v) for v in fixed_center] if fixed_center else None
                )
            if "camera_distance_factor" in fixed_camera_params:
                camera_distance_factor_local = fixed_camera_params[
                    "camera_distance_factor"
                ]
            if "camera_lift" in fixed_camera_params:
                camera_lift_local = fixed_camera_params["camera_lift"]
            if "camera_elev" in fixed_camera_params:
                camera_elev_local = fixed_camera_params["camera_elev"]
            if "top_down_view" in fixed_camera_params:
                top_down_view_local = fixed_camera_params["top_down_view"]
            if "top_down_tilt" in fixed_camera_params:
                top_down_tilt_local = fixed_camera_params["top_down_tilt"]
            if "min_camera_distance" in fixed_camera_params:
                min_camera_distance_local = fixed_camera_params["min_camera_distance"]
            if "min_camera_height" in fixed_camera_params:
                min_camera_height_local = fixed_camera_params["min_camera_height"]
            if "fixed_focal_length" in fixed_camera_params:
                fixed_focal_length_local = (
                    None
                    if fixed_camera_params["fixed_focal_length"] is None
                    else [float(v) for v in fixed_camera_params["fixed_focal_length"]]
                )
            if "focal_length" in fixed_camera_params:
                fixed_focal_length_local = (
                    None
                    if fixed_camera_params["focal_length"] is None
                    else [float(v) for v in fixed_camera_params["focal_length"]]
                )
            if "fixed_principal_point" in fixed_camera_params:
                fixed_principal_point_local = (
                    None
                    if fixed_camera_params["fixed_principal_point"] is None
                    else [
                        float(v)
                        for v in fixed_camera_params["fixed_principal_point"]
                    ]
                )
            if "principal_point" in fixed_camera_params:
                fixed_principal_point_local = (
                    None
                    if fixed_camera_params["principal_point"] is None
                    else [float(v) for v in fixed_camera_params["principal_point"]]
                )
            if "ensure_all_visible" in fixed_camera_params:
                ensure_all_visible_local = fixed_camera_params["ensure_all_visible"]
            save_camera_params_local = True

        scene_root = candidate_image_root or os.path.dirname(candidate_file)
        anchor_infos = _load_anchor_infos(scene_root)
        if not anchor_names and anchor_infos:
            anchor_names = [
                info.get("anchor")
                for info in anchor_infos
                if isinstance(info.get("anchor"), str) and info.get("anchor").strip()
            ]

        global_images = []
        candidate_images = []
        if use_image:
            full_pcd_path = _find_full_pcd_path(scene_root)
            if full_pcd_path:
                scan_pc, center = load_scene_pcd_from_ply(full_pcd_path)
            else:
                scan_pc, center = load_scene_pcd(room, pcd_dir)
            anchor_bboxes = _load_anchor_bboxes(scene_root)
            for view_idx, azim in enumerate(view_azims):
                view_dir = os.path.join(
                    output_dir, "render", room, str(i), f"view_{view_idx}"
                )
                render_fixed_center = fixed_camera_center_local
                if render_fixed_center is not None:
                    render_fixed_center = np.array(
                        render_fixed_center, dtype=np.float32
                    )
                image_path = render_point_cloud_with_pytorch3d_with_objects(
                    targets,
                    targets,
                    anchors,
                    center,
                    scan_pc,
                    save_dir=view_dir,
                    image_size=image_size,
                    draw_id=True,
                    draw_img=True,
                    camera_azim=azim,
                    camera_elev=camera_elev_local,
                    camera_distance_factor=camera_distance_factor_local,
                    camera_lift=camera_lift_local,
                    top_down_view=top_down_view_local,
                    top_down_tilt=top_down_tilt_local,
                    ensure_all_visible=ensure_all_visible_local,
                    min_camera_distance=min_camera_distance_local,
                    min_camera_height=min_camera_height_local,
                    fixed_camera_center=render_fixed_center,
                    fixed_focal_length=fixed_focal_length_local,
                    fixed_principal_point=fixed_principal_point_local,
                    adaptive_point_radius=adaptive_point_radius,
                    save_camera_params=save_camera_params_local,
                    anchor_bboxes=anchor_bboxes,
                    filter_ceiling_points=filter_ceiling_points,
                    ceiling_percentile=ceiling_percentile,
                )
                global_images.append(image_path)

            candidate_images = prepare_candidate_images(
                cand_list,
                candidate_image_root or os.path.dirname(candidate_file),
                output_dir,
                room,
                i,
                candidate_image_per_id,
                max_candidate_images,
                use_projection=candidate_image_use_projection,
                collage=candidate_image_collage,
                primary_smallest=candidate_image_primary_smallest,
                primary_only=candidate_image_primary_only,
                compress_collage=compress_collage,
                collage_quality=collage_quality,
                collage_max_size=collage_max_size if collage_max_size > 0 else None,
            )
            if global_images:
                print(f"Rendered images: {', '.join(global_images)}")
            if global_images and len(global_images) > 1:
                collage_path = os.path.join(
                    output_dir, "render", room, str(i), "global_collage.png"
                )
                stitched = _make_global_collage(
                    global_images,
                    collage_path,
                    compress=False,
                    quality=100,
                    max_size=None,
                )
                if stitched:
                    global_images = [stitched]

        if request_delay > 0:
            time.sleep(request_delay)

        try:
            response = process_query(
                query,
                objects_info,
                openai_api_key,
                openai_api_base,
                use_image,
                global_images,
                candidate_images,
                model_name,
                target_name=target_name,
                anchor_names=anchor_names,
                anchor_infos=anchor_infos,
                log_file=log_file,
                use_simple_prompt=use_simple_prompt,
            )
            (
                predicted_id,
                explanation,
                candidate_names,
                target_name_matches,
                anchor_id,
                relation_used,
            ) = parse_response(response)
        except Exception as e:
            print(f"Error processing query '{query}': {type(e).__name__}: {str(e)}")
            if log_file:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"\n\n=== ERROR ===\n{type(e).__name__}: {str(e)}\n")
            predicted_id = None
            explanation = None
            candidate_names = None
            target_name_matches = None
            anchor_id = None
            relation_used = None

        pred_bbox = None
        for obj in cand_list:
            if int(obj["bbox_id"]) == predicted_id:
                pred_bbox = obj
                break

        iou = 0
        if pred_bbox is not None:
            try:
                iou = calc_iou(gt_bbox, pred_bbox["bbox_3d"])
            except Exception:
                iou = 0

        if iou >= 0.25:
            correct_25 += 1
        if iou >= 0.5:
            correct_50 += 1
        total_predictions += 1

        results.append(
            {
                "query": query,
                "gt_id": gt_target_id if gt_target_id is not None else gt_id,
                "predicted_id": predicted_id,
                "pred_bbox": pred_bbox["bbox_3d"] if pred_bbox else None,
                "gt_bbox": gt_bbox,
                "unique": d.get("unique"),
                "explanation": explanation,
                "candidate_names": candidate_names,
                "target_name_matches": target_name_matches,
                "anchor_id": anchor_id,
                "relation_used": relation_used,
            }
        )

        acc_25 = correct_25 / total_predictions
        acc_50 = correct_50 / total_predictions
        print(f"Accuracy@0.25: {acc_25:.4f} | Accuracy@0.50: {acc_50:.4f}")
    if write_candidate_bboxes:
        _write_candidate_file(candidate_file, raw_candidates)

    acc_file = os.path.join(output_dir, "room_acc", f"{room}_acc.txt")
    save_to_file(
        acc_file,
        f"Accuracy@0.25 after {total_predictions} predictions: {acc_25 * 100:.2f}%\n"
        f"Accuracy@0.50 after {total_predictions} predictions: {acc_50 * 100:.2f}%",
    )
    save_to_file(output_file, json.dumps(results, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scanrefer", help="Dataset name")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument(
        "--task_root",
        default="",
        help="Task folder containing task.json/gt.json/ref.json (overrides scene_json/gt_json/candidate_image_root).",
    )
    parser.add_argument(
        "--scene_json",
        default="",
        help="Scene json with scene_id and description.",
    )
    parser.add_argument(
        "--candidate_dir",
        default="",
        help="Directory with per-scene candidate JSON files (optional).",
    )
    parser.add_argument(
        "--gt_json",
        default="",
        help="GT json with target_id and gt_bbox.",
    )
    parser.add_argument(
        "--pcd_dir",
        required=True,
        help="Point cloud directory for rendering",
    )
    parser.add_argument(
        "--openai_api_key", required=True, help="OpenAI API Key"
    )
    parser.add_argument(
        "--openai_api_base", required=True, help="OpenAI API Base URL"
    )
    parser.add_argument(
        "--model_name", required=True, help="Model name"
    )
    parser.add_argument(
        "--simple_prompt",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=False,
        help="Use simple prompt instead of optimized structured prompt (for ablation study)"
    )
    parser.add_argument(
        "--use_image",
        type=lambda v: v.lower() in ("1", "true", "yes", "y"),
        default=True,
        help="Whether to use image rendering (true/false)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=680,
        help="Rendered image size for VLM input.",
    )
    parser.add_argument(
        "--request_delay",
        type=float,
        default=0.0,
        help="Sleep seconds between API requests.",
    )
    parser.add_argument(
        "--max_candidates",
        type=int,
        default=0,
        help="Limit the number of candidates per query (0 means no limit).",
    )
    parser.add_argument(
        "--sort_by_score",
        type=lambda v: v.lower() in ("1", "true", "yes", "y"),
        default=False,
        help="Sort candidates by score before truncation.",
    )
    parser.add_argument(
        "--anchor_mode",
        choices=["mean", "top1"],
        default="mean",
        help="Anchor selection strategy for rendering.",
    )
    parser.add_argument(
        "--num_global_views",
        type=int,
        default=1,
        help="Number of global render views to generate.",
    )
    parser.add_argument(
        "--global_view_azims",
        default="",
        help="Comma-separated azimuths for global views (overrides num_global_views).",
    )
    parser.add_argument(
        "--candidate_image_root",
        default="",
        help="Root directory with per-candidate image folders.",
    )
    parser.add_argument(
        "--candidate_image_per_id",
        type=int,
        default=0,
        help="Number of candidate crop images per ID to attach (0 disables).",
    )
    parser.add_argument(
        "--max_candidate_images",
        type=int,
        default=0,
        help="Global cap for candidate crop images per query (0 means no cap).",
    )
    parser.add_argument(
        "--candidate_image_use_projection",
        type=lambda v: v.lower() in ("1", "true", "yes", "y"),
        default=False,
        help="Project 3D bbox to candidate images when camera params are available.",
    )
    parser.add_argument(
        "--candidate_image_collage",
        type=lambda v: v.lower() in ("1", "true", "yes", "y"),
        default=False,
        help="Stitch per-ID candidate images into a single collage.",
    )
    parser.add_argument(
        "--candidate_image_primary_smallest",
        type=lambda v: v.lower() in ("1", "true", "yes", "y"),
        default=False,
        help="Feed the smallest candidate crop first as the primary view.",
    )
    parser.add_argument(
        "--candidate_image_primary_only",
        type=lambda v: v.lower() in ("1", "true", "yes", "y"),
        default=False,
        help="Use only the primary (tightest) crop per ID; ignore aux images.",
    )
    parser.add_argument(
        "--camera_distance_factor",
        type=float,
        default=1.0,
        help="Scale camera distance for rendering.",
    )
    parser.add_argument(
        "--camera_lift",
        type=float,
        default=1.5,
        help="Vertical lift applied to the camera when not using top-down view.",
    )
    parser.add_argument(
        "--camera_elev",
        type=float,
        default=0.0,
        help="Elevation angle in degrees for non-top-down view.",
    )
    parser.add_argument(
        "--min_camera_distance",
        type=float,
        default=0.0,
        help="Hard minimum camera distance from the anchor.",
    )
    parser.add_argument(
        "--min_camera_height",
        type=float,
        default=None,
        help="Hard minimum camera height (absolute Z) for the camera center.",
    )
    parser.add_argument(
        "--fixed_camera_center",
        default="",
        help="Fixed camera center as 'x,y,z'. Orientation still looks at anchor mean.",
    )
    parser.add_argument(
        "--camera_focal_length",
        default="",
        help="Fixed camera focal length as 'fx,fy' in NDC units.",
    )
    parser.add_argument(
        "--camera_principal_point",
        default="",
        help="Fixed camera principal point as 'cx,cy' in NDC units.",
    )
    parser.add_argument(
        "--fixed_camera_params",
        default="",
        help="Path to camera_params.json; uses all fields except direction.",
    )
    parser.add_argument(
        "--adaptive_point_radius",
        type=lambda v: v.lower() in ("1", "true", "yes", "y"),
        default=False,
        help="Adapt point radius based on camera distance and focal length.",
    )
    parser.add_argument(
        "--save_camera_params",
        type=lambda v: v.lower() in ("1", "true", "yes", "y"),
        default=False,
        help="Save camera parameters to render/<scene>/<query>/<view>/camera_params.json",
    )
    parser.add_argument(
        "--top_down_view",
        type=lambda v: v.lower() in ("1", "true", "yes", "y"),
        default=True,
        help="Force a top-down view (not fully vertical).",
    )
    parser.add_argument(
        "--top_down_tilt",
        type=float,
        default=15.0,
        help="Tilt angle in degrees away from vertical for top-down view.",
    )
    parser.add_argument(
        "--ensure_all_visible",
        type=lambda v: v.lower() in ("1", "true", "yes", "y"),
        default=True,
        help="Adjust camera to keep all candidate bboxes in view.",
    )
    parser.add_argument(
        "--write_candidate_bboxes",
        type=lambda v: v.lower() in ("1", "true", "yes", "y"),
        default=True,
        help="Write updated candidate bbox_3d back to the candidate JSON.",
    )
    parser.add_argument(
        "--filter_ceiling_points",
        type=lambda v: v.lower() in ("1", "true", "yes", "y"),
        default=False,
        help="Filter out ceiling points from the point cloud (useful for ARKitScenes).",
    )
    parser.add_argument(
        "--ceiling_percentile",
        type=float,
        default=95.0,
        help="Percentile threshold for ceiling height (default: 95, removes top 5%% of points).",
    )
    parser.add_argument(
        "--compress_collage",
        type=lambda v: v.lower() in ("1", "true", "yes", "y"),
        default=True,
        help="Compress collage images to reduce size (default: True).",
    )
    parser.add_argument(
        "--collage_quality",
        type=int,
        default=85,
        help="JPEG quality for compressed collages (1-100, default: 85).",
    )
    parser.add_argument(
        "--collage_max_size",
        type=int,
        default=0,
        help="Maximum dimension (width or height) for collages (0 means no limit, default: 0).",
    )

    args = parser.parse_args()

    if args.task_root:
        task_root = os.path.abspath(args.task_root)
        scene_json = resolve_task_json(task_root)
        if not scene_json:
            raise ValueError(f"No task json found under {task_root}")
        gt_json = os.path.join(task_root, "gt.json")
        if not os.path.exists(gt_json):
            raise ValueError(f"Missing gt.json under {task_root}")
        args.scene_json = scene_json
        args.gt_json = gt_json
        args.candidate_image_root = task_root

    if not args.scene_json:
        raise ValueError("--scene_json is required unless --task_root is provided.")
    if not args.gt_json:
        raise ValueError("--gt_json is required unless --task_root is provided.")

    scan_ids = [os.path.splitext(os.path.basename(args.scene_json))[0]]
    print(f"Found {len(scan_ids)} scans in {args.scene_json}")

    for room in scan_ids:
        candidate_file = (
            os.path.join(args.candidate_dir, f"{room}.json")
            if args.candidate_dir
            else ""
        )
        fixed_camera_params = None
        if args.fixed_camera_params:
            fixed_camera_params = load_json(args.fixed_camera_params)
            if not isinstance(fixed_camera_params, dict):
                raise ValueError("--fixed_camera_params must point to a JSON object.")

        fixed_camera_center = None
        if args.fixed_camera_center:
            parts = [p.strip() for p in args.fixed_camera_center.split(",")]
            if len(parts) == 3:
                fixed_camera_center = [float(p) for p in parts]
            else:
                raise ValueError("--fixed_camera_center expects 'x,y,z'.")
        fixed_focal_length = None
        if args.camera_focal_length:
            parts = [p.strip() for p in args.camera_focal_length.split(",")]
            if len(parts) == 2:
                fixed_focal_length = [float(p) for p in parts]
            else:
                raise ValueError("--camera_focal_length expects 'fx,fy'.")
        fixed_principal_point = None
        if args.camera_principal_point:
            parts = [p.strip() for p in args.camera_principal_point.split(",")]
            if len(parts) == 2:
                fixed_principal_point = [float(p) for p in parts]
            else:
                raise ValueError("--camera_principal_point expects 'cx,cy'.")

        process_room(
            dataset=args.dataset,
            room=room,
            output_dir=args.output_dir,
            pcd_dir=args.pcd_dir,
            scene_json_path=args.scene_json,
            candidate_file=candidate_file,
            gt_json_path=args.gt_json,
            openai_api_key=args.openai_api_key,
            openai_api_base=args.openai_api_base,
            use_image=args.use_image,
            model_name=args.model_name,
            image_size=args.image_size,
            request_delay=args.request_delay,
            max_candidates=args.max_candidates,
            sort_by_score=args.sort_by_score,
            anchor_mode=args.anchor_mode,
            num_global_views=args.num_global_views,
            global_view_azims=args.global_view_azims,
            candidate_image_root=args.candidate_image_root,
            candidate_image_per_id=args.candidate_image_per_id,
            max_candidate_images=args.max_candidate_images,
            candidate_image_use_projection=args.candidate_image_use_projection,
            candidate_image_collage=args.candidate_image_collage,
            candidate_image_primary_smallest=args.candidate_image_primary_smallest,
            candidate_image_primary_only=args.candidate_image_primary_only,
            fixed_camera_center=fixed_camera_center,
            fixed_focal_length=fixed_focal_length,
            fixed_principal_point=fixed_principal_point,
            fixed_camera_params=fixed_camera_params,
            adaptive_point_radius=args.adaptive_point_radius,
            save_camera_params=args.save_camera_params,
            camera_distance_factor=args.camera_distance_factor,
            camera_lift=args.camera_lift,
            camera_elev=args.camera_elev,
            min_camera_distance=args.min_camera_distance,
            min_camera_height=args.min_camera_height,
            top_down_view=args.top_down_view,
            top_down_tilt=args.top_down_tilt,
            ensure_all_visible=args.ensure_all_visible,
            write_candidate_bboxes=args.write_candidate_bboxes,
            use_simple_prompt=args.simple_prompt,
            filter_ceiling_points=args.filter_ceiling_points,
            ceiling_percentile=args.ceiling_percentile,
            compress_collage=args.compress_collage,
            collage_quality=args.collage_quality,
            collage_max_size=args.collage_max_size,
        )
