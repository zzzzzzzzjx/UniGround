import re
from difflib import get_close_matches
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
import base64
import json
import os
import mimetypes
import numpy as np
import open3d as o3d
import torch

nltk.download("punkt_tab")
stemmer = PorterStemmer()


def fuzzy_match(names, object_names, threshold=0.8):
    matched_names = set()
    for name in names:
        matches = get_close_matches(name, object_names, n=1, cutoff=threshold)
        if matches:
            matched_names.add(matches[0])
    return matched_names


def stem_match(names, object_names):
    matched_names = set()
    if isinstance(names, str):
        names = [names]
    for name in names:
        name_stems = [stemmer.stem(word) for word in word_tokenize(name)]
        for obj_name in object_names:
            obj_name_stems = [stemmer.stem(word) for word in word_tokenize(obj_name)]
            if set(name_stems) & set(obj_name_stems):
                matched_names.add(obj_name)
    return matched_names


def load_json(file_path):
    """Load data from a JSON file."""
    with open(file_path, "r") as file:
        return json.load(file)


def save_to_file(file_path, content):
    """Save content to a file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a") as file:
        file.write(content)


def encode_img(image_path):
    """Encode image to Base64 format."""
    with open(image_path, "rb") as file:
        encoded_image = base64.b64encode(file.read()).decode("utf-8")
    mime, _ = mimetypes.guess_type(image_path)
    if not mime or not mime.startswith("image/"):
        mime = "image/png"
    return f"data:{mime};base64,{encoded_image}"


def read_file_to_list(file_path):
    """
    Read a file and convert its contents to a list.
    """
    with open(file_path, "r") as file:
        lines = file.read().splitlines()
    return sorted(lines)


def calc_iou(box_a, box_b):
    """Computes IoU of two axis aligned bboxes.
    Args:
        box_a, box_b: 6D of center and lengths
    Returns:
        iou
    """
    box_a = np.array(box_a)
    box_b = np.array(box_b)

    max_a = box_a[0:3] + box_a[3:6] / 2
    max_b = box_b[0:3] + box_b[3:6] / 2
    min_max = np.array([max_a, max_b]).min(0)

    min_a = box_a[0:3] - box_a[3:6] / 2
    min_b = box_b[0:3] - box_b[3:6] / 2
    max_min = np.array([min_a, min_b]).max(0)
    if not ((min_max > max_min).all()):
        return 0.0

    intersection = (min_max - max_min).prod()
    vol_a = box_a[3:6].prod()
    vol_b = box_b[3:6].prod()
    union = vol_a + vol_b - intersection
    return 1.0 * intersection / union


def parse_response(response):
    """
    Parse model response to extract predicted ID, explanation, candidate names, and
    structured fallback fields. Handles both multi-line and single-line formats.
    Also handles simple responses (just a number).
    """
    import re

    predicted_id = None
    explanation = None
    candidate_names = None
    target_name_matches = None
    anchor_id = None
    relation_used = None

    # First, try to parse as simple response (just a number)
    # This handles cases where the model only returns an ID
    simple_match = re.match(r'^\s*(\d+)\s*$', response.strip())
    if simple_match:
        try:
            predicted_id = int(simple_match.group(1))
            # For simple responses, set explanation to indicate it's a simple response
            explanation = f"ID {predicted_id}"
            return (
                predicted_id,
                explanation,
                candidate_names,
                target_name_matches,
                anchor_id,
                relation_used,
            )
        except ValueError:
            pass

    # Try to extract fields using regex patterns (works for both single-line and multi-line)
    patterns = {
        'predicted_id': r'Predicted ID:\s*(\d+)',
        'explanation': r'Explanation:\s*([^A-Z][^\n]*?)(?=\s+(?:Candidate Names|Target name matches|Anchor ID|Relation used|$))',
        'candidate_names': r'Candidate Names:\s*([^\n]*?)(?=\s+(?:Target name matches|Anchor ID|Relation used|$))',
        'target_name_matches': r'Target name matches\??:\s*(\w+)',
        'anchor_id': r'Anchor ID:\s*([^\n\s]+)',
        'relation_used': r'Relation used:\s*([^\n]*?)(?=\s*$)',
    }

    # Extract predicted_id
    match = re.search(patterns['predicted_id'], response, re.IGNORECASE)
    if match:
        try:
            predicted_id = int(match.group(1))
        except ValueError:
            predicted_id = None

    # Extract explanation
    match = re.search(patterns['explanation'], response, re.IGNORECASE | re.DOTALL)
    if match:
        explanation = match.group(1).strip()

    # Extract candidate_names
    match = re.search(patterns['candidate_names'], response, re.IGNORECASE)
    if match:
        candidate_names = match.group(1).strip()

    # Extract target_name_matches
    match = re.search(patterns['target_name_matches'], response, re.IGNORECASE)
    if match:
        value_lower = match.group(1).lower()
        if value_lower.startswith("y"):
            target_name_matches = True
        elif value_lower.startswith("n"):
            target_name_matches = False

    # Extract anchor_id
    match = re.search(patterns['anchor_id'], response, re.IGNORECASE)
    if match:
        value = match.group(1).strip()
        if value.lower() not in {"", "unknown", "none"}:
            try:
                anchor_id = int(value)
            except ValueError:
                anchor_id = value

    # Extract relation_used
    match = re.search(patterns['relation_used'], response, re.IGNORECASE)
    if match:
        relation_used = match.group(1).strip() or None

    return (
        predicted_id,
        explanation,
        candidate_names,
        target_name_matches,
        anchor_id,
        relation_used,
    )


# Data Loading and Processing
def load_bboxes(room, bbox_dir, file_type="pred"):
    """Load bounding boxes (GT or predicted)."""
    bbox_file = os.path.join(bbox_dir, f"{room}.json")
    bboxes = load_json(bbox_file)
    return {int(bbox["bbox_id"]): bbox for bbox in bboxes}


def generate_objects_info(pred_bbox_list):
    """Generate a formatted string of object information."""
    return "\n".join(
        [
            f"Object ID: {bbox['bbox_id']}, Type: {bbox['target']}, Dimensions: Width {bbox['bbox_3d'][3]:.2f}, Length {bbox['bbox_3d'][4]:.2f}, Height {bbox['bbox_3d'][5]:.2f}, Center Coordinates: X {bbox['bbox_3d'][0]:.2f}, Y {bbox['bbox_3d'][1]:.2f}, Z {bbox['bbox_3d'][2]:.2f}"
            for bbox in pred_bbox_list
            if bbox["target"] not in ["wall", "floor", "ceiling", "object", "objects"]
        ]
    )

# Rendering
def load_scene_pcd(room, scan_dir='/remote-home/share/vg_datasets/referit3d/scan_data/pcd_with_global_alignment/'):
    """Load and process point cloud data."""
    # ply_file_path = os.path.join('/remote-home/rongli/global_alignment_scene_ply', f"{room}.ply")
    # pcd = o3d.io.read_point_cloud(ply_file_path)
    # pc = np.asarray(pcd.points)
    # color = np.asarray(pcd.colors)

    pcds, colors, _, instance_labels = torch.load(
        os.path.join(scan_dir, '%s.pth' % room))

    scan_pc = np.concatenate((pcds, colors/255), axis=1).astype("float32")
    center = np.mean(scan_pc[:, :3], axis=0)
    return scan_pc, center
