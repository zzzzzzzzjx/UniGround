import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PointsRenderer,
    PointsRasterizationSettings,
    PointsRasterizer,
    AlphaCompositor,
    FoVPerspectiveCameras,
    PerspectiveCameras,
    look_at_view_transform,
)
from PIL import Image, ImageDraw, ImageFont
import os
import cv2
import json
import re
import random
from scipy.spatial import ConvexHull


def filter_ceiling(scan_pc, ceiling_percentile=95):
    """
    Filter out ceiling points from the point cloud.

    Args:
        scan_pc (np.ndarray): Point cloud data with shape (N, 6) containing xyz and rgb.
        ceiling_percentile (float): Percentile threshold for ceiling height (default: 95).

    Returns:
        np.ndarray: Filtered point cloud without ceiling points.
    """
    z_values = scan_pc[:, 2]
    ceiling_threshold = np.percentile(z_values, ceiling_percentile)

    # Keep only points below the ceiling threshold
    filtered_pc = scan_pc[z_values < ceiling_threshold]

    print(f"Filtered ceiling: removed {len(scan_pc) - len(filtered_pc)} points (threshold z={ceiling_threshold:.2f})")

    return filtered_pc


def render_point_cloud_with_pytorch3d_with_objects(
    objects,
    targets,
    anchors,
    center,
    scan_pc,
    save_dir=None,
    image_size=680,
    use_color_image=True,
    draw_bbox=False,
    draw_id=False,
    draw_img=False,
    draw_mask=False,
    draw_contour=False,
    camera_azim=0.0,
    camera_elev=0.0,
    camera_distance_factor=1.0,
    camera_lift=1.5,
    calibrate=False,
    top_down_view=False,
    top_down_tilt=15.0,
    ensure_all_visible=False,
    min_camera_distance=0.0,
    min_camera_height=None,
    fixed_camera_center=None,
    fixed_focal_length=None,
    fixed_principal_point=None,
    adaptive_point_radius=False,
    save_camera_params=False,
    anchor_bboxes=None,
    filter_ceiling_points=False,
    ceiling_percentile=95,
    point_radius=None,
    device="cuda",
):
    """
    Render point cloud with PyTorch3D and annotate with objects, targets, and anchors.

    Args:
        objects (list): List of objects to render.
        targets (list): List of target objects.
        anchors (list): List of anchor objects.
        center (array): Center of the point cloud.
        scan_pc (array): Point cloud data.
        save_dir (str): Directory to save the rendered image.
        image_size (int): Size of the output image.
        use_color_image (bool): Whether to use a color image.
        draw_bbox (bool): Whether to draw bounding boxes.
        draw_id (bool): Whether to draw object IDs.
        draw_img (bool): Whether to draw the image.
        draw_mask (bool): Whether to draw masks.
        draw_contour (bool): Whether to draw contours.
        filter_ceiling_points (bool): Whether to filter ceiling points (useful for ARKitScenes).
        ceiling_percentile (float): Percentile threshold for ceiling height (default: 95).
        device (str): Device to use for rendering.

    Returns:
        str: Path to the saved image.
    """
    # Filter ceiling points if enabled (useful for ARKitScenes)
    if filter_ceiling_points:
        scan_pc = filter_ceiling(scan_pc, ceiling_percentile)

    point_cloud = create_point_cloud(scan_pc, device)
    os.makedirs(save_dir, exist_ok=True)

    # Compute the mean position of anchors
    accumulated_positions = torch.zeros(3, dtype=torch.float32)
    for anchor in anchors:
        anchor_bbox_3d = torch.tensor(anchor["bbox_3d"][:3], dtype=torch.float32)
        accumulated_positions += anchor_bbox_3d

    mean_position = accumulated_positions / len(anchors)
    anchor_bbox_3d = mean_position.to(dtype=torch.float32)

    min_bounds = point_cloud.points_padded().min(dim=1)[0][0]
    max_bounds = point_cloud.points_padded().max(dim=1)[0][0]
    scene_range = (max_bounds - min_bounds).max().clamp_min(1e-6)
    axis_len = float(scene_range * 0.15)
    if axis_len < 0.2:
        axis_len = 0.2

    if ensure_all_visible and fixed_focal_length is None and fixed_principal_point is None:
        calibrate = True

    cameras = setup_camera(
        anchor_bbox_3d=anchor_bbox_3d,
        center=center,
        image_size=image_size,
        camera_distance_factor=camera_distance_factor,
        camera_lift=camera_lift,
        device=device,
        point_cloud=point_cloud,
        calibrate=calibrate,
        azim=camera_azim,
        elev=camera_elev,
        top_down_view=top_down_view,
        top_down_tilt=top_down_tilt,
        ensure_all_visible=ensure_all_visible,
        fit_bboxes=targets if ensure_all_visible else None,
        min_camera_distance=min_camera_distance,
        min_camera_height=min_camera_height,
        fixed_camera_center=fixed_camera_center,
        fixed_focal_length=fixed_focal_length,
        fixed_principal_point=fixed_principal_point,
    )
    
    # 自适应渲染点云半径
    if point_radius is None:
        point_radius = 0.005
    if adaptive_point_radius:
        min_bounds = point_cloud.points_padded().min(dim=1)[0][0]
        max_bounds = point_cloud.points_padded().max(dim=1)[0][0]
        scene_range = (max_bounds - min_bounds).max().clamp_min(1e-6)
        cam_center = cameras.get_camera_center()[0].to(anchor_bbox_3d.device)
        dist = torch.linalg.norm(cam_center - anchor_bbox_3d).clamp_min(1e-6)
        ref_dist = scene_range * float(camera_distance_factor)
        dist_ratio = ref_dist / dist
        focal_scale = cameras.focal_length[0].abs().max().clamp_min(1e-6)
        base_px = torch.tensor(
            point_radius * image_size / 2.0,
            device=cam_center.device,
            dtype=cam_center.dtype,
        )
        px = base_px * torch.sqrt(dist_ratio) * focal_scale
        min_px = base_px * 0.7
        max_px = base_px * 3.0
        min_px = min_px.to(px.device)
        max_px = max_px.to(px.device)
        px = torch.clamp(px, min=min_px, max=max_px)
        point_radius = float(px * 2.0 / image_size)

    image_np, rasterizer = render_point_cloud(
        point_cloud, cameras, image_size, device, point_radius=point_radius
    )

    depth_map = compute_depth_map(rasterizer, point_cloud)

    color_image = Image.fromarray((image_np * 255).astype(np.uint8))

    if not draw_img:
        width, height = color_image.size
        color_image = Image.new("RGB", (width, height), (255, 255, 255))

    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf", 15, encoding="unic"
    )

    color = (255, 0, 0)  # grey color
    annotate_image(
        color_image,
        anchors,
        targets,
        cameras,
        image_size,
        font,
        scan_pc=scan_pc,
        depth_map=None,
        bbox_color=color,
        draw_bbox=draw_bbox,
        draw_mask=draw_mask,
        draw_id=draw_id,
        draw_contour=draw_contour,
        anchor_bboxes=anchor_bboxes,
        axis_origin=anchor_bbox_3d,
        axis_len=axis_len,
        draw_axes=True,
        device=device,
    )

    if save_camera_params:
        camera_center = cameras.get_camera_center()[0].detach().cpu().numpy()
        anchor_pos = anchor_bbox_3d.detach().cpu().numpy()
        vec = camera_center - anchor_pos
        dist = float(np.linalg.norm(vec))
        direction = (vec / dist).tolist() if dist > 1e-6 else [0.0, 0.0, 0.0]
        focal = cameras.focal_length[0].detach().cpu().numpy().tolist()
        principal = cameras.principal_point[0].detach().cpu().numpy().tolist()
        params = {
            "camera_center": camera_center.tolist(),
            "anchor": anchor_pos.tolist(),
            "direction": direction,
            "distance": dist,
            "focal_length": [float(v) for v in focal],
            "principal_point": [float(v) for v in principal],
            "point_radius": float(point_radius),
            "adaptive_point_radius": bool(adaptive_point_radius),
            "camera_azim": float(camera_azim),
            "camera_elev": float(camera_elev),
            "camera_distance_factor": float(camera_distance_factor),
            "camera_lift": float(camera_lift),
            "top_down_view": bool(top_down_view),
            "top_down_tilt": float(top_down_tilt),
            "min_camera_distance": float(min_camera_distance),
            "min_camera_height": None if min_camera_height is None else float(min_camera_height),
            "fixed_camera_center": None
            if fixed_camera_center is None
            else [float(v) for v in fixed_camera_center],
            "fixed_focal_length": None
            if fixed_focal_length is None
            else [float(v) for v in fixed_focal_length],
            "fixed_principal_point": None
            if fixed_principal_point is None
            else [float(v) for v in fixed_principal_point],
            "ensure_all_visible": bool(ensure_all_visible),
            "filter_ceiling_points": bool(filter_ceiling_points),
            "ceiling_percentile": float(ceiling_percentile),
        }
        params_path = os.path.join(save_dir, "camera_params.json")
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)

    f_name = f"{save_dir}/rendered.png"
    color_image.save(f_name)
    print(f"Annotated image saved at {f_name}")
    color_image.close()

    return f_name


def annotate_image(
    color_image,
    anchors,
    targets,
    cameras,
    image_size,
    font,
    depth_map,
    scan_pc,
    bbox_color=(0, 255, 0),
    draw_bbox=False,
    draw_mask=False,
    draw_contour=False,
    draw_id=False,
    anchor_bboxes=None,
    axis_origin=None,
    axis_len=None,
    draw_axes=False,
    device="cuda",
):
    """
    Annotate the image with bounding boxes, masks, contours, and IDs.

    Args:
        color_image (Image): The image to annotate.
        anchors (list): List of anchor objects.
        targets (list): List of target objects.
        cameras (PerspectiveCameras): Camera settings.
        image_size (int): Size of the output image.
        font (ImageFont): Font for drawing text.
        depth_map (array): Depth map of the point cloud.
        scan_pc (array): Point cloud data.
        bbox_color (tuple): Color for bounding boxes.
        draw_bbox (bool): Whether to draw bounding boxes.
        draw_mask (bool): Whether to draw masks.
        draw_contour (bool): Whether to draw contours.
        draw_id (bool): Whether to draw object IDs.
    """
    draw = ImageDraw.Draw(color_image, "RGBA")

    if draw_mask:
        draw_masks(draw, targets, cameras, scan_pc, image_size, device)

    if draw_contour:
        draw_contours(draw, targets, cameras, scan_pc, image_size, device)

    if draw_bbox:
        draw_bboxes(draw, anchors + targets, cameras, image_size, bbox_color, device)

    if draw_id:
        draw_ids(draw, anchors + targets, cameras, image_size, font, device)

    if anchor_bboxes:
        draw_anchor_markers(draw, anchor_bboxes, cameras, image_size, device)

    if draw_axes:
        draw_axes_indicator(
            draw,
            cameras,
            image_size,
            origin=axis_origin,
            axis_len=axis_len,
            font=font,
            device=device,
        )

    return


def draw_anchor_markers(draw, anchor_bboxes, cameras, image_size, device):
    if not anchor_bboxes:
        return
    size = max(6, int(image_size * 0.012))
    for bbox in anchor_bboxes:
        if not bbox or len(bbox) < 3:
            continue
        center = torch.tensor([bbox[:3]], dtype=torch.float32, device=device)
        proj = cameras.transform_points_screen(
            center, image_size=(image_size, image_size)
        )
        if proj.ndim == 3:
            x = float(proj[0, 0, 0])
            y = float(proj[0, 0, 1])
        else:
            x = float(proj[0, 0])
            y = float(proj[0, 1])
        if x < 0 or x >= image_size or y < 0 or y >= image_size:
            continue
        points = [
            (x, y - size),
            (x - size, y + size),
            (x + size, y + size),
        ]
        draw.polygon(points, fill=(255, 0, 0))


def _draw_arrow(draw, start, end, color, label=None, font=None):
    draw.line([start, end], fill=color, width=3)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    norm = (dx * dx + dy * dy) ** 0.5
    if norm < 1e-3:
        return
    ux = dx / norm
    uy = dy / norm
    head_len = 10.0
    head_w = 5.0
    left = (end[0] - ux * head_len - uy * head_w, end[1] - uy * head_len + ux * head_w)
    right = (end[0] - ux * head_len + uy * head_w, end[1] - uy * head_len - ux * head_w)
    draw.polygon([end, left, right], fill=color)
    if label and font:
        draw.text((end[0] + 2, end[1] + 2), label, fill=color, font=font)


def draw_axes_indicator(draw, cameras, image_size, origin, axis_len, font, device):
    if origin is None or axis_len is None:
        return
    if isinstance(origin, torch.Tensor):
        origin_t = origin.to(device=device, dtype=torch.float32).view(1, 3)
    else:
        origin_t = torch.tensor(origin, dtype=torch.float32, device=device).view(1, 3)
    axis_len = float(axis_len)
    axes = torch.tensor(
        [[axis_len, 0.0, 0.0], [0.0, axis_len, 0.0], [0.0, 0.0, axis_len]],
        dtype=torch.float32,
        device=device,
    )
    pts = torch.cat([origin_t, origin_t + axes], dim=0)
    proj = cameras.transform_points_screen(pts, image_size=(image_size, image_size))
    proj = proj[..., :2].detach().cpu().numpy()
    origin_px = (float(proj[0, 0]), float(proj[0, 1]))
    if not (0 <= origin_px[0] < image_size and 0 <= origin_px[1] < image_size):
        return
    x_px = (float(proj[1, 0]), float(proj[1, 1]))
    y_px = (float(proj[2, 0]), float(proj[2, 1]))
    z_px = (float(proj[3, 0]), float(proj[3, 1]))
    _draw_arrow(draw, origin_px, x_px, (255, 0, 0), label="X", font=font)
    _draw_arrow(draw, origin_px, y_px, (0, 255, 0), label="Y", font=font)
    _draw_arrow(draw, origin_px, z_px, (0, 128, 255), label="Z", font=font)


def draw_masks(draw, targets, cameras, scan_pc, image_size, device):
    """
    Draw masks on the image.

    Args:
        draw (ImageDraw): ImageDraw object.
        targets (list): List of target objects.
        cameras (PerspectiveCameras): Camera settings.
        scan_pc (array): Point cloud data.
        image_size (int): Size of the output image.
    """
    for bbox in targets:
        bbox_id = bbox["bbox_id"]
        obj_label = bbox["label"]
        x, y, z, w, l, h = bbox["bbox_3d"]

        in_bbox_points = scan_pc[
            (scan_pc[:, 0] >= x - w / 2)
            & (scan_pc[:, 0] <= x + w / 2)
            & (scan_pc[:, 1] >= y - l / 2)
            & (scan_pc[:, 1] <= y + l / 2)
            & (scan_pc[:, 2] >= z - h / 2)
            & (scan_pc[:, 2] <= z + h / 2)
        ]

        projected_points = cameras.transform_points_screen(
            torch.tensor(in_bbox_points[:, :3], device=device),
            image_size=(image_size, image_size),
        )
        projected_points = projected_points[..., :2]

        visible_points = [(int(px), int(py)) for px, py in projected_points]

        mask_color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            100,
        )  # Random color with transparency
        draw.polygon(visible_points, fill=mask_color)


def draw_contours(draw, targets, cameras, scan_pc, image_size, device):
    """
    Draw contours on the image.

    Args:
        draw (ImageDraw): ImageDraw object.
        targets (list): List of target objects.
        cameras (PerspectiveCameras): Camera settings.
        scan_pc (array): Point cloud data.
        image_size (int): Size of the output image.
    """
    for bbox in targets:
        bbox_id = bbox["bbox_id"]
        obj_label = bbox["label"]
        x, y, z, w, l, h = bbox["bbox_3d"]

        in_bbox_points = scan_pc[
            (scan_pc[:, 0] >= x - w / 2)
            & (scan_pc[:, 0] <= x + w / 2)
            & (scan_pc[:, 1] >= y - l / 2)
            & (scan_pc[:, 1] <= y + l / 2)
            & (scan_pc[:, 2] >= z - h / 2)
            & (scan_pc[:, 2] <= z + h / 2)
        ]

        projected_points = cameras.transform_points_screen(
            torch.tensor(in_bbox_points[:, :3], device=device),
            image_size=(image_size, image_size),
        )
        projected_points = projected_points[..., :2]

        visible_points = [(int(px), int(py)) for px, py in projected_points]

        points_array = np.array(visible_points)
        try:
            hull = ConvexHull(points_array)  # Compute the convex hull
            contour_points = points_array[hull.vertices]  # Get contour points in order
            contour_points = [(int(x), int(y)) for x, y in contour_points]

            contour_color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
                255,
            )  # Random color with transparency
            draw.line(
                contour_points + [contour_points[0]], fill=contour_color, width=3
            )  # Close the contour loop
        except:
            pass


def draw_bboxes(draw, bboxes, cameras, image_size, bbox_color, device):
    """
    Draw bounding boxes on the image.

    Args:
        draw (ImageDraw): ImageDraw object.
        bboxes (list): List of bounding boxes.
        cameras (PerspectiveCameras): Camera settings.
        image_size (int): Size of the output image.
        bbox_color (tuple): Color for bounding boxes.
    """
    for bbox in bboxes:
        x, y, z, w, l, h = bbox["bbox_3d"]

        # Define the eight corners of the 3D bounding box
        corners = [
            [x - w / 2, y - l / 2, z - h / 2],
            [x - w / 2, y + l / 2, z - h / 2],
            [x + w / 2, y - l / 2, z - h / 2],
            [x + w / 2, y + l / 2, z - h / 2],
            [x - w / 2, y - l / 2, z + h / 2],
            [x - w / 2, y + l / 2, z + h / 2],
            [x + w / 2, y - l / 2, z + h / 2],
            [x + w / 2, y + l / 2, z + h / 2],
        ]

        # Project the 3D corners to the 2D image plane
        corners_2d = cameras.transform_points_screen(
            torch.tensor(corners, device=device), image_size=(image_size, image_size)
        )
        corners_2d = corners_2d[..., :2].cpu().numpy()

        # Check if each corner is within the image boundaries
        valid_corners = [
            (0 <= x < image_size and 0 <= y < image_size) for x, y in corners_2d
        ]

        # Skip drawing if all corners are out of image boundaries
        if not any(valid_corners):
            continue

        # Draw the 3D bounding box
        draw_bbox_function(draw, corners_2d, valid_corners, bbox_color)


def draw_ids(draw, bboxes, cameras, image_size, font, device):
    """
    Draw object IDs on the image.

    Args:
        draw (ImageDraw): ImageDraw object.
        bboxes (list): List of bounding boxes.
        cameras (PerspectiveCameras): Camera settings.
        image_size (int): Size of the output image.
        font (ImageFont): Font for drawing text.
    """
    for bbox in bboxes:
        bbox_id = bbox["bbox_id"]
        x, y, z, w, l, h = bbox["bbox_3d"]

        # Define the eight corners of the 3D bounding box
        corners = [
            [x - w / 2, y - l / 2, z - h / 2],
            [x - w / 2, y + l / 2, z - h / 2],
            [x + w / 2, y - l / 2, z - h / 2],
            [x + w / 2, y + l / 2, z - h / 2],
            [x - w / 2, y - l / 2, z + h / 2],
            [x - w / 2, y + l / 2, z + h / 2],
            [x + w / 2, y - l / 2, z + h / 2],
            [x + w / 2, y + l / 2, z + h / 2],
        ]

        # Project the 3D corners to the 2D image plane
        corners_2d = cameras.transform_points_screen(
            torch.tensor(corners, device=device), image_size=(image_size, image_size)
        )
        corners_2d = corners_2d[..., :2].cpu().numpy()

        # Check if each corner is within the image boundaries
        valid_corners = [
            (0 <= x < image_size and 0 <= y < image_size) for x, y in corners_2d
        ]

        # Skip drawing if all corners are out of image boundaries
        if not any(valid_corners):
            continue

        # Draw the label and bbox_id
        draw_label(draw, corners_2d, bbox_id, font, image_size)


def draw_label(draw, corners_2d, bbox_id, font, image_size):
    """
    Draw label and bbox_id at the center of the top face of the bounding box.

    Args:
        draw (ImageDraw): ImageDraw object.
        corners_2d (array): 2D coordinates of the bounding box corners.
        bbox_id (int): Bounding box ID.
        font (ImageFont): Font for drawing text.
        image_size (int): Size of the output image.
    """
    # Find the center of the top face
    center_x = int(
        (corners_2d[4][0] + corners_2d[5][0] + corners_2d[6][0] + corners_2d[7][0]) / 4
    )
    center_y = int(
        (corners_2d[4][1] + corners_2d[5][1] + corners_2d[6][1] + corners_2d[7][1]) / 4
    )
    if 0 <= center_x < image_size and 0 <= center_y < image_size:
        text = f"{bbox_id}"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        background_x0 = center_x - text_width // 2 - 2
        background_y0 = center_y - text_height // 2 - 2
        background_x1 = center_x + text_width // 2 + 2
        background_y1 = center_y + text_height // 2 + 2
        draw.rectangle(
            [background_x0, background_y0, background_x1, background_y1],
            fill=(255, 255, 255),
        )
        draw.text(
            (center_x - text_width // 2, center_y - text_height // 2),
            text,
            font=font,
            fill=(255, 0, 0),
        )

    return


def draw_bbox_function(draw, corners_2d, valid_corners, bbox_color):
    """
    Draw the 3D bounding box by connecting the projected corners.

    Args:
        draw (ImageDraw): ImageDraw object.
        corners_2d (array): 2D coordinates of the bounding box corners.
        valid_corners (list): List of booleans indicating if each corner is within the image boundaries.
        bbox_color (tuple): Color for the bounding box.
    """
    # Draw the 3D bounding box by connecting the projected corners
    for i, (start, end) in enumerate(
        [
            (0, 1),
            (1, 3),
            (3, 2),
            (2, 0),
            (4, 5),
            (5, 7),
            (7, 6),
            (6, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]
    ):
        if valid_corners[start] and valid_corners[end]:
            draw.line(
                [tuple(corners_2d[start]), tuple(corners_2d[end])],
                fill=bbox_color,
                width=1,
            )
    return


def create_point_cloud(scan_pc, device):
    """
    Create a point cloud from scan data.

    Args:
        scan_pc (np.ndarray): The scan data containing points and colors.
        device (str): The device to use for computation.

    Returns:
        Pointclouds: The created point cloud.
    """
    points = torch.tensor(scan_pc[:, :3], dtype=torch.float32)
    colors = torch.tensor(scan_pc[:, 3:], dtype=torch.float32)
    point_cloud = Pointclouds(points=[points], features=[colors]).to(device)
    return point_cloud


def _collect_bbox_corners(bboxes, device):
    if not bboxes:
        return None
    corners = []
    for bbox in bboxes:
        bbox_3d = bbox.get("bbox_3d")
        if not bbox_3d:
            continue
        x, y, z, w, l, h = bbox_3d
        corners.extend(
            [
                [x - w / 2, y - l / 2, z - h / 2],
                [x - w / 2, y + l / 2, z - h / 2],
                [x + w / 2, y - l / 2, z - h / 2],
                [x + w / 2, y + l / 2, z - h / 2],
                [x - w / 2, y - l / 2, z + h / 2],
                [x - w / 2, y + l / 2, z + h / 2],
                [x + w / 2, y - l / 2, z + h / 2],
                [x + w / 2, y + l / 2, z + h / 2],
            ]
        )
    if not corners:
        return None
    return torch.tensor(corners, dtype=torch.float32, device=device).unsqueeze(0)


def setup_camera(
    point_cloud,
    anchor_bbox_3d,
    center,
    image_size,
    camera_distance_factor=1.0,
    camera_lift=1.0,
    azim=0.0,
    elev=0.0,
    device="cuda",
    calibrate=True,
    top_down_view=False,
    top_down_tilt=15.0,
    ensure_all_visible=False,
    fit_bboxes=None,
    min_camera_distance=0.0,
    min_camera_height=None,
    fixed_camera_center=None,
    fixed_focal_length=None,
    fixed_principal_point=None,
):
    """
    Set up the camera for rendering the point cloud.

    Args:
        point_cloud (Pointclouds): The point cloud to render.
        anchor_bbox_3d (torch.Tensor): The 3D bounding box of the anchor.
        center (np.ndarray): The center of the point cloud.
        image_size (int): The size of the output image.
        camera_distance_factor (float): The factor to adjust camera distance.
        camera_lift (float): The lift to apply to the camera.
        device (str): The device to use for computation.
        calibrate (bool): Whether to calibrate the camera.

    Returns:
        PerspectiveCameras: The set up camera.
    """
    min_bounds = point_cloud.points_padded().min(dim=1)[0]
    max_bounds = point_cloud.points_padded().max(dim=1)[0]

    center = torch.tensor(center, dtype=torch.float32, device=point_cloud.device)
    anchor = anchor_bbox_3d.to(point_cloud.device)
    fit_points = _collect_bbox_corners(fit_bboxes, point_cloud.device)

    if fixed_camera_center is not None:
        fixed_center = torch.tensor(
            fixed_camera_center, dtype=anchor.dtype, device=anchor.device
        )
        cam_vec = fixed_center - anchor
    elif top_down_view:
        scene_range = (max_bounds - min_bounds).max()
        base_dist = scene_range * camera_distance_factor
        direction = torch.tensor([0.0, 0.0, 1.0], device=point_cloud.device)

        tilt = torch.deg2rad(torch.tensor(top_down_tilt, device=point_cloud.device))
        cos_t = torch.cos(tilt)
        sin_t = torch.sin(tilt)
        rot_tilt = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, cos_t, -sin_t], [0.0, sin_t, cos_t]],
            device=point_cloud.device,
            dtype=direction.dtype,
        )

        az = torch.deg2rad(torch.tensor(azim, device=point_cloud.device))
        cos_az = torch.cos(az)
        sin_az = torch.sin(az)
        rot_az = torch.tensor(
            [[cos_az, -sin_az, 0.0], [sin_az, cos_az, 0.0], [0.0, 0.0, 1.0]],
            device=point_cloud.device,
            dtype=direction.dtype,
        )

        direction = rot_az @ (rot_tilt @ direction)
        direction = direction / direction.norm().clamp_min(1e-6)

        dist = base_dist
        if ensure_all_visible and fit_points is not None:
            deltas = fit_points[0] - anchor
            proj = torch.matmul(deltas, direction)
            dist = torch.max(dist, proj.max() + 0.1)

        cam_vec = direction * dist
    else:
        center[2] += camera_lift
        cam_vec = center + camera_distance_factor * (center - anchor) - anchor

        az = torch.deg2rad(torch.tensor(azim, device=point_cloud.device))
        el = torch.deg2rad(torch.tensor(elev, device=point_cloud.device))
        cos_az = torch.cos(az)
        sin_az = torch.sin(az)
        cos_el = torch.cos(el)
        sin_el = torch.sin(el)

        rot_az = torch.tensor(
            [[cos_az, -sin_az, 0.0], [sin_az, cos_az, 0.0], [0.0, 0.0, 1.0]],
            device=point_cloud.device,
            dtype=cam_vec.dtype,
        )
        rot_el = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, cos_el, -sin_el], [0.0, sin_el, cos_el]],
            device=point_cloud.device,
            dtype=cam_vec.dtype,
        )

        cam_vec = rot_az @ (rot_el @ cam_vec)

    cam_dist = torch.linalg.norm(cam_vec)
    if cam_dist > 1e-6:
        direction = cam_vec / cam_dist
        dist = cam_dist
        if min_camera_distance and min_camera_distance > 0:
            min_dist = torch.tensor(
                min_camera_distance, device=cam_vec.device, dtype=cam_vec.dtype
            )
            dist = torch.maximum(dist, min_dist)
        if min_camera_height is not None:
            min_h = torch.tensor(
                min_camera_height, device=cam_vec.device, dtype=cam_vec.dtype
            )
            if direction[2] > 1e-6:
                needed = (min_h - anchor[2]) / direction[2]
                dist = torch.maximum(dist, needed)
            else:
                camera_position = anchor + cam_vec
                if camera_position[2] < min_h:
                    cam_vec = cam_vec + torch.tensor(
                        [0.0, 0.0, min_h - camera_position[2]],
                        device=cam_vec.device,
                        dtype=cam_vec.dtype,
                    )
                    cam_dist = torch.linalg.norm(cam_vec)
                    if cam_dist > 1e-6:
                        direction = cam_vec / cam_dist
                        dist = cam_dist
        cam_vec = direction * dist

    camera_position = anchor + cam_vec
    R, T = look_at_view_transform(
        eye=camera_position.unsqueeze(0),
        at=anchor.unsqueeze(0),
        up=((0, 0, 1),),
    )

    if fixed_focal_length is not None:
        focal_length = torch.tensor(
            [fixed_focal_length], dtype=torch.float32, device=point_cloud.device
        )
    else:
        focal_length = torch.tensor([[1.0, 1.0]]).to(point_cloud.device)
    if fixed_principal_point is not None:
        principal_point = torch.tensor(
            [fixed_principal_point], dtype=torch.float32, device=point_cloud.device
        )
    else:
        principal_point = torch.tensor([[0.0, 0.0]]).to(point_cloud.device)

    cameras = PerspectiveCameras(
        device=device,
        R=R,
        T=T,
        focal_length=focal_length,
        principal_point=principal_point,
    )

    if calibrate and fixed_focal_length is None and fixed_principal_point is None:
        points_for_fit = fit_points if fit_points is not None else point_cloud.points_padded()
        points_ndc = cameras.transform_points_ndc(points_for_fit)
        points_ndc = points_ndc[..., :3]

        min_ndc = points_ndc.min(dim=1)[0]
        max_ndc = points_ndc.max(dim=1)[0]
        center_ndc = (min_ndc + max_ndc) / 2.0
        size_ndc = (max_ndc - min_ndc).max(dim=1)[0].clamp_min(1e-6)
        scale = (2.0 / size_ndc).unsqueeze(1)

        new_focal_length = focal_length * scale
        new_principal_point = principal_point * scale - center_ndc[:, :2] * scale

        cameras = PerspectiveCameras(
            device=device,
            R=R,
            T=T,
            focal_length=new_focal_length,
            principal_point=new_principal_point,
        )
    return cameras


def render_point_cloud(point_cloud, cameras, image_size, device, point_radius=0.05):
    """
    Render the point cloud.

    Args:
        point_cloud (Pointclouds): The point cloud to render.
        cameras (PerspectiveCameras): The camera settings.
        image_size (int): The size of the output image.
        device (str): The device to use for computation.

    Returns:
        np.ndarray: The rendered image.
        PointsRasterizer: The rasterizer used for rendering.
    """
    raster_settings = PointsRasterizationSettings(
        image_size=image_size, radius=point_radius, points_per_pixel=10
    )
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer, compositor=AlphaCompositor(background_color=255)
    )
    images = renderer(point_cloud)
    image_np = images[0, ..., :3].cpu().numpy()
    return image_np, rasterizer


def compute_depth_map(rasterizer, point_cloud):
    """
    Compute the depth map of the point cloud.

    Args:
        rasterizer (PointsRasterizer): The rasterizer used for rendering.
        point_cloud (Pointclouds): The point cloud to render.

    Returns:
        np.ndarray: The computed depth map.
    """
    fragments = rasterizer(point_cloud)
    depth_map = fragments.zbuf[0].cpu().numpy()
    depth_map = np.min(depth_map, axis=2)

    return depth_map
