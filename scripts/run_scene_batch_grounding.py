import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value.")


def calc_iou(box_a, box_b):
    max_a = [box_a[0] + box_a[3] / 2, box_a[1] + box_a[4] / 2, box_a[2] + box_a[5] / 2]
    max_b = [box_b[0] + box_b[3] / 2, box_b[1] + box_b[4] / 2, box_b[2] + box_b[5] / 2]
    min_max = [min(max_a[0], max_b[0]), min(max_a[1], max_b[1]), min(max_a[2], max_b[2])]

    min_a = [box_a[0] - box_a[3] / 2, box_a[1] - box_a[4] / 2, box_a[2] - box_a[5] / 2]
    min_b = [box_b[0] - box_b[3] / 2, box_b[1] - box_b[4] / 2, box_b[2] - box_b[5] / 2]
    max_min = [max(min_a[0], min_b[0]), max(min_a[1], min_b[1]), max(min_a[2], min_b[2])]

    if not (min_max[0] > max_min[0] and min_max[1] > max_min[1] and min_max[2] > max_min[2]):
        return 0.0

    intersection = (
        (min_max[0] - max_min[0])
        * (min_max[1] - max_min[1])
        * (min_max[2] - max_min[2])
    )
    vol_a = box_a[3] * box_a[4] * box_a[5]
    vol_b = box_b[3] * box_b[4] * box_b[5]
    union = vol_a + vol_b - intersection
    return intersection / union if union > 0 else 0.0


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_task_json(task_dir):
    task_path = task_dir / "task.json"
    if task_path.exists():
        return task_path
    candidates = sorted(task_dir.glob("*.json"))
    for path in candidates:
        if path.name == "gt.json":
            continue
        return path
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_root", required=True, help="Scene root, e.g. assets/scene0000_00_2")
    parser.add_argument("--output_root", required=True, help="Output parent dir, e.g. outputs/candidates/batch_01")
    parser.add_argument("--pcd_dir", required=True)
    parser.add_argument("--openai_api_key", required=True)
    parser.add_argument("--openai_api_base", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--use_image", type=str2bool, default=True)
    parser.add_argument("--num_global_views", type=int, default=3)
    parser.add_argument("--candidate_image_per_id", type=int, default=10)
    parser.add_argument("--max_candidate_images", type=int, default=25)
    parser.add_argument("--candidate_image_collage", type=str2bool, default=False)
    parser.add_argument("--candidate_image_primary_smallest", type=str2bool, default=False)
    parser.add_argument("--candidate_image_primary_only", type=str2bool, default=False)
    parser.add_argument("--adaptive_point_radius", type=str2bool, default=False)
    parser.add_argument("--fixed_camera_params", default="")
    parser.add_argument("--filter_ceiling_points", type=str2bool, default=False)
    parser.add_argument("--ceiling_percentile", type=float, default=95.0)
    parser.add_argument("--compress_collage", type=str2bool, default=True)
    parser.add_argument("--collage_quality", type=int, default=85)
    parser.add_argument("--collage_max_size", type=int, default=0)
    parser.add_argument("--extra_args", default="", help="Extra args passed to inference script.")
    parser.add_argument("--force", type=str2bool, default=False)
    args = parser.parse_args()

    scene_root = Path(args.scene_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    total = 0
    correct_25 = 0
    correct_50 = 0
    skipped = 0

    for task_dir in sorted(p for p in scene_root.iterdir() if p.is_dir()):
        task_json = resolve_task_json(task_dir)
        gt_json = task_dir / "gt.json"
        if task_json is None or not gt_json.exists():
            print(f"Skipping {task_dir}: missing task.json or gt.json")
            continue

        gt_data = load_json(gt_json)
        if isinstance(gt_data, dict) and gt_data.get("target_id") is None:
            print(f"Skipping {task_dir}: target_id is null in gt.json")
            skipped += 1
            continue

        scene_data = load_json(task_json)
        if isinstance(scene_data, dict):
            scene_id = scene_data.get("scene_id", task_dir.name)
        elif isinstance(scene_data, list) and scene_data:
            scene_id = scene_data[0].get("scene_id", task_dir.name)
        else:
            scene_id = task_dir.name

        out_dir = output_root / task_dir.name
        pred_path = out_dir / "pred" / f"{scene_id}.json"
        if pred_path.exists() and not args.force:
            skipped += 1
        else:
            cmd = [
                "python",
                "inference/inference_grounding.py",
                "--output_dir",
                str(out_dir),
                "--scene_json",
                str(task_json),
                "--gt_json",
                str(gt_json),
                "--pcd_dir",
                args.pcd_dir,
                "--openai_api_key",
                args.openai_api_key,
                "--openai_api_base",
                args.openai_api_base,
                "--model_name",
                args.model_name,
                "--use_image",
                str(args.use_image),
                "--num_global_views",
                str(args.num_global_views),
                "--candidate_image_root",
                str(task_dir),
                "--candidate_image_per_id",
                str(args.candidate_image_per_id),
                "--max_candidate_images",
                str(args.max_candidate_images),
            ]
            if args.candidate_image_collage:
                cmd += ["--candidate_image_collage", "True"]
            if args.candidate_image_primary_smallest:
                cmd += ["--candidate_image_primary_smallest", "True"]
            if args.candidate_image_primary_only:
                cmd += ["--candidate_image_primary_only", "True"]
            if args.adaptive_point_radius:
                cmd += ["--adaptive_point_radius", "True"]
            if args.fixed_camera_params:
                cmd += ["--fixed_camera_params", args.fixed_camera_params]
            if args.filter_ceiling_points:
                cmd += ["--filter_ceiling_points", "True"]
            cmd += ["--ceiling_percentile", str(args.ceiling_percentile)]
            if args.compress_collage:
                cmd += ["--compress_collage", "True"]
            cmd += ["--collage_quality", str(args.collage_quality)]
            if args.collage_max_size > 0:
                cmd += ["--collage_max_size", str(args.collage_max_size)]
            if args.extra_args:
                cmd += shlex.split(args.extra_args)

            subprocess.run(cmd, check=True)

        if pred_path.exists():
            preds = load_json(pred_path)
            for item in preds:
                total += 1
                if item.get("pred_bbox") is not None and item.get("gt_bbox") is not None:
                    iou = calc_iou(item["gt_bbox"], item["pred_bbox"])
                else:
                    iou = 0.0
                if iou >= 0.25:
                    correct_25 += 1
                if iou >= 0.5:
                    correct_50 += 1
                acc_25 = correct_25 / total
                acc_50 = correct_50 / total
                print(
                    f"Global Accuracy@0.25: {acc_25:.4f} | "
                    f"Global Accuracy@0.50: {acc_50:.4f}"
                )

    if total == 0:
        print("No predictions found.")
        return

    acc_25 = correct_25 / total
    acc_50 = correct_50 / total
    print(f"Processed {total} queries across {scene_root}. Skipped: {skipped}")
    print(f"Accuracy@0.25: {acc_25:.4f} | Accuracy@0.50: {acc_50:.4f}")


if __name__ == "__main__":
    main()
