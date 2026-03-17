#!/usr/bin/env python3
"""
Convert ARKitScenes data format to ScanNet format for compatibility with existing grounding pipeline.

Usage:
    python scripts/convert_arkitscenes_to_scannet_format.py \
        --input_root /home/yf/yf/guazai/data/41048085 \
        --output_root /home/yf/yf/guazai/data/converted/41048085

"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List


def load_json(path: Path) -> Any:
    """Load JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    """Save JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def convert_query_to_task(query_data: Dict[str, Any], gt_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert ARKitScenes query.json to ScanNet task.json format.

    ARKitScenes query.json:
    {
        "scan_id": "arkitscenes/Training/47333462",
        "target_id": 2,
        "text": "choose the cabinet that is farthest from the bed",
        "target": "cabinet",
        "anchors": ["bed"],
        "anchor_ids": [1],
        ...
    }

    ScanNet task.json:
    {
        "scan_id": "scene_id",
        "target_id": "2",
        "target_name": "cabinet",
        "caption": "choose the cabinet that is farthest from the bed",
        "parsed_query": {
            "Target": "cabinet",
            "Anchor": "bed"
        },
        "unique": false
    }
    """
    # Extract scene_id from scan_id
    scan_id = query_data.get("scan_id", "")
    if "/" in scan_id:
        scene_id = scan_id.split("/")[-1]
    else:
        scene_id = scan_id

    # Get target and anchors
    target = query_data.get("target", "")
    anchors = query_data.get("anchors", [])

    # Create parsed_query
    parsed_query = {
        "Target": target
    }

    # Handle anchors - can be single or multiple
    if anchors:
        if len(anchors) == 1:
            parsed_query["Anchor"] = anchors[0]
        else:
            parsed_query["Anchor"] = anchors
    else:
        parsed_query["Anchor"] = ""

    # Build task.json
    task_data = {
        "scan_id": scene_id,
        "target_id": str(query_data.get("target_id", "")),
        "target_name": target,
        "caption": query_data.get("text", ""),
        "parsed_query": parsed_query,
        "unique": False  # ARKitScenes doesn't have this field, default to False
    }

    return task_data


def convert_task_directory(
    input_task_dir: Path,
    output_task_dir: Path,
    copy_images: bool = False,
    copy_ply: bool = False,
    verbose: bool = True
) -> bool:
    """
    Convert a single task directory from ARKitScenes to ScanNet format.

    Returns:
        True if conversion successful, False otherwise.
    """
    if verbose:
        print(f"Converting {input_task_dir.name}...")

    # Check required files
    query_json = input_task_dir / "query.json"
    gt_json = input_task_dir / "gt.json"

    if not query_json.exists():
        print(f"  ⚠️  Missing query.json, skipping")
        return False

    if not gt_json.exists():
        print(f"  ⚠️  Missing gt.json, skipping")
        return False

    # Create output directory
    output_task_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    query_data = load_json(query_json)
    gt_data = load_json(gt_json)

    # Convert query.json to task.json
    task_data = convert_query_to_task(query_data, gt_data)
    save_json(output_task_dir / "task.json", task_data)

    # Copy query.json (keep original for reference)
    save_json(output_task_dir / "query.json", query_data)

    # Copy gt.json (no conversion needed)
    save_json(output_task_dir / "gt.json", gt_data)

    # Copy ref.json if exists
    ref_json = input_task_dir / "ref.json"
    if ref_json.exists():
        ref_data = load_json(ref_json)
        save_json(output_task_dir / "ref.json", ref_data)

    # Copy anchor files
    for anchor_file in input_task_dir.glob("anchor_*.json"):
        anchor_data = load_json(anchor_file)
        save_json(output_task_dir / anchor_file.name, anchor_data)

    # Copy full_pcd.ply
    full_pcd = input_task_dir / "full_pcd.ply"
    if full_pcd.exists():
        if copy_ply:
            shutil.copy2(full_pcd, output_task_dir / "full_pcd.ply")
        else:
            # Create symlink instead of copying (saves space)
            output_ply = output_task_dir / "full_pcd.ply"
            if not output_ply.exists():
                output_ply.symlink_to(full_pcd.resolve())

    # Copy candidate object directories
    for candidate_dir in input_task_dir.iterdir():
        if not candidate_dir.is_dir():
            continue

        # Skip if not a numeric directory (candidate ID)
        if not candidate_dir.name.isdigit():
            continue

        output_candidate_dir = output_task_dir / candidate_dir.name
        output_candidate_dir.mkdir(parents=True, exist_ok=True)

        # Copy images
        for img_file in candidate_dir.glob("*.png"):
            if copy_images:
                shutil.copy2(img_file, output_candidate_dir / img_file.name)
            else:
                # Create symlink
                output_img = output_candidate_dir / img_file.name
                if not output_img.exists():
                    output_img.symlink_to(img_file.resolve())

        # Copy PLY files
        for ply_file in candidate_dir.glob("*.ply"):
            if copy_ply:
                shutil.copy2(ply_file, output_candidate_dir / ply_file.name)
            else:
                # Create symlink
                output_ply = output_candidate_dir / ply_file.name
                if not output_ply.exists():
                    output_ply.symlink_to(ply_file.resolve())

    if verbose:
        print(f"  ✓ Converted successfully")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert ARKitScenes data format to ScanNet format"
    )
    parser.add_argument(
        "--input_root",
        type=str,
        required=True,
        help="Input root directory. Single scene (e.g., /path/to/47333462) or batch root containing multiple scene dirs (use with --batch)"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Output root directory"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch mode: input_root contains multiple scene subdirectories to convert"
    )
    parser.add_argument(
        "--copy_images",
        action="store_true",
        help="Copy images instead of creating symlinks (uses more disk space)"
    )
    parser.add_argument(
        "--copy_ply",
        action="store_true",
        help="Copy PLY files instead of creating symlinks (uses more disk space)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print verbose output"
    )

    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    if not input_root.exists():
        print(f"❌ Input directory does not exist: {input_root}")
        return 1

    # Determine list of (input_scene_dir, output_scene_dir) pairs
    if args.batch:
        scene_dirs = sorted([d for d in input_root.iterdir() if d.is_dir() and d.name.isdigit()])
        pairs = [(d, output_root / d.name) for d in scene_dirs]
        print(f"Batch mode: found {len(pairs)} scenes in {input_root}")
    else:
        pairs = [(input_root, output_root)]

    total_success = 0
    total_failed = 0

    for scene_input, scene_output in pairs:
        if args.batch:
            print(f"\n{'='*60}")
            print(f"Scene: {scene_input.name}")

        try:
            task_dirs = sorted([d for d in scene_input.iterdir() if d.is_dir() and d.name.isdigit()])
        except OSError as e:
            print(f"  ⚠️  I/O error reading scene, skipping: {e}")
            continue
        if not task_dirs:
            print(f"  ⚠️  No task directories found, skipping")
            continue

        print(f"  Found {len(task_dirs)} task directories")
        for task_dir in task_dirs:
            success = convert_task_directory(
                task_dir,
                scene_output / task_dir.name,
                copy_images=args.copy_images,
                copy_ply=args.copy_ply,
                verbose=args.verbose
            )
            if success:
                total_success += 1
            else:
                total_failed += 1

        results_json = scene_input / "results.json"
        if results_json.exists():
            shutil.copy2(results_json, scene_output / "results.json")
            if args.verbose:
                print(f"  ✓ Copied results.json")

    print()
    print("=" * 60)
    print(f"Conversion complete! ✓ {total_success}  ✗ {total_failed}  Total: {total_success + total_failed}")
    if not args.copy_images and not args.copy_ply:
        print("ℹ️  Images and PLY files are symlinked (not copied).")

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    exit(main())
