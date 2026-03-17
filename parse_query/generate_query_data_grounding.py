import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List

from openai import OpenAI


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def find_task_files(root: Path, task_name: str) -> List[Path]:
    if root.is_file():
        return [root]
    matches = []
    for path in root.rglob(task_name):
        if path.is_file():
            matches.append(path)
    return sorted(matches)


def extract_caption(task: Dict[str, Any]) -> str:
    return (
        task.get("caption")
        or task.get("description")
        or task.get("query")
        or ""
    )


def extract_target_id(task: Dict[str, Any]) -> Any:
    return (
        task.get("target_id")
        if task.get("target_id") is not None
        else task.get("object_id")
    )


def extract_target_name(task: Dict[str, Any]) -> Any:
    return (
        task.get("target_name")
        if task.get("target_name") is not None
        else task.get("object_name")
    )


def parse_with_vlm(prompt: str, caption: str, client: OpenAI, model_name: str) -> Any:
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": [{"type": "text", "text": f"Query: {caption}"}]},
    ]
    resp = client.chat.completions.create(model=model_name, messages=messages)
    text = resp.choices[0].message.content
    try:
        text = text.replace("'", '"')
        return json.loads(text)
    except Exception:
        return text


def build_output(task: Dict[str, Any], parsed_query: Any) -> Dict[str, Any]:
    return {
        "scan_id": task.get("scan_id") or task.get("scene_id"),
        "target_id": extract_target_id(task),
        "target_name": extract_target_name(task),
        "caption": extract_caption(task),
        "parsed_query": parsed_query,
        "unique": task.get("unique"),
    }


def resolve_scene_id(task: Dict[str, Any]) -> str:
    return task.get("scan_id") or task.get("scene_id") or "unknown_scene"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse task.json files with VLM for grounding (scene-level output)."
    )
    parser.add_argument(
        "--input_path",
        required=True,
        help="Task folder or parent folder to scan for task.json.",
    )
    parser.add_argument(
        "--task_name",
        default="task.json",
        help="Task file name to search for (default: task.json).",
    )
    parser.add_argument(
        "--prompt_file",
        required=True,
        help="Path to the prompt file.",
    )
    parser.add_argument(
        "--openai_api_key",
        required=True,
        help="OpenAI API key.",
    )
    parser.add_argument(
        "--openai_api_base",
        required=True,
        help="OpenAI API base URL.",
    )
    parser.add_argument(
        "--model_name",
        required=True,
        help="Model name for OpenAI API.",
    )
    parser.add_argument(
        "--output_dir",
        default="data/ScanRefer/query_grounding",
        help="Output root (scene-level JSON files).",
    )
    parser.add_argument(
        "--skip_existing",
        type=lambda v: v.lower() in ("1", "true", "yes", "y"),
        default=True,
        help="Skip if output file already exists (default: True).",
    )
    args = parser.parse_args()

    input_root = Path(args.input_path)
    task_files = find_task_files(input_root, args.task_name)
    if not task_files:
        raise FileNotFoundError(f"No {args.task_name} found under {input_root}")

    prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    client = OpenAI(api_key=args.openai_api_key, base_url=args.openai_api_base)
    output_dir = Path(args.output_dir)

    # Group task files by scene_id
    grouped: Dict[str, List[Path]] = {}
    for task_path in task_files:
        task = load_json(task_path)
        scene_id = resolve_scene_id(task)
        grouped.setdefault(scene_id, []).append(task_path)

    for scene_id, paths in grouped.items():
        out_path = output_dir / f"{scene_id}.json"
        if args.skip_existing and out_path.exists():
            print(f"Skip {scene_id}: output exists.")
            continue

        outputs: List[Dict[str, Any]] = []
        for task_path in paths:
            task = load_json(task_path)
            caption = extract_caption(task)
            if not caption:
                print(f"Skip {task_path}: missing caption/description.")
                continue
            parsed_query = parse_with_vlm(prompt, caption, client, args.model_name)
            outputs.append(build_output(task, parsed_query))

        save_json(out_path, outputs)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
