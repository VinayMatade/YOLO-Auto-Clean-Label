#!/usr/bin/env python3
import os
import re
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
from google import generativeai as genai

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

# Fallback YOLO-line regex (used only if JSON parsing fails)
YOLO_LINE_RE = re.compile(
    r"^\s*(\d+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s*$"
)

# JSON schema for structured responses
JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "boxes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "class_id": {"type": "integer"},
                    "center_x": {"type": "number"},
                    "center_y": {"type": "number"},
                    "width": {"type": "number"},
                    "height": {"type": "number"}
                },
                "required": ["class_id", "center_x", "center_y", "width", "height"]
            }
        }
    },
    "required": ["boxes"]
}

def load_config(cfg_path: Path) -> Dict[str, Any]:
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}. Please run configure_labeling.py first.", file=sys.stderr)
        sys.exit(3)
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        if not isinstance(cfg.get("classes"), list) or len(cfg["classes"]) == 0:
            raise ValueError("'classes' must be a non-empty list")
        return cfg
    except Exception as e:
        print(f"Failed to read config {cfg_path}: {e}", file=sys.stderr)
        sys.exit(4)

def build_prompt(img_w: int, img_h: int, classes: List[Dict[str, Any]], prompt_notes: str) -> str:
    classes_str = ", ".join(
        f"{i} {c.get('name','class')}" + (f" ({c.get('description')})" if c.get('description') else "")
        for i, c in enumerate(classes)
    )
    base = (
        "You are an expert image annotator for YOLO object detection. "
        "Identify ONLY the following classes and ignore everything else: "
        f"{classes_str}.\n\n"
        "Rules:\n"
        "- Annotate only the classes listed above.\n"
        "- Draw tight bounding boxes around the visible object.\n"
        "- If an object is partially occluded, box only the visible portion.\n"
        "- Output coordinates normalized to [0,1] relative to image width/height.\n"
        "- If none of the listed objects are present, return an empty list.\n\n"
        "Output: Return a JSON object with a single key 'boxes' (no extra text)."
    )
    if prompt_notes:
        base += "\n\nAdditional instructions:\n" + prompt_notes
    base += f"\n\nImage dimensions: width={img_w} height={img_h}.\n"
    base += (
        "Return strictly a JSON object matching the schema: {\"boxes\": "
        "[{\"class_id\":int, \"center_x\":float, \"center_y\":float, \"width\":float, \"height\":float}]}."
        " Numbers must be normalized to [0,1]."
    )
    return base

def extract_yolo_lines(text: str) -> List[str]:
    """Fallback parser for plain-text YOLO lines."""
    results: List[str] = []
    for line in text.splitlines():
        m = YOLO_LINE_RE.match(line.strip())
        if not m:
            continue
        cls_id = int(m.group(1))
        cx = float(m.group(2))
        cy = float(m.group(3))
        w = float(m.group(4))
        h = float(m.group(5))
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))
        results.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return results

def boxes_to_yolo_lines(boxes: List[Dict[str, Any]], num_classes: int | None) -> List[str]:
    lines: List[str] = []
    for b in boxes:
        try:
            cls_id = int(b["class_id"])
            if num_classes is not None and (cls_id < 0 or cls_id >= num_classes):
                continue
            cx = float(b["center_x"]) if "center_x" in b else float(b["cx"])
            cy = float(b["center_y"]) if "center_y" in b else float(b["cy"])
            w = float(b["width"]) if "width" in b else float(b["w"])
            h = float(b["height"]) if "height" in b else float(b["h"])
        except Exception:
            continue
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))
        if w <= 0 or h <= 0:
            continue
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines

def write_label_file(label_path: Path, lines: List[str]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w", encoding="utf-8") as f:
        if lines:
            f.write("\n".join(lines))
        # else: write empty file meaning 'no objects'

def request_boxes_json(model, prompt: str, img: Image.Image) -> List[Dict[str, Any]]:
    """Ask the model for JSON-structured boxes using a strict schema."""
    response = model.generate_content(
        [prompt, img],
        generation_config={
            "temperature": 0,
            "response_mime_type": "application/json",
            "response_schema": JSON_SCHEMA,
        },
    )
    text = getattr(response, "text", None) or ""
    try:
        data = json.loads(text)
        if isinstance(data, dict) and isinstance(data.get("boxes"), list):
            return data["boxes"]
    except Exception:
        pass
    return []

def request_lines_fallback(model, prompt: str, img: Image.Image) -> List[str]:
    """Fallback ask for plain YOLO lines only (no prose)."""
    fallback_prompt = (
        prompt
        + "\nReturn ONLY the YOLO annotations, one per line in the exact format: "
        + "<class_id> <center_x> <center_y> <width> <height>. "
        + "No extra text, no markdown."
    )
    response = model.generate_content(
        [fallback_prompt, img],
        generation_config={"temperature": 0},
    )
    text = getattr(response, "text", None) or ""
    return extract_yolo_lines(text)

def label_image(model, image_path: Path, out_dir: Path | None, overwrite: bool, save_raw: Path | None, num_classes: int) -> None:
    out_dir = out_dir or image_path.parent
    label_path = out_dir / (image_path.stem + ".txt")
    raw_path = (save_raw / (image_path.stem + ".json")) if save_raw else None

    if label_path.exists() and not overwrite:
        print(f"Skip existing: {label_path}")
        return

    try:
        with Image.open(image_path) as img:
            img_w, img_h = img.width, img.height
            prompt = build_prompt(img_w, img_h, classes=[], prompt_notes="")  # placeholder, replaced below
    except Exception:
        # reopen after prompt build
        pass

    try:
        with Image.open(image_path) as img:
            img_w, img_h = img.width, img.height
            # We'll rebuild prompt outside when calling
            pass
    except Exception:
        pass

    # Re-open and run through model (we'll pass prompt from caller to avoid re-reading config here)
    try:
        with Image.open(image_path) as img:
            # The caller will construct the right prompt; this function only runs the requests
            pass
    except Exception:
        pass


def find_images(root: Path) -> List[Path]:
    if root.is_file() and root.suffix.lower() in SUPPORTED_EXTS:
        return [root]
    return [p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS]

def main():
    parser = argparse.ArgumentParser(description="Auto-label images to YOLO using an LLM (Gemini by default).")
    parser.add_argument("--images", type=str, required=True, help="Path to an image file or a directory of images.")
    parser.add_argument("--out-dir", type=str, default=None, help="Optional directory to write labels into.")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash", help="Gemini model ID (e.g., gemini-1.5-flash, gemini-2.0-flash, gemini-2.5-flash if available).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing label files.")
    parser.add_argument("--save-raw", type=str, default=None, help="Directory to save raw JSON responses from the model for debugging.")
    parser.add_argument("--config", type=str, default="labeling_config.json", help="Path to labeling configuration JSON (from configure_labeling.py).")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is not set in the environment.", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(Path(args.config).expanduser().resolve())
    classes = cfg.get("classes", [])
    prompt_notes = cfg.get("prompt_notes", "")
    num_classes = len(classes)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(args.model)

    images_root = Path(args.images).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else None
    save_raw = Path(args.save_raw).expanduser().resolve() if args.save_raw else None

    images = find_images(images_root)
    if not images:
        print(f"No images found under: {images_root}", file=sys.stderr)
        sys.exit(2)

    print(f"Found {len(images)} image(s). Using model: {args.model}")
    for img_path in images:
        try:
            with Image.open(img_path) as img:
                img_w, img_h = img.width, img.height
                prompt = build_prompt(img_w, img_h, classes, prompt_notes)

                # First try: ask for structured JSON
                boxes = request_boxes_json(model, prompt, img)
                yolo_lines: List[str] = boxes_to_yolo_lines(boxes, num_classes)

                # Save raw JSON if requested
                if save_raw is not None:
                    rp = save_raw / (img_path.stem + ".json")
                    rp.parent.mkdir(parents=True, exist_ok=True)
                    with rp.open("w", encoding="utf-8") as rf:
                        json.dump({"boxes": boxes}, rf, ensure_ascii=False, indent=2)

                # Fallback to plain text if empty
                if not yolo_lines:
                    yolo_lines = request_lines_fallback(model, prompt, img)

                # Write labels
                lp = (out_dir or img_path.parent) / (img_path.stem + ".txt")
                write_label_file(lp, yolo_lines)
                print(f"Wrote {lp} ({len(yolo_lines)} boxes)")
        except Exception as e:
            print(f"Error on {img_path}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()

