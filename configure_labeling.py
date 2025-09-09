#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

DEFAULT_CLASSES = [
    ("Circle", "flat magenta 2D circle"),
    ("Pyramid", "3D with square base + 4 triangular faces meeting at an apex"),
    ("Human", "entire person visible on the field"),
    ("Cone", "3D, circular base tapering to a point"),
    ("Square", "flat magenta 2D square with 4 right angles"),
    ("Hexagon", "flat magenta 2D hexagon with 6 straight sides"),
    ("Kite", "flat magenta 2D kite: two pairs of adjacent equal sides"),
    ("H-pad", "ground marking with a large letter 'H'"),
    ("Chair", "single-person seat with back"),
    ("Cube", "3D box; 6 equal square faces"),
    ("Flood", "standing water/flooded area"),
    ("Cylinder", "yellow 3D solid with two parallel circular bases"),
]

DEFAULT_NOTES = (
    "- Annotate only the classes listed above.\n"
    "- Draw tight bounding boxes around the visible object. If partially occluded, box only the visible portion.\n"
    "- Output coordinates normalized to [0,1] relative to image width/height.\n"
    "- If none of the listed objects are present, return an empty list.\n"
)

CONFIG_FILENAME = "labeling_config.json"


def prompt_yes_no(msg: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        ans = input(f"{msg} {suffix} ").strip().lower()
        if not ans:
            return default
        if ans in {"y", "yes"}:
            return True
        if ans in {"n", "no"}:
            return False
        print("Please answer y or n.")


def read_multiline(prompt: str) -> str:
    print(prompt)
    print("(Finish by submitting an empty line)")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def main():
    print("=== Configure labeling classes and prompt ===")

    # Where to write config
    default_path = Path(CONFIG_FILENAME)
    out_path_str = input(f"Output config path [{default_path}]: ").strip()
    out_path = Path(out_path_str) if out_path_str else default_path

    # Choose classes
    default_class_str = ", ".join(name for name, _ in DEFAULT_CLASSES)
    classes_line = input(
        "Enter class names in order (comma-separated) \n"
        f"(IDs will be assigned from 0..N-1).\n"
        f"Press Enter to use defaults: {default_class_str}\n> "
    ).strip()

    if not classes_line:
        class_names = [name for name, _ in DEFAULT_CLASSES]
        use_defaults = True
    else:
        class_names = [c.strip() for c in classes_line.split(",") if c.strip()]
        use_defaults = False

    classes = []
    if use_defaults and prompt_yes_no("Use default class descriptions?", True):
        classes = [{"name": n, "description": d} for n, d in DEFAULT_CLASSES]
    else:
        if prompt_yes_no("Would you like to enter an optional description for each class?", False):
            for name in class_names:
                desc = input(f"Description for '{name}' (optional): ").strip()
                classes.append({"name": name, "description": desc} if desc else {"name": name})
        else:
            classes = [{"name": name} for name in class_names]

    # Additional prompt notes
    if prompt_yes_no("Would you like to add extra prompt instructions/notes?", True):
        print("You can start from these defaults if you want, or replace them entirely:")
        print("----- Default Notes -----\n" + DEFAULT_NOTES + "-------------------------")
        notes = read_multiline("Enter additional notes (paste or type):")
        if not notes:
            notes = DEFAULT_NOTES
    else:
        notes = DEFAULT_NOTES

    config = {
        "classes": classes,
        "prompt_notes": notes,
    }

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"Saved configuration with {len(classes)} classes to: {out_path}")
    except Exception as e:
        print(f"ERROR: failed to write config to {out_path}: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()

