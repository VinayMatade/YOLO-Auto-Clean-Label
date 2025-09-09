#!/usr/bin/env python3
import argparse
import collections
import hashlib
import json
import os
import random
import shutil
import sys
import time
from typing import Dict, List, Tuple, Set

try:
    from PIL import Image
except Exception as e:
    print("ERROR: Pillow (PIL) is required. Install with: python3 -m pip install pillow", file=sys.stderr)
    sys.exit(2)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def is_image_file(path: str) -> bool:
    _, ext = os.path.splitext(path)
    return ext.lower() in IMAGE_EXTS

def corresponding_label_path(image_path: str) -> str:
    # Convert /subset/images/foo/bar.jpg -> /subset/labels/foo/bar.txt
    parts = image_path.split(os.sep)
    try:
        idx = parts.index("images")
    except ValueError:
        return ""
    parts[idx] = "labels"
    base, _ = os.path.splitext(parts[-1])
    parts[-1] = base + ".txt"
    return os.sep.join(parts)

def file_sha256(path: str, buf_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def image_resolution(path: str) -> Tuple[int, int]:
    try:
        with Image.open(path) as im:
            im.load()
            return im.width, im.height
    except Exception:
        return (0, 0)

def dhash64(path: str, hash_size: int = 8) -> int:
    # difference hash: grayscale, resize to (hash_size+1, hash_size)
    try:
        with Image.open(path) as im:
            im = im.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
            pixels = im.load()
            bits = 0
            for y in range(hash_size):
                for x in range(hash_size):
                    bits <<= 1
                    bits |= 1 if pixels[x, y] > pixels[x + 1, y] else 0
            return bits
    except Exception:
        return 0

def hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()

class ImageInfo:
    __slots__ = ("subset", "image", "label", "sha256", "dhash", "res", "w", "h")

    def __init__(self, subset: str, image: str):
        self.subset = subset
        self.image = image
        self.label = corresponding_label_path(image)
        self.sha256 = None  # type: str | None
        self.dhash = None   # type: int | None
        self.w = 0
        self.h = 0
        self.res = 0

    def load_meta(self):
        if self.sha256 is None:
            self.sha256 = file_sha256(self.image)
        if self.dhash is None:
            self.dhash = dhash64(self.image)
        if self.res == 0:
            w, h = image_resolution(self.image)
            self.w, self.h = w, h
            self.res = w * h

def scan_subset(root: str, subset: str) -> List[ImageInfo]:
    img_dir = os.path.join(root, subset, "images")
    out: List[ImageInfo] = []
    for dirpath, _, filenames in os.walk(img_dir):
        for fn in filenames:
            if is_image_file(fn):
                out.append(ImageInfo(subset, os.path.join(dirpath, fn)))
    return out

def cluster_exact_duplicates(items: List[ImageInfo]) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    for i, info in enumerate(items):
        try:
            info.sha256 = file_sha256(info.image)
        except Exception:
            info.sha256 = "ERR" + info.image
        groups.setdefault(info.sha256, []).append(i)
    return {k: v for k, v in groups.items() if len(v) > 1}

def cluster_near_duplicates(items: List[ImageInfo], threshold: int = 5) -> List[List[int]]:
    # Compute dhash for all
    for info in items:
        if info.dhash is None:
            info.dhash = dhash64(info.image)
    # Bucket by 12-bit prefix to reduce pairwise comparisons
    buckets: Dict[int, List[int]] = {}
    for i, info in enumerate(items):
        prefix = (info.dhash or 0) >> (64 - 12)
        buckets.setdefault(prefix, []).append(i)

    # Union-Find
    parent = list(range(len(items)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for _, idxs in buckets.items():
        m = len(idxs)
        # Compare within bucket
        for a in range(m):
            ia = idxs[a]
            ha = items[ia].dhash or 0
            for b in range(a + 1, m):
                ib = idxs[b]
                hb = items[ib].dhash or 0
                if hamming64(ha, hb) <= threshold:
                    union(ia, ib)

    # Build clusters
    groups: Dict[int, List[int]] = {}
    for i in range(len(items)):
        groups.setdefault(find(i), []).append(i)
    clusters = [g for g in groups.values() if len(g) > 1]
    return clusters

def choose_keep(indices: List[int], items: List[ImageInfo]) -> int:
    # Prefer valid subset, then highest resolution, then lexicographically smallest path
    best = None
    best_idx = None
    for i in indices:
        info = items[i]
        info.load_meta()
        prefer_valid = 0 if info.subset == "valid" else 1
        key = (prefer_valid, -info.res, info.image)
        if best is None or key < best:
            best = key
            best_idx = i
    return best_idx if best_idx is not None else indices[0]

def parse_main_class(label_path: str) -> int:
    # YOLO txt: class cx cy w h
    try:
        with open(label_path, "r") as f:
            counts = collections.Counter()
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                try:
                    cls = int(float(parts[0]))
                except Exception:
                    continue
                counts[cls] += 1
            if not counts:
                return -1
            return counts.most_common(1)[0][0]
    except Exception:
        return -1

def plan_removals(root: str,
                  subsets: List[str],
                  targets: Dict[str, int],
                  dhash_threshold: int,
                  seed: int,
                  cross_split_dedup: bool) -> Dict:
    random.seed(seed)

    # 1) Scan
    per_subset_items: Dict[str, List[ImageInfo]] = {s: scan_subset(root, s) for s in subsets}
    all_items: List[ImageInfo] = []
    subset_offsets: Dict[str, Tuple[int, int]] = {}  # subset -> (start, end)
    for s in subsets:
        start = len(all_items)
        all_items.extend(per_subset_items[s])
        subset_offsets[s] = (start, len(all_items))

    initial_counts = {s: len(per_subset_items[s]) for s in subsets}

    # 2) Exact duplicates (within split or across splits)
    if cross_split_dedup:
        exact_groups = cluster_exact_duplicates(all_items)
    else:
        exact_groups = {}
        for s in subsets:
            start, end = subset_offsets[s]
            part_groups = cluster_exact_duplicates(all_items[start:end])
            # remap indices to global
            for h, idxs in part_groups.items():
                exact_groups[h] = [start + i for i in idxs]

    exact_remove: Set[int] = set()
    exact_kept = []
    for _, idxs in exact_groups.items():
        keep = choose_keep(idxs, all_items)
        exact_kept.append(keep)
        for i in idxs:
            if i != keep:
                exact_remove.add(i)

    # 3) Near-duplicates (exclude already removed)
    remaining_idxs = [i for i in range(len(all_items)) if i not in exact_remove]
    remaining_items = [all_items[i] for i in remaining_idxs]

    if cross_split_dedup:
        near_clusters = cluster_near_duplicates(remaining_items, threshold=dhash_threshold)
        # remap cluster indices back to global
        near_clusters = [[remaining_idxs[i] for i in cluster] for cluster in near_clusters]
    else:
        near_clusters = []
        for s in subsets:
            start, end = subset_offsets[s]
            local_idxs = [i for i in range(start, end) if i not in exact_remove]
            local_items = [all_items[i] for i in local_idxs]
            clusters = cluster_near_duplicates(local_items, threshold=dhash_threshold)
            clusters = [[local_idxs[i] for i in cluster] for cluster in clusters]
            near_clusters.extend(clusters)

    near_remove: Set[int] = set()
    near_kept = []
    for cluster in near_clusters:
        keep = choose_keep(cluster, all_items)
        near_kept.append(keep)
        for i in cluster:
            if i != keep:
                near_remove.add(i)

    removed_by_dup = exact_remove | near_remove

    # 4) After dedup, compute remaining per subset
    remaining_per_subset: Dict[str, List[int]] = {s: [] for s in subsets}
    for i, info in enumerate(all_items):
        if i in removed_by_dup:
            continue
        remaining_per_subset[info.subset].append(i)

    after_dedup_counts = {s: len(remaining_per_subset[s]) for s in subsets}

    # 5) Downsample to targets (class-balanced by main class)
    to_remove_for_target: Set[int] = set()
    sampling_details: Dict[str, Dict] = {}
    for s in subsets:
        target = targets.get(s, None)
        keep_idxs = remaining_per_subset[s]
        n = len(keep_idxs)
        sampling_details[s] = {}
        if target is None or n <= target:
            sampling_details[s]["kept"] = n
            sampling_details[s]["removed"] = 0
            continue
        # Build class buckets
        class_to_idxs: Dict[int, List[int]] = collections.defaultdict(list)
        for gi in keep_idxs:
            info = all_items[gi]
            cls = parse_main_class(info.label) if os.path.isfile(info.label) else -1
            class_to_idxs[cls].append(gi)
        # desired per class
        desired: Dict[int, int] = {}
        for cls, lst in class_to_idxs.items():
            desired[cls] = int(round(target * (len(lst) / n)))
        # Adjust rounding to hit target exactly
        total_desired = sum(desired.values())
        # Fix shortfall/excess by adjusting largest classes
        if total_desired != target:
            # Sort classes by remaining capacity
            classes_sorted = sorted(class_to_idxs.keys(), key=lambda c: len(class_to_idxs[c]) - desired.get(c, 0), reverse=(total_desired < target))
            diff = target - total_desired
            i = 0
            while diff != 0 and classes_sorted:
                c = classes_sorted[i % len(classes_sorted)]
                cap = len(class_to_idxs[c]) - desired.get(c, 0)
                if diff > 0 and cap > 0:
                    desired[c] += 1
                    diff -= 1
                elif diff < 0 and desired.get(c, 0) > 0:
                    desired[c] -= 1
                    diff += 1
                i += 1
        # Sample per class
        keep_set: Set[int] = set()
        for cls, lst in class_to_idxs.items():
            random.shuffle(lst)
            q = min(desired.get(cls, 0), len(lst))
            keep_set.update(lst[:q])
        # If due to rounding/caps we still miss by a few, fill randomly from leftovers
        if len(keep_set) < target:
            leftovers = [gi for gi in keep_idxs if gi not in keep_set]
            random.shuffle(leftovers)
            keep_set.update(leftovers[: (target - len(keep_set))])
        # Everything else is removed
        for gi in keep_idxs:
            if gi not in keep_set:
                to_remove_for_target.add(gi)
        sampling_details[s]["kept"] = len(keep_set)
        sampling_details[s]["removed"] = len(keep_idxs) - len(keep_set)

    final_remove = removed_by_dup | to_remove_for_target

    # Summaries
    summary = {
        "initial_counts": initial_counts,
        "after_dedup_counts": after_dedup_counts,
        "targets": targets,
        "planned_final_counts": {s: after_dedup_counts[s] - sum(1 for gi in to_remove_for_target if all_items[gi].subset == s) for s in subsets},
        "exact_duplicates_groups": len(exact_groups),
        "exact_removed": len(exact_remove),
        "near_duplicates_clusters": len(near_clusters),
        "near_removed": len(near_remove),
        "sampling_details": sampling_details,
    }

    # Build file lists
    def gi_to_record(gi: int):
        info = all_items[gi]
        return {
            "subset": info.subset,
            "image": info.image,
            "label": info.label if info.label else "",
        }

    plan = {
        "summary": summary,
        "remove_list": [gi_to_record(gi) for gi in sorted(final_remove)],
        "dedup_removed": [gi_to_record(gi) for gi in sorted(removed_by_dup)],
        "downsample_removed": [gi_to_record(gi) for gi in sorted(to_remove_for_target)],
    }
    return plan


def apply_plan(plan: Dict, backup_root: str):
    removed = plan.get("remove_list", [])
    # Create backup directories mirroring subset/images and subset/labels
    for rec in removed:
        subset = rec["subset"]
        img_src = rec["image"]
        lbl_src = rec.get("label")
        # backup paths
        img_rel = os.path.relpath(img_src, start=os.path.join(os.path.dirname(os.path.dirname(img_src)), subset))
        # img_rel like images/..../file.jpg
        img_dst = os.path.join(backup_root, subset, img_rel)
        os.makedirs(os.path.dirname(img_dst), exist_ok=True)
        try:
            if os.path.exists(img_src):
                shutil.move(img_src, img_dst)
        except Exception as e:
            print(f"WARN: failed to move image {img_src} -> {img_dst}: {e}", file=sys.stderr)
        if lbl_src and os.path.isfile(lbl_src):
            lbl_rel = os.path.relpath(lbl_src, start=os.path.join(os.path.dirname(os.path.dirname(lbl_src)), subset))
            lbl_dst = os.path.join(backup_root, subset, lbl_rel)
            os.makedirs(os.path.dirname(lbl_dst), exist_ok=True)
            try:
                shutil.move(lbl_src, lbl_dst)
            except Exception as e:
                print(f"WARN: failed to move label {lbl_src} -> {lbl_dst}: {e}", file=sys.stderr)

def main():
    p = argparse.ArgumentParser(description="Deduplicate YOLO images and downsample to targets, with backup.")
    p.add_argument("--root", required=True, help="Dataset root containing train/valid subfolders")
    p.add_argument("--subsets", nargs="*", default=["train", "valid"], help="Subsets to process (default: train valid)")
    p.add_argument("--train-target", type=int, default=None)
    p.add_argument("--valid-target", type=int, default=None)
    p.add_argument("--dhash-threshold", type=int, default=5, help="Hamming distance threshold for near-duplicates (default 5)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cross-split-dedup", action="store_true", help="Deduplicate across splits (avoid leakage)")
    p.add_argument("--apply", action="store_true", help="Apply changes (move files to backup). Default is dry-run")
    p.add_argument("--backup-dir", default=None, help="Optional backup directory. If not set, a timestamped backup is created under root")
    p.add_argument("--output-json", default=None, help="Write plan summary JSON to this path as well as stdout")
    args = p.parse_args()

    targets = {}
    if args.train_target is not None:
        targets["train"] = args.train_target
    if args.valid_target is not None:
        targets["valid"] = args.valid_target

    plan = plan_removals(
        root=args.root,
        subsets=args.subsets,
        targets=targets,
        dhash_threshold=args.dhash_threshold,
        seed=args.seed,
        cross_split_dedup=args.cross_split_dedup,
    )

    # Print plan summary
    print(json.dumps(plan["summary"], indent=2))

    if args.output_json:
        try:
            with open(args.output_json, "w") as f:
                json.dump(plan, f, indent=2)
        except Exception as e:
            print(f"WARN: failed to write JSON plan to {args.output_json}: {e}", file=sys.stderr)

    if not args.apply:
        print("\nDry-run complete. To apply, re-run with --apply.")
        return

    # Apply
    backup_root = args.backup_dir
    if not backup_root:
        ts = time.strftime("backup_removed_%Y%m%d_%H%M%S")
        backup_root = os.path.join(args.root, ts)
    os.makedirs(backup_root, exist_ok=True)

    apply_plan(plan, backup_root=backup_root)

    # Write final plan to backup folder for record
    try:
        with open(os.path.join(backup_root, "cleanup_plan.json"), "w") as f:
            json.dump(plan, f, indent=2)
    except Exception as e:
        print(f"WARN: failed to write cleanup_plan.json to backup folder: {e}", file=sys.stderr)

    print(f"Applied. Removed files were moved to: {backup_root}")


if __name__ == "__main__":
    main()

