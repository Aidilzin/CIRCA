"""
oversample_minority_classes.py
================================
CIRCA -- Phase 0 Data Preparation (updated: 2026-05-27)
Tiered minority-class oversampling for unified_pcb_v3 training split.

Class distribution before oversampling (BEFORE counts from rebuilt dataset):
  0: missing_hole           390  -- CRITICAL (0% recall in baseline -> promoted to 5x)
  1: mouse_bite           2,574  -- good
  2: open_circuit         1,892  -- good
  3: short               3,556  -- DOMINANT -> EXCLUDED
  4: excess_solder          304  -- CRITICAL (51% recall in baseline -> added to 5x)
  5: insufficient_solder  3,568  -- DOMINANT -> EXCLUDED
  6: cold_solder_joint      270  -- CRITICAL (5x, working well)

Tier rules (v4 -- updated based on Phase 1/2 confusion matrix analysis):
  TIER 1 -- Critical (5x): class 0 (missing_hole), class 4 (excess_solder), class 6 (cold_solder_joint)
  EXCLUDED: classes 3 (short) and 5 (insufficient_solder) -- dominant, not oversampled

Usage:
    python scripts/oversample_minority_classes.py
    python scripts/oversample_minority_classes.py --dry-run
    python scripts/oversample_minority_classes.py --dataset-root datasets/unified_pcb_v3
    python scripts/oversample_minority_classes.py --dataset-root datasets/unified_pcb_v3_preproc
"""

import argparse
import shutil
import random
from collections import defaultdict, Counter
from pathlib import Path

# -- Configuration -------------------------------------------------------------
DATASET_ROOT = Path("datasets/unified_pcb_v3")
SEED = 42

CLASS_NAMES = [
    "missing_hole",        # 0  CRITICAL -- 5x (0% recall in baseline)
    "mouse_bite",          # 1  good
    "open_circuit",        # 2  good
    "short",               # 3  EXCLUDED -- dominant
    "excess_solder",       # 4  CRITICAL -- 5x (51% recall in baseline)
    "insufficient_solder", # 5  EXCLUDED -- dominant
    "cold_solder_joint",   # 6  CRITICAL -- 5x
]

# Classes excluded from oversampling (dominant -- would worsen imbalance)
EXCLUDED_CLASSES = {3, 5}

# Tier definitions: class_id -> max_total_copies (including original)
# max_copies=5 means 4 extra copies -> 5x the original count
# v4: all three weak classes promoted to CRITICAL (5x) based on Phase 1/2 confusion matrix
CLASS_TIER = {
    0: 5,  # missing_hole   -- CRITICAL (390 instances, 0% recall  -> 5x)
    4: 5,  # excess_solder  -- CRITICAL (304 instances, 51% recall -> 5x)
    6: 5,  # cold_solder_joint -- CRITICAL (270 instances, 96% recall -> maintain 5x)
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CIRCA -- Tiered minority-class oversampler for unified_pcb_v3."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Analyse and report without copying any files."
    )
    parser.add_argument(
        "--dataset-root", type=str, default=str(DATASET_ROOT),
        help="Root of the dataset (default: datasets/unified_pcb_v3)."
    )
    return parser.parse_args()


def find_image_for_label(label_path: Path, images_dir: Path) -> Path | None:
    """Return the corresponding image file for a label, or None if not found."""
    stem = label_path.stem
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / (stem + ext)
        if candidate.exists():
            return candidate
    return None


def get_classes_in_label(label_path: Path) -> set[int]:
    """Return the set of class IDs present in a YOLO label file."""
    classes = set()
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                classes.add(int(parts[0]))
    return classes


def compute_class_distribution(labels_dir: Path) -> tuple[Counter, dict[int, list[Path]]]:
    """
    Count annotation instances per class and map each class to its label files.
    Returns:
        instance_counts: Counter[class_id -> int]
        class_to_labels: dict[class_id -> list[Path]]
    """
    instance_counts: Counter = Counter()
    class_to_labels: dict[int, list[Path]] = defaultdict(list)

    for label_file in sorted(labels_dir.glob("*.txt")):
        # Skip already-oversampled copies -- only count/use originals as sources
        if "_os" in label_file.stem:
            continue
        classes_in_file = get_classes_in_label(label_file)
        for cls in classes_in_file:
            instance_counts[cls] += 1
            class_to_labels[cls].append(label_file)

    return instance_counts, class_to_labels


def print_distribution(title: str, counts: Counter) -> None:
    n_classes = len(CLASS_NAMES)
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")
    total = sum(counts.values())
    for cls_id in range(n_classes):
        count = counts.get(cls_id, 0)
        bar = "#" * min(35, int(35 * count / max(total, 1)))
        if cls_id in EXCLUDED_CLASSES:
            tag = " < EXCLUDED (dominant)"
        elif cls_id in CLASS_TIER and CLASS_TIER[cls_id] >= 5:
            tag = f" < CRITICAL -> {CLASS_TIER[cls_id]}x"
        elif cls_id in CLASS_TIER:
            tag = f" < moderate -> {CLASS_TIER[cls_id]}x"
        else:
            tag = ""
        pct = 100 * count / total if total else 0
        print(f"  {cls_id:2d}  {CLASS_NAMES[cls_id]:<24} {count:6d} ({pct:4.1f}%)  {bar}{tag}")
    print(f"{'-' * 65}")
    print(f"  Total annotation instances: {total}")


def oversample(
    labels_dir: Path,
    images_dir: Path,
    class_to_labels: dict[int, list[Path]],
    dry_run: bool,
    rng: random.Random,
) -> int:
    """
    For each tiered class, duplicate its unique training label files up to
    CLASS_TIER[cls]-1 extra times (total copies including original = CLASS_TIER[cls]).

    Images containing EXCLUDED classes alongside minority classes are still copied
    -- excluded classes are not used as the primary selection criterion, but their
    co-occurrence annotations are preserved without amplifying dominance beyond what
    already exists.

    Returns the total number of file pairs written (or would write in dry-run).
    """
    # Build map: label_path -> max_copies it qualifies for
    # A label may contain multiple classes; use the highest tier applicable
    label_max_copies: dict[Path, int] = {}

    for cls_id, max_copies in CLASS_TIER.items():
        for label_path in class_to_labels.get(cls_id, []):
            if label_path not in label_max_copies or label_max_copies[label_path] < max_copies:
                label_max_copies[label_path] = max_copies

    print(f"\n  Found {len(label_max_copies)} unique training images qualifying for oversampling.")

    if not label_max_copies:
        print("  Nothing to oversample. Exiting.")
        return 0

    total_copied = 0
    tier_counts: dict[int, int] = {}

    for label_src, max_copies in sorted(label_max_copies.items()):
        image_src = find_image_for_label(label_src, images_dir)
        if image_src is None:
            print(f"  [WARN] No image found for label: {label_src.name} -- skipping.")
            continue

        tier_counts[max_copies] = tier_counts.get(max_copies, 0) + 1

        for copy_idx in range(1, max_copies):  # extra copies: 1..(max_copies-1)
            new_stem = f"{label_src.stem}_os{copy_idx}"
            label_dst = labels_dir / f"{new_stem}.txt"
            image_dst = images_dir / f"{new_stem}{image_src.suffix}"

            # Idempotent: skip if already exists
            if label_dst.exists() and image_dst.exists():
                continue

            if not dry_run:
                shutil.copy2(label_src, label_dst)
                shutil.copy2(image_src, image_dst)

            total_copied += 1

    print(f"\n  Tier breakdown of qualifying images:")
    for copies in sorted(tier_counts):
        count = tier_counts[copies]
        if count:
            print(f"    {copies}x tier: {count} unique images -> +{count * (copies - 1)} copies each")

    return total_copied


def main() -> None:
    args = parse_args()
    rng = random.Random(SEED)

    dataset_root = Path(args.dataset_root)
    labels_dir = dataset_root / "train" / "labels"
    images_dir = dataset_root / "train" / "images"

    if not labels_dir.exists():
        print(f"[ERROR] Labels directory not found: {labels_dir}")
        print("  Run this script from the CIRCA project root (d:\\FYP\\CIRCA).")
        raise SystemExit(1)

    print("\n" + "=" * 65)
    print("  CIRCA -- Tiered Minority Class Oversampler  (v4 -- 7-class)")
    print("  Excluded (dominant): Class 3 (short), Class 5 (insufficient_solder)")
    print("  CRITICAL (5x): Class 0 (missing_hole), Class 4 (excess_solder), Class 6 (cold_solder_joint)")
    print(f"  Mode: {'DRY RUN -- no files written' if args.dry_run else 'LIVE -- files will be copied'}")
    print("=" * 65)

    # -- Before distribution (originals only -- _os* files excluded) --------------
    before_counts, class_to_labels = compute_class_distribution(labels_dir)
    print_distribution("BEFORE -- Original Class Distribution (excl. _os* copies)", before_counts)

    # -- Oversample ------------------------------------------------------------
    n_copied = oversample(
        labels_dir=labels_dir,
        images_dir=images_dir,
        class_to_labels=class_to_labels,
        dry_run=args.dry_run,
        rng=rng,
    )

    # -- After distribution ----------------------------------------------------
    if not args.dry_run:
        # Count ALL files (including _os* copies) to show true final state
        after_counts: Counter = Counter()
        for label_file in sorted(labels_dir.glob("*.txt")):
            for line in label_file.open():
                parts = line.strip().split()
                if parts:
                    after_counts[int(parts[0])] += 1
        print_distribution("AFTER -- Full Class Distribution (incl. _os* copies)", after_counts)
        n_train_images = len(list(images_dir.glob("*.*")))
        n_train_labels = len(list(labels_dir.glob("*.txt")))
        print(f"\n  Done. Copied {n_copied} image/label pairs.")
        print(f"  Training images: {n_train_images}")
        print(f"  Training labels: {n_train_labels}")
    else:
        print(f"\n  [DRY RUN] Would write approximately {n_copied} file pairs.")
        print("  Re-run without --dry-run to apply changes.")

    print()


if __name__ == "__main__":
    main()
