"""
cap_dominant_classes.py
=======================
CIRCA -- Phase 0 Data Preparation: Dominant-Class Capping.

Problem:
    After rebuilding unified_pcb_v3, images that contain ONLY dominant-class
    annotations (insufficient_solder=5, short=3) inflate the class ratio to
    ~10:1 vs the rarest minority class. These images provide zero useful signal
    for learning minority defects.

Solution:
    Randomly subsample dominant-only images down to a target cap, keeping all
    mixed images (those that co-occur with at least one minority class) intact.

Dominant classes (excluded from oversampling):
    3: short
    5: insufficient_solder

Target ratio after capping: ~5:1 (dominant_max : minority_min)
Default cap: 1000 dominant-only images kept (from ~3468 original)

Usage:
    python scripts/cap_dominant_classes.py
    python scripts/cap_dominant_classes.py --dataset-root datasets/unified_pcb_v3
    python scripts/cap_dominant_classes.py --cap 800 --dry-run
    python scripts/cap_dominant_classes.py --seed 42
"""

import argparse
import random
from collections import Counter
from pathlib import Path

# Classes that are dominant and should be capped
DOMINANT_CLASSES = {3, 5}  # short, insufficient_solder
CLASS_NAMES = [
    "missing_hole", "mouse_bite", "open_circuit", "short",
    "excess_solder", "insufficient_solder", "cold_solder_joint",
]

DEFAULT_CAP = 1000  # keep at most this many dominant-only images
DEFAULT_SEED = 42


def get_classes_in_label(label_file: Path) -> set:
    """Return the set of class IDs present in a YOLO label file."""
    classes = set()
    with open(label_file) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                classes.add(int(parts[0]))
    return classes


def count_annotations(label_files: list) -> Counter:
    counter = Counter()
    for f in label_files:
        with open(f) as fp:
            for line in fp:
                parts = line.strip().split()
                if parts:
                    counter[int(parts[0])] += 1
    return counter


def print_distribution(title: str, counter: Counter, total_images: int) -> None:
    total_ann = sum(counter.values())
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)
    for i, name in enumerate(CLASS_NAMES):
        c = counter.get(i, 0)
        bar = "#" * min(40, int(40 * c / max(total_ann, 1)))
        tag = " <- DOMINANT" if i in DOMINANT_CLASSES else ""
        print(f"  {i}  {name:<24} {c:6,} ({100*c/max(total_ann,1):4.1f}%)  {bar}{tag}")
    print("-" * 65)
    print(f"  Total annotation instances: {total_ann:,}")
    print(f"  Total training images:      {total_images:,}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CIRCA -- Cap dominant-only training images to reduce class imbalance."
    )
    parser.add_argument(
        "--dataset-root", type=str, default="datasets/unified_pcb_v3",
        help="Path to the dataset root (must contain train/labels/).",
    )
    parser.add_argument(
        "--cap", type=int, default=DEFAULT_CAP,
        help=f"Max number of dominant-only images to keep (default: {DEFAULT_CAP}).",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Random seed for reproducible sampling (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be removed without actually deleting files.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    labels_dir = dataset_root / "train" / "labels"
    images_dir = dataset_root / "train" / "images"

    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    print(f"\n{'='*65}")
    print(f"  CIRCA -- Dominant-Class Capper")
    print(f"  Dataset : {dataset_root}")
    print(f"  Cap     : {args.cap} dominant-only images kept")
    print(f"  Seed    : {args.seed}")
    print(f"  Mode    : {'DRY RUN -- no files deleted' if args.dry_run else 'LIVE -- files will be deleted'}")
    print(f"{'='*65}")

    # Categorize all label files
    dominant_only: list[Path] = []
    mixed: list[Path] = []
    all_labels: list[Path] = list(labels_dir.glob("*.txt"))

    for lbl in all_labels:
        classes = get_classes_in_label(lbl)
        if classes and classes.issubset(DOMINANT_CLASSES):
            dominant_only.append(lbl)
        else:
            mixed.append(lbl)

    print(f"\n  Total label files   : {len(all_labels):,}")
    print(f"  Mixed images        : {len(mixed):,}  (have >=1 minority annotation -- NEVER removed)")
    print(f"  Dominant-only images: {len(dominant_only):,}  (only dominant classes -- eligible for capping)")

    # BEFORE distribution
    before_counter = count_annotations(all_labels)
    print_distribution("BEFORE -- Full Class Distribution", before_counter, len(all_labels))

    # Decide which dominant-only images to remove
    n_to_keep = min(args.cap, len(dominant_only))
    n_to_remove = len(dominant_only) - n_to_keep

    if n_to_remove <= 0:
        print(f"\n  [SKIP] Dominant-only count ({len(dominant_only)}) is already <= cap ({args.cap}). Nothing to remove.")
        return

    random.seed(args.seed)
    random.shuffle(dominant_only)
    to_remove: list[Path] = dominant_only[n_to_keep:]   # discard the tail

    print(f"\n  Removing {n_to_remove:,} dominant-only images (keeping {n_to_keep:,} randomly selected).")

    removed_count = 0
    missing_image = 0
    for lbl in to_remove:
        # Find the matching image file (try common extensions)
        img = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            candidate = images_dir / (lbl.stem + ext)
            if candidate.exists():
                img = candidate
                break

        if args.dry_run:
            removed_count += 1
            continue

        lbl.unlink()
        removed_count += 1
        if img:
            img.unlink()
        else:
            missing_image += 1

    if args.dry_run:
        print(f"\n  [DRY RUN] Would remove {removed_count:,} label + image pairs.")
    else:
        print(f"\n  Removed {removed_count:,} label files.")
        if missing_image:
            print(f"  WARNING: {missing_image} labels had no matching image file.")

        # AFTER distribution
        remaining_labels = list(labels_dir.glob("*.txt"))
        after_counter = count_annotations(remaining_labels)
        print_distribution("AFTER -- Full Class Distribution", after_counter, len(remaining_labels))

        # Report ratio
        dom_max = max(after_counter.get(i, 0) for i in DOMINANT_CLASSES)
        minority_vals = [after_counter.get(i, 0) for i in range(7) if i not in DOMINANT_CLASSES and after_counter.get(i, 0) > 0]
        min_minority = min(minority_vals) if minority_vals else 1
        ratio = dom_max / min_minority
        print(f"\n  Imbalance ratio (dominant_max : minority_min): {ratio:.1f}:1")
        if ratio <= 10:
            print("  [OK] Ratio is within the <=10:1 target.")
        else:
            print("  [WARN] Ratio still exceeds 10:1 — consider a lower --cap value.")

    print()


if __name__ == "__main__":
    main()
