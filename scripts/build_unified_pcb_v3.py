"""
CIRCA Dataset Builder — unified_pcb_v3 (7-class)
=================================================
Merges 6 public PCB defect sources into a single clean 7-class YOLO dataset.

USAGE:
    python scripts/build_unified_pcb_v3.py --dry-run   # preview counts only
    python scripts/build_unified_pcb_v3.py             # execute build

REQUIRES: pip install imagehash Pillow scikit-learn tqdm

7-CLASS TAXONOMY:
    0: missing_hole         (IPC-A-600 §3.4)
    1: mouse_bite           (IPC-A-600 §3.3)
    2: open_circuit         (IPC-A-600 §3.2)
    3: short                (IPC-A-600 §3.2)
    4: excess_solder        (IPC-A-610H §5)
    5: insufficient_solder  (IPC-A-610H §5)
    6: cold_solder_joint    (IPC-A-610H §5)
"""

import argparse
import shutil
import hashlib
from pathlib import Path
from collections import defaultdict

import imagehash
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIGURATION — edit SOURCE_ROOTS to point at your downloaded dataset folders
# ---------------------------------------------------------------------------

DATASETS_DIR = Path("datasets")
OUTPUT_DIR = DATASETS_DIR / "unified_pcb_v3"
ARCHIVE_DIR = DATASETS_DIR / "unified_pcb_v3_dropped"

# Each entry: (source_dir, remap_dict, source_type)
# source_type: "id"  → label file uses integer IDs (standard YOLO)
#              "name" → label file uses class names (needs name→id map)
# remap_dict maps SOURCE_ID → UNIFIED_ID (integer keys for "id" type)
# Drop: any class not in remap_dict → label line deleted; image archived if all labels dropped.

SOURCES = [
    # A: PKU-Market-PCB (Roboflow YOLOv12 export)
    # Source classes: 0=missing_hole, 1=mouse_bite, 2=open_circuit, 3=short, 4=spur, 5=spurious_copper
    {
        "name": "PKU",
        "root": DATASETS_DIR / "PKU-Market-PCB (Bare-Board, 6 classes \u2192 keep 4)",
        "type": "id",
        "remap": {0: 0, 1: 1, 2: 2, 3: 3},  # drop 4 (spur) and 5 (spurious_copper)
    },

    # B: DsPCBSD+ 2024 (Lv et al., 2024) — flat YOLO layout under Data_YOLO/
    # Classes (from COCO annotations audit): 0=SH(short), 1=SP(spur), 2=SC(spurious_copper),
    #   3=OP(open_circuit), 4=MB(mouse_bite), 5=HB(hole_breakout), 6=CS(conductor_scratch),
    #   7=CFO(conductor_foreign_object), 8=BMFO(base_material_foreign_object)
    # Keep: SH→short(3), OP→open_circuit(2), MB→mouse_bite(1). Drop all others.
    {
        "name": "DsPCBSD+",
        "root": DATASETS_DIR / "DsPCBSD+ 2024 (Bare-Board, large-scale)" / "Data_YOLO",
        "type": "id",
        "remap": {
            0: 3,  # SH (Short) -> short
            3: 2,  # OP (Open circuit) -> open_circuit
            4: 1,  # MB (Mouse Bite) -> mouse_bite
            # 1=SP(spur), 2=SC(spurious_copper), 5=HB, 6=CS, 7=CFO, 8=BMFO -> all DROPPED
        },
    },

    # C: SolDef_AI (Fontana et al., 2024) — Roboflow YOLOv12 export
    # data.yaml classes: ['exc_solder', 'good', 'no_good', 'poor_solder', 'spike']
    # YOLO label files use integer IDs (0-indexed alphabetical Roboflow order).
    # Keep: exc_solder(0)->excess_solder(4), poor_solder(3)->cold_solder_joint(6)
    # Drop: good(1), no_good(2), spike(4)
    {
        "name": "SolDef_AI",
        "root": DATASETS_DIR / "SolDef_AI",
        "type": "id",
        "remap": {0: 4, 3: 6},  # exc_solder->excess_solder(4), poor_solder->cold_solder_joint(6)
    },

    # D: PCB Solder Defect Detection V2 (emmts/pcb-solder-defect-detection-v2-s89jo)
    # Roboflow YOLOv12 export. data.yaml classes: ['Cold_solder', 'Excessive_solder', 'Insufficient_solder']
    # YOLO label files use integer IDs -> use type="id" with numeric keys.
    {
        "name": "SolderV2",
        "root": DATASETS_DIR / "PCB Solder Defect Detection V2 (Assembly Solder, 3 classes)",
        "type": "id",
        "remap": {
            0: 6,  # Cold_solder -> cold_solder_joint
            1: 4,  # Excessive_solder -> excess_solder
            2: 5,  # Insufficient_solder -> insufficient_solder
        },
    },

    # E: Excessive Solder kydra (emmts/excessive-solder-kydra)
    # Roboflow YOLOv12 export. data.yaml classes: ['Cold Solder', 'Excessive_solder', 'Insufficient_solder']
    {
        "name": "kydra",
        "root": DATASETS_DIR / "Excessive Solder \u2014 kydra (Assembly Solder, 3 classes)",
        "type": "id",
        "remap": {
            0: 6,  # Cold Solder -> cold_solder_joint
            1: 4,  # Excessive_solder -> excess_solder
            2: 5,  # Insufficient_solder -> insufficient_solder
        },
    },

    # F: Hue Solder Dataset (emmts/hue-dbgbs-reqtv)
    # Roboflow YOLOv12 export. data.yaml classes: ['Insufficient Solder', 'Missing Component', 'Shorted']
    # ID 1 (Missing Component) -> DROPPED (component-level defect, not solder defect)
    {
        "name": "Hue",
        "root": DATASETS_DIR / "Hue Solder Dataset (Assembly + Short overlap)",
        "type": "id",
        "remap": {
            0: 5,  # Insufficient Solder -> insufficient_solder
            # 1: Missing Component -> intentionally dropped
            2: 3,  # Shorted -> short
        },
    },
]

UNIFIED_NAMES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "excess_solder",
    "insufficient_solder",
    "cold_solder_joint",
]

SPLIT_RATIOS = (0.70, 0.15, 0.15)
SEED = 42
DHASH_THRESHOLD = 6  # Hamming distance for near-duplicate detection

# ---------------------------------------------------------------------------

def read_yaml_classes(yaml_path: Path) -> dict[str, int]:
    """Parse a Roboflow data.yaml and return {class_name: id} dict."""
    import re
    text = yaml_path.read_text(encoding="utf-8")
    names = re.findall(r"^\s*-\s*(.+)$", text, re.MULTILINE)
    return {n.strip(): i for i, n in enumerate(names)}


def load_label_file_id(label_path: Path, remap: dict) -> list[str]:
    """Remap a YOLO label file using integer class IDs."""
    out = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        src_id = int(parts[0])
        if src_id in remap:
            out.append(f"{remap[src_id]} " + " ".join(parts[1:]))
    return out


def load_label_file_name(label_path: Path, remap: dict, yaml_classes: dict) -> list[str]:
    """Remap a YOLO label file using class names resolved via yaml_classes."""
    out = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        src_id = int(parts[0])
        # Resolve id → name using yaml_classes (inverted)
        id_to_name = {v: k for k, v in yaml_classes.items()}
        src_name = id_to_name.get(src_id, "")
        if src_name in remap:
            out.append(f"{remap[src_name]} " + " ".join(parts[1:]))
    return out


def image_dhash(img_path: Path) -> imagehash.ImageHash:
    try:
        return imagehash.dhash(Image.open(img_path).convert("RGB"))
    except Exception:
        return None


def build_dataset(dry_run: bool = False):
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Building unified_pcb_v3 (7-class)\n" + "=" * 60)

    all_entries = []  # list of (image_path, remapped_labels: list[str], source_name)
    class_counts = defaultdict(int)

    for src_cfg in SOURCES:
        src_name = src_cfg["name"]
        root = src_cfg["root"]
        skip_if_missing = src_cfg.get("skip_if_missing", False)

        if not root.exists():
            if skip_if_missing:
                print(f"  [SKIP] {src_name}: folder not found — {root}")
                continue
            else:
                print(f"  [WARN] {src_name}: folder not found — {root}")
                continue

        remap = src_cfg["remap"]
        src_type = src_cfg["type"]

        # Resolve yaml_classes for name-based sources
        yaml_classes = {}
        if src_type == "name":
            yaml_path = next(root.rglob("data.yaml"), None)
            if yaml_path:
                yaml_classes = read_yaml_classes(yaml_path)
            else:
                print(f"  [WARN] {src_name}: no data.yaml found for name resolution")

        # Find all images
        # Standard Roboflow layout: root/train/images, root/valid/images, root/test/images
        # DsPCBSD+ layout: root/images/train, root/images/val  (inverted structure)
        image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        img_dirs = [
            root / "train" / "images",
            root / "valid" / "images",
            root / "test" / "images",
            root / "images",          # flat layout
            root / "images" / "train",  # DsPCBSD+ inverted layout
            root / "images" / "val",    # DsPCBSD+ inverted layout
        ]

        found = 0
        dropped = 0
        for img_dir in img_dirs:
            if not img_dir.exists():
                continue
            lbl_dir = Path(str(img_dir).replace("images", "labels"))

            for img_file in img_dir.iterdir():
                if img_file.suffix.lower() not in image_exts:
                    continue
                lbl_file = lbl_dir / (img_file.stem + ".txt")
                if not lbl_file.exists():
                    continue

                if src_type == "id":
                    labels = load_label_file_id(lbl_file, remap)
                else:
                    labels = load_label_file_name(lbl_file, remap, yaml_classes)

                if labels:
                    for lbl in labels:
                        cid = int(lbl.split()[0])
                        class_counts[cid] += 1
                    all_entries.append((img_file, labels, src_name))
                    found += 1
                else:
                    dropped += 1

        print(f"  [{src_name}] kept: {found}, dropped (no valid labels): {dropped}")

    print(f"\nTotal before dedup: {len(all_entries)} images")

    # --- Deduplication ---
    print("\nRunning pHash deduplication (dHash <= 6)...")
    hash_map: dict[str, Path] = {}
    dedup_entries = []
    duplicates = 0
    for img_path, labels, src_name in tqdm(all_entries, desc="Hashing"):
        h = image_dhash(img_path)
        if h is None:
            continue
        is_dup = False
        for existing_hash_str in list(hash_map.keys()):
            existing_hash = imagehash.hex_to_hash(existing_hash_str)
            if abs(h - existing_hash) <= DHASH_THRESHOLD:
                is_dup = True
                duplicates += 1
                break
        if not is_dup:
            hash_map[str(h)] = img_path
            dedup_entries.append((img_path, labels, src_name))

    print(f"After dedup: {len(dedup_entries)} images ({duplicates} duplicates removed)")

    if dry_run:
        print("\n[DRY RUN] Class distribution (train instances approx):")
        for cid, cnt in sorted(class_counts.items()):
            print(f"  {cid}: {UNIFIED_NAMES[cid]:25s} {cnt:>6} instances")
        return

    # --- Split ---
    print(f"\nSplitting {len(dedup_entries)} images (70/15/15, seed={SEED})...")
    indices = list(range(len(dedup_entries)))
    # Stratify on dominant class per image
    strat_labels = []
    for _, labels, _ in dedup_entries:
        ids = [int(l.split()[0]) for l in labels]
        from collections import Counter
        dominant = Counter(ids).most_common(1)[0][0]
        strat_labels.append(dominant)

    train_idx, valtest_idx = train_test_split(
        indices, test_size=(SPLIT_RATIOS[1] + SPLIT_RATIOS[2]),
        random_state=SEED, stratify=strat_labels)
    valtest_strat = [strat_labels[i] for i in valtest_idx]
    val_idx, test_idx = train_test_split(
        valtest_idx, test_size=0.5, random_state=SEED, stratify=valtest_strat)

    splits = {"train": train_idx, "valid": val_idx, "test": test_idx}

    # --- Write output ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, split_indices in splits.items():
        (OUTPUT_DIR / split_name / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split_name / "labels").mkdir(parents=True, exist_ok=True)

        for idx in tqdm(split_indices, desc=f"Writing {split_name}"):
            img_path, labels, src_name = dedup_entries[idx]
            dest_img = OUTPUT_DIR / split_name / "images" / img_path.name
            dest_lbl = OUTPUT_DIR / split_name / "labels" / (img_path.stem + ".txt")
            # Handle filename collisions
            counter = 0
            while dest_img.exists():
                counter += 1
                dest_img = OUTPUT_DIR / split_name / "images" / f"{img_path.stem}_{counter}{img_path.suffix}"
                dest_lbl = OUTPUT_DIR / split_name / "labels" / f"{img_path.stem}_{counter}.txt"
            shutil.copy2(img_path, dest_img)
            dest_lbl.write_text("\n".join(labels))

    # --- Write data.yaml ---
    yaml_content = f"""path: {OUTPUT_DIR.resolve().as_posix()}
train: train/images
val:   valid/images
test:  test/images

nc: {len(UNIFIED_NAMES)}
names:
{chr(10).join(f'  {i}: {n}' for i, n in enumerate(UNIFIED_NAMES))}
"""
    (OUTPUT_DIR / "data.yaml").write_text(yaml_content)

    # --- Final report ---
    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print(f"Output: {OUTPUT_DIR}")
    print(f"  Train: {len(splits['train'])} images")
    print(f"  Val:   {len(splits['valid'])} images")
    print(f"  Test:  {len(splits['test'])} images")
    print("\nClass distribution (all splits combined):")
    for cid, cnt in sorted(class_counts.items()):
        if cid < len(UNIFIED_NAMES):
            print(f"  {cid}: {UNIFIED_NAMES[cid]:25s} {cnt:>6} instances")
    print("\ndata.yaml written. Ready for train_engine.py.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build unified_pcb_v3 dataset")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count and remap without writing any files")
    args = parser.parse_args()
    build_dataset(dry_run=args.dry_run)
