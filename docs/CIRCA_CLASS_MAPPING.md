# CIRCA — Unified 7-Class IPC Taxonomy (unified_pcb_v3)

> **Dataset path:** `D:/FYP/CIRCA/datasets/unified_pcb_v3`
> **nc: 7** | Sources: 6 datasets (+ DsPCBSD+ once downloaded) | IPC-A-600 (bare-board) + IPC-A-610H (assembly)

---

## Class Taxonomy

| Unified ID | Class Name | IPC Reference | Source Datasets | Raw Class Name(s) Remapped From |
|:---|:---|:---|:---|:---|
| 0 | `missing_hole` | IPC-A-600 §3.4 | PKU/JR, DsPCBSD+ | `missing_hole` |
| 1 | `mouse_bite` | IPC-A-600 §3.3 | PKU/JR, DsPCBSD+ | `mouse_bite`, `rat_bite` |
| 2 | `open_circuit` | IPC-A-600 §3.2 | PKU/JR, DsPCBSD+ | `open_circuit` |
| 3 | `short` | IPC-A-600 §3.2 | PKU/JR, DsPCBSD+, Hue | `short`, `Short`, `Shorted`, `short_circuit` |
| 4 | `excess_solder` | IPC-A-610H §5 | SolDef_AI, kydra, v2-s89jo | `exc_solder` (α-idx 0), `Excessive_solder` |
| 5 | `insufficient_solder` | IPC-A-610H §5 | SolDef_AI, kydra, Hue, v2-s89jo | `poor_solder` (α-idx 3), `Insufficient_solder`, `Insufficient Solder`, `Missing_solder` |
| 6 | `cold_solder_joint` | IPC-A-610H §5 | kydra, v2-s89jo | `Cold Solder`, `Cold_solder` |

---

## SolDef_AI Class Mapping (Alphabetical Roboflow Export)

Roboflow exports SolDef_AI with alphabetically sorted class indices. When using the "YOLOv8 Object Detection" export format, polygon annotations are automatically converted to axis-aligned bounding boxes — no custom conversion script is required.

| Raw ID | Raw Class Name | Action | Unified ID |
|:---:|:---|:---|:---:|
| 0 | `exc_solder` | Remap → | **4** |
| 1 | `good` | DROP → `negatives_reserve/` | — |
| 2 | `no_good` | DROP (umbrella label) → `negatives_reserve/` | — |
| 3 | `poor_solder` | Remap → | **5** |
| 4 | `spike` | DROP (insufficient data; see Limitations) | — |

Python mapping dict: `{0: 4, 3: 5}` — raw IDs 1, 2, 4 omitted.

---

## Dropped / Excluded Classes (Limitations)

| Class | Reason for Exclusion | Thesis Reference |
|:---|:---|:---|
| `spur` | Visually near-identical to `short`; creates inter-class confusion during training | §1.5, §3.4.1.5 |
| `spurious_copper` | Visually near-identical to `spur`; insufficient independent source coverage | §1.5, §3.4.1.5 |
| `solder_spike` | Only 91 training instances from a single source; below minimum threshold | §1.5, §3.4.1.5 |
| `scratch` | Only 52 training instances; below minimum threshold | §1.5, §3.4.1.5 |
| `pinhole` | Only 56 training instances; below minimum threshold | §1.5, §3.4.1.5 |
| `solder_bridge` | No board-level annotated public dataset found; documented as future work | §5 |
| Component-level defects | Missing component, misalignment, tombstoning, lifted lead, solder ball — no public datasets with sufficient coverage | §1.5, §3.4.1.5 |

---

## data.yaml (unified_pcb_v3)

```yaml
path: D:/FYP/CIRCA/datasets/unified_pcb_v3
train: train/images
val:   valid/images
test:  test/images

nc: 7
names:
  0: missing_hole
  1: mouse_bite
  2: open_circuit
  3: short
  4: excess_solder
  5: insufficient_solder
  6: cold_solder_joint
```

---

## Dataset Sources

| # | Dataset | Source / Slug | Images | Classes Used | Unified IDs | License |
|:---|:---|:---|:---|:---|:---|:---|
| 1 | **PKU-Market-PCB via JR** | `jr-mqqnk/pcb-defects-detection-anddl` (Roboflow) | ~1,500 | `missing_hole`, `mouse_bite`, `open_circuit`, `short` (drop `spur`, `spurious_copper`) | 0, 1, 2, 3 | Public Domain |
| 2 | **DsPCBSD+** (Lv et al., 2024) | Figshare: `10.6084/m9.figshare.24970329` | 10,259 | Mapped equivalents of `short`, `open_circuit`, `mouse_bite`, `missing_hole` | 0, 1, 2, 3 | CC BY 4.0 |
| 3 | **SolDef_AI** (Fontana et al., 2024) | Kaggle: `mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection` | ~1,150 | `exc_solder` → 4, `poor_solder` → 5 (drop `spike`, `good`, `no_good`) | 4, 5 | CC BY 4.0 |
| 4 | **PCB Solder Defect Detection V2** | `emmts/pcb-solder-defect-detection-v2-s89jo` (Roboflow) | 6,116 | `Cold_solder`, `Excessive_solder`, `Insufficient_solder` | 4, 5, 6 | CC BY 4.0 |
| 5 | **Excessive Solder / kydra** | `emmts/excessive-solder-kydra` (Roboflow) | 1,162 | `Cold Solder`, `Excessive_solder`, `Insufficient_solder` | 4, 5, 6 | CC BY 4.0 |
| 6 | **Hue** | `emmts/hue-dbgbs-reqtv` (Roboflow) | 3,232 | `Insufficient Solder`, `Shorted` | 3, 5 | CC BY 4.0 |

---

## Build Script

Run `scripts/build_unified_pcb_v3.py` to merge, remap, deduplicate, and split all sources into `unified_pcb_v3/`.

```powershell
# Preview counts without writing files
python scripts/build_unified_pcb_v3.py --dry-run

# Execute full build
python scripts/build_unified_pcb_v3.py
```
