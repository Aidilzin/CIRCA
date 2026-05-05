# CIRCA — Unified 12-Class IPC Taxonomy (v3)

> **Dataset path:** `D:/FYP/CIRCA/datasets/unified_pcb_v2`
> **nc: 12** | Sources: 8 datasets | IPC-A-600 (bare-board) + IPC-A-610H (assembly)

## Class Taxonomy

| Unified ID | Class Name | IPC Reference | Source Datasets | Raw Class Name(s) Remapped From |
|:---|:---|:---|:---|:---|
| 0 | `missing_hole` | IPC-A-600 §3.4 | PKU/JR, Bare PCB | `missing_hole` |
| 1 | `mouse_bite` | IPC-A-600 §3.3 | PKU/JR, Bare PCB | `mouse_bite` |
| 2 | `open_circuit` | IPC-A-600 §3.2 | PKU/JR, Bare PCB | `open_circuit` |
| 3 | `short` | IPC-A-600 §3.2 | PKU/JR, Bare PCB, Hue | `short`, `Short`, `Shorted` |
| 4 | `spur` | IPC-A-600 §3.3 | PKU/JR, Bare PCB | `spur` |
| 5 | `spurious_copper` | IPC-A-600 §3.3 | PKU/JR, Bare PCB | `spurious_copper`, `falsecopper` |
| 6 | `excess_solder` | IPC-A-610H §5 | SolDef_AI, kydra, f8m5i, v2-s89jo | `exc_solder` (α-idx 0), `Excessive_solder`, `excess_solder` |
| 7 | `insufficient_solder` | IPC-A-610H §5 | SolDef_AI, kydra, Hue, v2-s89jo | `poor_solder` (α-idx 3), `Insufficient_solder`, `INSUFFICIENT SOLDER`, `Insufficient Solder`, `Missing_solder` |
| 8 | `solder_spike` | IPC-A-610H §5 | SolDef_AI | `spike` (α-idx 4) |
| 9 | `cold_solder_joint` | IPC-A-610H §5 | kydra, v2-s89jo | `Cold Solder`, `Cold_solder` |
| 10 | `scratch` | IPC-A-600 §3 | Bare PCB defects | `scratch` |
| 11 | `pinhole` | IPC-A-600 §3 | Bare PCB defects | `pinhole` |

**SolDef_AI class mapping (alphabetical Roboflow export order):**
Roboflow exports SolDef_AI with alphabetically sorted class indices:
- Raw ID 0 → `exc_solder` → Unified ID **6**
- Raw ID 1 → `good` → **DROPPED** (routed to `negatives_reserve/`)
- Raw ID 2 → `no_good` → **DROPPED** (umbrella label; routed to `negatives_reserve/`)
- Raw ID 3 → `poor_solder` → Unified ID **7**
- Raw ID 4 → `spike` → Unified ID **8**

Python mapping dict: `{0: 6, 3: 7, 4: 8}` (raw IDs 1 and 2 omitted → negatives_reserve).

## YAML data.yaml

```yaml
path: D:/FYP/CIRCA/datasets/unified_pcb_v2
train: train/images
val:   valid/images
test:  test/images
nc: 12
names:
  0: missing_hole
  1: mouse_bite
  2: open_circuit
  3: short
  4: spur
  5: spurious_copper
  6: excess_solder
  7: insufficient_solder
  8: solder_spike
  9: cold_solder_joint
  10: scratch
  11: pinhole
```

## Dropped / Excluded Classes

**Classes dropped during remapping:**
- `good` / `GOOD` (SolDef_AI): YOLO learns non-defective regions implicitly as background.
- `no_good` (SolDef_AI): umbrella label overlapping with IDs 6, 7, 8 — creates label conflicts.
- `missing_component` (Hue): IPC-A-610 component-level defect excluded per Chapter 1 §1.5.
- `solder_bridge`: No clean board-level annotated public dataset found across all 8 sources. Documented as future work in Chapter 5.

**Dropped datasets (quality audit):**
- `sthr7` (`pcb-deffect-detection-solder-sthr7`): Dropped — zero `solder_spike` instances despite documentation, and high annotation noise.
- `lsb7m` (`pcb-deffect-detection-solder-lsb7m`): Dropped — single-class (`Excessive_solder` only), fully superseded by kydra, f8m5i, and v2-s89jo.

**RAHUL dataset deduplication:** perceptual-hash (pHash, Hamming ≤ 6) cross-check against PKU/JR required before final merge.

## Dataset Sources

| # | Dataset | Slug / Source | Images | Raw Classes Used | Unified IDs | License |
|:---|:---|:---|:---|:---|:---|:---|
| 1 | **SolDef_AI** (Fontana et al., 2024) | Kaggle: `mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection` | ~1,150 | `exc_solder` (α-idx 0), `poor_solder` (α-idx 3), `spike` (α-idx 4) | 6, 7, 8 | CC BY 4.0 |
| 2 | **PKU-Market-PCB via JR** | `jr-mqqnk/pcb-defects-detection-anddl` | ~1,500 | 6 standard bare-board classes | 0–5 | Public Domain |
| 3 | **RAHUL PCB Defects** | `rahul-jhj03/pcb-defects-dataset` | Unknown | 6 standard bare-board classes | 0–5 | CC BY 4.0 |
| 4 | **Bare PCB Defects** | `bare-pcb-defects/obj-detection-pcb-defects-yolov8` | ~9,666 | `missing_hole`, `mouse_bite`, `open_circuit`, `short`, `spur`, `falsecopper`, `scratch`, `pinhole` | 0–5, 10, 11 | CC BY 4.0 |
| 5 | **Hue** (PCB Defect Detection, 2025b) | `emmts/hue-dbgbs-reqtv` | 3,232 | `Insufficient Solder`, `Shorted` | 7, 3 | CC BY 4.0 |
| 6 | **solder-f8m5i-xnbzp** (PCB Defect Detection, 2025c) | `emmts/solder-f8m5i-xnbzp` | Unknown | `Excessive_solder` | 6 | CC BY 4.0 |
| 7 | **Excessive Solder / kydra** (PCB Defect Detection, 2025e) | `emmts/excessive-solder-kydra` | 1,162 | `Cold Solder`, `Excessive_solder`, `Insufficient_solder` | 9, 6, 7 | CC BY 4.0 |
| 8 | **PCB Solder Defect Detection V2** (PCB Defect Detection, 2025f) | `emmts/pcb-solder-defect-detection-v2-s89jo` | 6,116 | `Cold_solder`, `Excessive_solder`, `Insufficient_solder` | 9, 6, 7 | CC BY 4.0 |
