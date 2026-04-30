# CIRCA — Class Mapping Document

> This document maps the source categories from PKU, DsPCBSD+, PCB Defect, SolDef_AI, and PCB Solder Joint datasets to the unified CIRCA 15-class taxonomy based on IPC-A-600 and IPC-A-610 standards.

## CIRCA 15-Class Taxonomy

| Unified ID | Class Name | Description / IPC Reference | Source(s) |
| :--- | :--- | :--- | :--- |
| 0 | `missing_hole` | Hole missing where expected | Bare-board sources |
| 1 | `mouse_bite` | Edge breakout on trace | Bare-board sources |
| 2 | `open_circuit` | Complete break in conductor path | Bare-board sources |
| 3 | `short` | Unintended connection between conductors | Bare-board sources |
| 4 | `spur` | Unintended conductor protrusion | Bare-board sources |
| 5 | `spurious_copper` | Isolated copper fragments | Bare-board sources |
| 6 | `hole_breakout` | Hole breaking conductor edge | DsPCBSD+ |
| 7 | `conductor_scratch` | Surface damage to trace | DsPCBSD+ |
| 8 | `conductor_foreign_object` | Conductive debris | DsPCBSD+ |
| 9 | `base_material_foreign_object`| Non-conductive debris on laminate | DsPCBSD+ |
| 10 | `excess_solder` | Too much solder (IPC-A-610 §5) | SolDef_AI |
| 11 | `insufficient_solder` | Too little solder (IPC-A-610 §5) | SolDef_AI |
| 12 | `solder_spike` | Sharp protrusion of solder | SolDef_AI |
| 13 | `solder_bridge` | Solder bridging two pads | PCB Solder Joint |
| 14 | `cold_solder_joint` | Poor wetting / dull finish | PCB Solder Joint |

## Source Mapping Logic

- **PKU-Market-PCB-ver1:** Mapped fundamental 6 classes directly (0-5).
- **DsPCBSD+ (Lv et al., 2024):** Provided specialized industrial classes (6-9).
- **PCB Defect Dataset (Roboflow):** Mapped variant lighting/angle examples to the fundamental 6 classes.
- **SolDef_AI (Fontana et al., 2024):** Mapped `exc_solder`→10, `poor_solder`→11, `spike`→12.
- **PCB Solder Joint (Work, 2025):** Mapped `STICKSOLDER`→13, `COLDSOLDER`→14.

## Data Integrity Guardrails
- **Namespace isolation:** Filename prefixing (`d1_`, `d2_`, `d3_`) used to prevent collisions.
- **Normalisation:** All bounding boxes validated in [0.0, 1.0] range.
