# CIRCA Defect Detection Application — Running and Packaging Handbook

This document provides complete instructions to execute and package the CIRCA PySide6 desktop application on a Windows host machine.

---

## 💻 1. Running the Desktop App Locally

To run the application locally on your laptop using the calibrated parameters and the production YOLOv12-Nano FP16 engine, execute the following commands in your shell:

### Step A: Activate the Virtual Environment
```powershell
.venv\Scripts\activate
```

### Step B: Set Debug Environment Flags (Optional)
To log execution details and print console alerts:
```powershell
$env:CIRCA_DEBUG="1"
```

### Step C: Run the Entry Point
```powershell
python main.py
```
*Note: This will launch the main window, automatically load the calibrated detection parameters (`circa_thresholds.yaml`), perform hardware diagnostics on your CPU, and display the file-inspection workspace.*

---

## 📦 2. Packaging the App into an Independent `.exe`

To compile the application into a single standalone Windows executable (`circa.exe`) containing the PySide6 UI, OpenCV, and the OpenVINO model weights:

### Step A: Verify PyInstaller Installation
Ensure `pyinstaller` is installed in your virtual environment:
```powershell
.venv\Scripts\pip install pyinstaller
```

### Step B: Execute the Packaging Script
Use the compiler command to bundle the asset directories (`models/` and `config/`) and the OpenVINO runtime dependencies directly into the bundle's `sys._MEIPASS` runtime space:
```powershell
.venv\Scripts\pyinstaller --noconfirm --onedir --windowed --add-data "models;models" --add-data "config;config" --add-data ".venv\Lib\site-packages\openvino\libs;openvino\libs" --name "circa" main.py
```

### Step C: Locate the Bundled Executable
Once the compiler completes:
- The executable will be generated at `dist/circa/circa.exe`.
- The folder `dist/circa/` is self-contained and ready to be zipped and shared.

---

## 🔎 3. System Calibrations
The default confidence thresholds loaded by the executable are located in [circa_thresholds.yaml](file:///d:/FYP/CIRCA/config/circa_thresholds.yaml):
- `missing_hole`: 0.10 *(Lowered to trigger warning panel overlays)*
- `insufficient_solder`: 0.50
- `cold_solder_joint`: 0.10
- `excess_solder`: 0.60
- `short`: 0.40
- `open_circuit`: 0.50
- `mouse_bite`: 0.60
