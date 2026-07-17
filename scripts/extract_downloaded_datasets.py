import os
import zipfile
import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DatasetExtractor")

def main():
    root_dir = Path("d:/FYP/CIRCA")
    datasets_dir = root_dir / "datasets"
    dest_dir = datasets_dir / "copypaste_backgrounds"
    os.makedirs(dest_dir, exist_ok=True)
    
    # 1. Extract Zenodo DSLR ZIP
    dslr_zip_path = datasets_dir / "cvl_pcb_dslr_1.zip"
    if dslr_zip_path.exists():
        logger.info("Extracting images from Zenodo DSLR zip: %s", dslr_zip_path.name)
        extracted_dslr_count = 0
        with zipfile.ZipFile(dslr_zip_path, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                if file_info.filename.lower().endswith(('.jpg', '.jpeg', '.png')) and not file_info.is_dir():
                    # Flatten the path
                    basename = os.path.basename(file_info.filename)
                    if basename:
                        target_path = dest_dir / f"dslr_{basename}"
                        with zip_ref.open(file_info) as source, open(target_path, 'wb') as target:
                            shutil.copyfileobj(source, target)
                        extracted_dslr_count += 1
        logger.info("Extracted %d DSLR images.", extracted_dslr_count)
    else:
        logger.warning("Zenodo DSLR zip not found at: %s", dslr_zip_path)

    # 2. Extract YOLOv8 (MiracleFactory) ZIP
    yolov8_zip_path = datasets_dir / "yolov8.zip"
    if yolov8_zip_path.exists():
        logger.info("Extracting images from YOLOv8 (MiracleFactory) zip: %s", yolov8_zip_path.name)
        extracted_mf_count = 0
        with zipfile.ZipFile(yolov8_zip_path, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                if file_info.filename.lower().endswith(('.jpg', '.jpeg', '.png')) and not file_info.is_dir():
                    # Flatten the path
                    basename = os.path.basename(file_info.filename)
                    if basename:
                        target_path = dest_dir / f"mf_{basename}"
                        with zip_ref.open(file_info) as source, open(target_path, 'wb') as target:
                            shutil.copyfileobj(source, target)
                        extracted_mf_count += 1
        logger.info("Extracted %d MiracleFactory images.", extracted_mf_count)
    else:
        logger.warning("YOLOv8 zip not found at: %s", yolov8_zip_path)

    # 3. Clean up the zip files to save workspace disk space
    for zip_file in [dslr_zip_path, yolov8_zip_path]:
        if zip_file.exists():
            logger.info("Removing source zip to save disk space: %s", zip_file.name)
            zip_file.unlink()

    # Get total backgrounds
    total_bg = len(os.listdir(dest_dir))
    logger.info("Done! Total backgrounds available in '%s': %d", dest_dir, total_bg)

if __name__ == "__main__":
    main()
