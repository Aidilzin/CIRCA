#!/usr/bin/env python
"""
scripts/download_exhibition_backgrounds.py
------------------------------------------
Exhibition Prep: Downloader for High-Res Motherboard/PCB Backgrounds.

Downloads CC0/CC-BY licensed top-down PCB and motherboard images from Wikimedia
Commons to serve as diverse backgrounds for the copy-paste augmentation pipeline.
"""

import os
import urllib.request
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BackgroundDownloader")

# Curated list of top-down motherboard/PCBA images from Wikimedia Commons
WIKIMEDIA_URLS = {
    "acer_aspire_one_top_down.jpg": "https://upload.wikimedia.org/wikipedia/commons/b/b3/Acer_Aspire_One_ZG5_motherboard_DA0ZG5MB8F0_-_top_view.jpg",
    "apple_iic_top_down.jpg": "https://upload.wikimedia.org/wikipedia/commons/4/40/Apple-IIc-Motherboard-Flat-Top.jpg",
    "atari_lynx_top_down.jpg": "https://upload.wikimedia.org/wikipedia/commons/c/c2/Atari-Lynx-Motherboard-Top.jpg",
    "amiga_cd32_top_down.jpg": "https://upload.wikimedia.org/wikipedia/commons/4/41/Amiga-CD32-Motherboard-Top.jpg",
    "nintendo_n64_top_down.jpg": "https://upload.wikimedia.org/wikipedia/commons/3/36/Nintendo-N64-Motherboard-Top.jpg",
    "sega_master_system_top_down.jpg": "https://upload.wikimedia.org/wikipedia/commons/f/f6/Sega-Master-System-Mk1-Motherboard-Flat-Top.jpg",
    "nintendo_nes_top_down.jpg": "https://upload.wikimedia.org/wikipedia/commons/5/5a/Nintendo-NES-Mk1-Motherboard-Top.jpg",
    "epia_px10000g_top_down.jpg": "https://upload.wikimedia.org/wikipedia/commons/4/47/Top_EPIA_PX10000G_Motherboard_new.jpg",
    "nintendo_snes_top_down.jpg": "https://upload.wikimedia.org/wikipedia/commons/3/34/Nintendo-Super-NES-Mk1-Motherboard-Top.jpg",
    "snes_1chip_top_down.jpg": "https://upload.wikimedia.org/wikipedia/commons/a/aa/Super-Nintendo-1Chip-Motherboard-Top-Flat.jpg"
}

def main():
    dest_dir = "datasets/copypaste_backgrounds"
    os.makedirs(dest_dir, exist_ok=True)
    
    logger.info("Starting background downloads into '%s'...", dest_dir)
    
    downloaded_count = 0
    # Descriptive User-Agent following Wikimedia's policy to prevent HTTP 429
    headers = {
        'User-Agent': 'CircaPCBInspectionSystem/1.0 (contact: aidilzin@circa-project.org; student research project)'
    }
    
    for filename, url in WIKIMEDIA_URLS.items():
        dest_path = os.path.join(dest_dir, filename)
        if os.path.exists(dest_path):
            logger.info("File already exists, skipping: %s", filename)
            downloaded_count += 1
            continue
            
        logger.info("Downloading %s...", filename)
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as response, open(dest_path, 'wb') as out_file:
                out_file.write(response.read())
            logger.info("Successfully downloaded %s", filename)
            downloaded_count += 1
        except Exception as exc:
            logger.error("Failed to download %s: %s", filename, exc)
            
    logger.info("Completed! Successfully acquired %d/%d backgrounds.", downloaded_count, len(WIKIMEDIA_URLS))

if __name__ == "__main__":
    main()
