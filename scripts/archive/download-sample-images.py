import os
import time
import urllib.request

urls = {
    "canoscan_main_pcb.jpg": "https://upload.wikimedia.org/wikipedia/commons/e/e1/CanoScan_5600F_main_PCB_top_view.jpg",
    "dac1138kx_pcb.jpg": "https://upload.wikimedia.org/wikipedia/commons/9/94/DAC-1138KX_-_pcb_top_view.jpg",
    "olympia_display_pcb.jpg": "https://upload.wikimedia.org/wikipedia/commons/1/1a/Olympia_CD200_display_PCB_top_view.jpg",
    "seagate_harddrive_pcb.jpg": "https://upload.wikimedia.org/wikipedia/commons/7/75/Seagate_ST296N_-_PCB_top_view.jpg",
    "wd_harddrive_pcb.jpg": "https://upload.wikimedia.org/wikipedia/commons/4/4a/Western_Digital_WD93044A_PCB_top_view.jpg"
}

dest_dir = "datasets/benchmark_real_pcbs"
os.makedirs(dest_dir, exist_ok=True)

headers = {'User-Agent': 'Mozilla/5.0'}

print(f"Starting curation of {len(urls)} real full-board PCB images into '{dest_dir}'...")

for filename, url in urls.items():
    dest_path = os.path.join(dest_dir, filename)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"  {filename} already exists and is not empty, skipping.")
        continue
    try:
        print(f"  Downloading {filename} from {url}...")
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
            out_file.write(response.read())
        print(f"    Saved successfully ({os.path.getsize(dest_path)} bytes)")
    except Exception as e:
        print(f"    Failed to download {filename}: {e}")
    # Sleep to respect rate limits
    time.sleep(3)

print("PCB curation complete!")
