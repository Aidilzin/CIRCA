# -*- coding: utf-8 -*-
"""
CIRCA Padlet Thumbnail Generator (Stand-alone Script)
Generates: docs/circa_thumbnail.png (1200x630) from docs/circa_thumbnail.html
"""
import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

def render_thumbnail():
    EDGE_PATHS = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"
    ]
    edge_exe = next((p for p in EDGE_PATHS if Path(p).exists()), None)
    if not edge_exe:
        print("[ERROR] Edge browser executable not found.")
        return

    thumb_html = DOCS / "circa_thumbnail.html"
    out_thumb = DOCS / "circa_thumbnail.png"

    print("[THUMBNAIL] Rendering Padlet thumbnail PNG via Edge...")
    subprocess.run([edge_exe, "--headless", "--disable-gpu", "--window-size=1200,630", f"--screenshot={out_thumb}", thumb_html.as_uri()], check=True)
    print(f"[THUMBNAIL] Saved PNG: {out_thumb}")

def main():
    render_thumbnail()

if __name__ == "__main__":
    main()
