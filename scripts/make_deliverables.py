# -*- coding: utf-8 -*-
"""
CIRCA Master Exhibition Deliverables Orchestrator
Calls isolated generators: generate_poster, generate_deck, generate_thumbnail
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import generate_poster
import generate_deck
import generate_thumbnail

def main():
    print("=" * 70)
    print(" CIRCA EXHIBITION DELIVERABLES ORCHESTRATOR")
    print("=" * 70)
    
    print("\n[1/3] Generating Poster Deliverables...")
    generate_poster.main()
    
    print("\n[2/3] Generating Pitch Deck Deliverables...")
    generate_deck.main()
    
    print("\n[3/3] Generating Padlet Thumbnail Deliverables...")
    generate_thumbnail.main()
    
    print("\n" + "=" * 70)
    print(" ALL DELIVERABLES GENERATED SUCCESSFULLY IN docs/")
    print("=" * 70)

if __name__ == "__main__":
    main()
