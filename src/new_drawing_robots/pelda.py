"""
Pre-processing Parameter Visualizer

Ez a szkript segít megtalálni a legjobb előfeldolgozási 
paramétereket a számfelismeréshez.

Végigpróbál többféle 'blockSize' és 'C' értéket,
és elmenti az eredményképeket az 'Output_pictures/preprocess_test/' mappába.
"""

import cv2
import numpy as np
import json
import logging
from pathlib import Path
import shutil
import itertools
from orchestrator import load_config, setup_directories, select_image

def run_preprocessing(gray_image, blur_k, bSize, C_val):
    """Lefuttatja a teljes kép előfeldolgozási lépést."""
    try:
        if blur_k % 2 == 0:
            blur_k += 1
        
        background = cv2.GaussianBlur(gray_image, (blur_k, blur_k), 0)
        normalized_image = cv2.divide(gray_image, background, scale=255.0)

        clean_image = cv2.adaptiveThreshold(
            normalized_image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            bSize, C_val
        )
        return clean_image
    except Exception as e:
        logging.warning(f"Hiba b{bSize}_C{C_val} esetén: {e}")
        return None

def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # --- Paraméterek, amiket tesztelni akarunk ---
    BLUR_KERNELS = [51] # Maradhat fix, ha bevált
    BLOCK_SIZES = [7, 11, 15, 19] # Páratlan számok
    C_VALUES = [7, 10, 13, 15]
    # ---------------------------------------------

    try:
        config = load_config()
        setup_directories(config) # Létrehozza a mappákat
        
        # Válassz képet
        image_path = select_image(config)
        
        # Hozzuk létre a teszt mappát
        base_path = config['_base_path']
        test_dir = base_path / "Output_pictures" / "preprocess_test"
        if test_dir.exists():
            shutil.rmtree(test_dir)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info("="*60)
        logging.info(f"Kép betöltése: {image_path.name}")
        logging.info(f"Kimeneti mappa: {test_dir}")
        logging.info("="*60)

        original_image = cv2.imread(str(image_path))
        gray_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        
        combinations = list(itertools.product(BLUR_KERNELS, BLOCK_SIZES, C_VALUES))
        
        for idx, (blur_k, bSize, c_val) in enumerate(combinations):
            
            logging.info(f"[{idx+1}/{len(combinations)}] Tesztelés: blur{blur_k}_b{bSize}_C{c_val}")
            
            processed_image = run_preprocessing(gray_original, blur_k, bSize, c_val)
            
            if processed_image is not None:
                filename = f"blur{blur_k}_b{bSize}_C{c_val}.jpg"
                cv2.imwrite(str(test_dir / filename), processed_image)
                
        logging.info("="*60)
        logging.info(f"✓ Teszt befejezve! Eredmények itt: {test_dir}")
        logging.info("Válaszd ki a legjobb képet, és frissítsd a config.json-t!")
        logging.info("="*60)

    except Exception as e:
        logging.error(f"Hiba történt: {e}", exc_info=True)

if __name__ == "__main__":
    main()