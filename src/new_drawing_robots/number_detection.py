"""
Number Detection - Strategy: Dot -> Raw ROI -> Raw OCR (No Preprocess)
V3.1 - Hozzáadva egy 'processed' debug kép mentése, hogy lássuk,
mit lát az OCR a skálázás és zajszűrés után.
"""

import cv2
import numpy as np
import json
import logging
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image
import easyocr
import sys
import shutil
try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None
    logging.warning("PaddleOCR not found. To use it, run: pip install paddleocr")


def setup_ocr(config):
    """Initialize OCR engines"""
    readers = {
        'easyocr': None,
        'paddleocr': None
    }
    
    try:
        gpu_enabled = config['number_detection']['easyocr_gpu']
        import torch
        if gpu_enabled and not torch.cuda.is_available():
            gpu_enabled = False
            logging.warning("CUDA not available, using CPU for all OCR engines")
    except Exception:
        gpu_enabled = False
        logging.warning("Torch not found, using CPU for all OCR engines")

    if config['number_detection']['use_easyocr']:
        try:
            readers['easyocr'] = easyocr.Reader(['en'], gpu=gpu_enabled)
            logging.info(f"✓ EasyOCR initialized (GPU: {gpu_enabled})")
        except Exception as e:
            logging.warning(f"EasyOCR initialization failed: {e}")
            
    if config['number_detection'].get('use_paddleocr', False):
        if PaddleOCR is None:
            logging.warning("PaddleOCR is enabled in config, but package not found.")
        else:
            try:
                readers['paddleocr'] = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=gpu_enabled, show_log=False)
                logging.info(f"✓ PaddleOCR initialized (GPU: {gpu_enabled})")
            except Exception as e:
                logging.warning(f"PaddleOCR initialization failed: {e}")
    
    return readers


def process_roi_for_ocr(raw_gray_roi, debug_dir, dot_id):
    """
    Prepares a RAW grayscale tile (ROI) for OCR.
    "Preprocess Nélkül" - Csak felskálázzuk és zajszűrjük.
    """
    if raw_gray_roi.size == 0:
        return None

    # 1. Felskálázás OCR-hez
    h, w = raw_gray_roi.shape
    if h == 0 or w == 0:
        return None
        
    scale_factor = 4
    scaled = cv2.resize(raw_gray_roi, (w * scale_factor, h * scale_factor), 
                       interpolation=cv2.INTER_CUBIC)
    
    # 2. Egy enyhe zajszűrés sokat segít az OCR-nek
    denoised = cv2.fastNlMeansDenoising(scaled, None, 5, 7, 21)
    
    # --- ÚJ DEBUG LÉPÉS: Mentsük el, amit az OCR lát ---
    if debug_dir:
        try:
            cv2.imwrite(str(debug_dir / f"_DEBUG_dot_{dot_id}_FOR_OCR.jpg"), denoised)
        except Exception: pass
    # --- DEBUG VÉGE ---

    return { 'image': denoised, 'original': raw_gray_roi, 'scale_factor': scale_factor }


def detect_with_easyocr(processed_data, reader):
    """Detect with EasyOCR, return number and LOCAL bbox"""
    if reader is None or processed_data is None:
        return []
    
    try:
        results = reader.readtext(
            processed_data['image'],
            allowlist='0123456789',
            detail=1,
            batch_size=4
        )
        
        detections = []
        for bbox, text, conf in results:
            if text.isdigit():
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                sf = processed_data['scale_factor']
                local_x = int(np.mean(x_coords) / sf)
                local_y = int(np.mean(y_coords) / sf)
                
                detections.append({
                    'number': int(text), 
                    'confidence': int(conf*100), 
                    'method': 'easyocr',
                    'local_x': local_x,
                    'local_y': local_y
                })
        return detections
    except Exception:
        return []


def detect_with_paddleocr(processed_data, reader):
    """Detect with PaddleOCR, return number and LOCAL bbox"""
    if reader is None or processed_data is None:
        return []
        
    try:
        results = reader.ocr(processed_data['image'], cls=True)
        detections = []
        if results and results[0]:
            for res in results[0]:
                bbox = res[0]
                text, conf = res[1]
                if text.isdigit():
                    x_coords = [p[0] for p in bbox]
                    y_coords = [p[1] for p in bbox]
                    sf = processed_data['scale_factor']
                    local_x = int(np.mean(x_coords) / sf)
                    local_y = int(np.mean(y_coords) / sf)

                    detections.append({
                        'number': int(text), 
                        'confidence': int(conf*100), 
                        'method': 'paddleocr',
                        'local_x': local_x,
                        'local_y': local_y
                    })
        return detections
    except Exception:
        return []


def load_dots(global_dots_json):
    """Load dot coordinates"""
    try:
        with open(global_dots_json) as f:
            data = json.load(f)
        
        dots = data.get("circles", [])
        logging.info(f"Loaded {len(dots)} dots")
        return dots
    except Exception as e:
        logging.error(f"Failed to load dots: {e}")
        return []


def visualize_final_pairs(original_image, dots, final_detections, output_path):
    """Creates the final visualization"""
    vis = original_image.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    
    for dot in dots:
        x, y = dot["global_coordinates"]["x"], dot["global_coordinates"]["y"]
        cv2.circle(vis, (x, y), 3, (255, 0, 0), -1) 
    
    for det in final_detections:
        dot_id = det['dot_id']
        num_x = det['global_coordinates']['x']
        num_y = det['global_coordinates']['y']
        
        dot_coord = None
        for dot in dots:
            if dot['id'] == dot_id:
                dot_coord = (dot["global_coordinates"]["x"], dot["global_coordinates"]["y"])
                break
        
        color = (0, 165, 255) 
        if len(det['methods']) > 1:
            color = (0, 255, 0) 
        
        cv2.circle(vis, (num_x, num_y), 5, (0, 0, 255), -1) 
        label = f"{det['number']}"
        cv2.putText(vis, label, (num_x + 8, num_y - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(vis, label, (num_x + 8, num_y - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                   
        if dot_coord:
            cv2.line(vis, dot_coord, (num_x, num_y), (150, 150, 150), 1)
    
    cv2.imwrite(str(output_path), vis)


def run_segment_based_detection(config, picture_name, expected_range=None):
    """
    Main - (Dot -> Raw ROI -> Raw OCR -> Find Closest) Strategy
    """
    logging.info("\n" + "="*60)
    logging.info("NUMBER DETECTION (Strategy: Dot -> Raw ROI -> No Preprocess)")
    logging.info("="*60 + "\n")
    
    base_path = config['_base_path']
    config_dir = base_path / config['paths']['config_dir']
    debug_dir = base_path / config['paths']['number_debug_dir']
    
    if debug_dir.exists():
        shutil.rmtree(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load ORIGINAL image
    picture_path = base_path / config['paths']['pictures_dir'] / picture_name
    if not picture_path.exists():
        logging.error(f"❌ A képfájl nem található: {picture_path}")
        return
    
    original_image = cv2.imread(str(picture_path))
    gray_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    h, w = gray_original.shape
    logging.info(f"📷 Eredeti kép betöltve: {w}x{h} ({picture_name})")

    # 2. Load Dots
    dots_json_path = config_dir / config['filenames']['global_dots']
    if not dots_json_path.exists():
        logging.error(f"❌ A pontfájl nem található: {dots_json_path}")
        return
    dots = load_dots(dots_json_path)
    if not dots:
        logging.error("❌ Nincsenek betöltött pontok!")
        return

    # 3. DEBUG: Pontok mentése az eredeti képre
    logging.info("DEBUG: Az összes pont mentése az eredeti képre...")
    try:
        viz_dots = original_image.copy()
        for d in dots:
            gx, gy = d['global_coordinates']['x'], d['global_coordinates']['y']
            cv2.circle(viz_dots, (gx, gy), 5, (0, 0, 255), 2)
            cv2.putText(viz_dots, str(d['id']), (gx + 5, gy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.imwrite(str(debug_dir / "_DEBUG_original_with_all_dots.jpg"), viz_dots)
        logging.info(f"✓ Debug kép mentve: _DEBUG_original_with_all_dots.jpg")
    except Exception as e:
        logging.warning(f"Nem sikerült a debug kép mentése: {e}")
    # --- DEBUG VÉGE ---

    # 4. Setup OCR
    readers = setup_ocr(config)
    
    # 5. Iterate over DOTS
    logging.info("="*60)
    logging.info("Párosítás pontonként (Dot -> Raw ROI -> No Preprocess)")
    logging.info("="*60)
    
    final_detections = []
    roi_radius = config['number_detection']['number_search_roi_radius'] 
    
    for idx, dot in enumerate(dots):
        dot_id = dot['id']
        dot_x = dot['global_coordinates']['x']
        dot_y = dot['global_coordinates']['y']
        
        # 4a. Cut search-ROI from ORIGINAL GRAY image
        x1 = max(0, dot_x - roi_radius)
        y1 = max(0, dot_y - roi_radius)
        x2 = min(w, dot_x + roi_radius)
        y2 = min(h, dot_y + roi_radius)
        
        raw_roi = gray_original[y1:y2, x1:x2]
        
        dot_local_x = dot_x - x1
        dot_local_y = dot_y - y1
        
        # 4b. DEBUG: Nyers ROI mentése
        try:
            debug_roi_img = cv2.cvtColor(raw_roi, cv2.COLOR_GRAY2BGR)
            cv2.line(debug_roi_img, (dot_local_x - 5, dot_local_y), (dot_local_x + 5, dot_local_y), (0, 0, 255), 1)
            cv2.line(debug_roi_img, (dot_local_x, dot_local_y - 5), (dot_local_x, dot_local_y + 5), (0, 0, 255), 1)
            cv2.imwrite(str(debug_dir / f"_DEBUG_dot_{dot_id}_RAW_roi.jpg"), debug_roi_img)
        except Exception as e:
            pass 
        # --- DEBUG VÉGE ---

        # 4c. Run "No Preprocess" (csak skálázás/zajszűrés) és OCR
        processed_data = process_roi_for_ocr(raw_roi, debug_dir, dot_id)
        if not processed_data:
            continue
            
        easy_dets = detect_with_easyocr(processed_data, readers['easyocr'])
        paddle_dets = detect_with_paddleocr(processed_data, readers['paddleocr'])
        
        all_roi_detections = easy_dets + paddle_dets
        
        if not all_roi_detections:
            continue
            
        # 4d. Find closest number
        best_det = None
        min_dist = float('inf')
        
        for det in all_roi_detections:
            dist = np.sqrt((det['local_x'] - dot_local_x)**2 + (det['local_y'] - dot_local_y)**2)
            if dist < min_dist:
                min_dist = dist
                best_det = det
        
        if best_det:
            if expected_range:
                min_n, max_n = expected_range
                if not (min_n <= best_det['number'] <= max_n):
                    continue 

            votes = [d for d in all_roi_detections if d['number'] == best_det['number']]
            methods = list(set(d['method'] for d in votes))

            final_detections.append({
                'dot_id': dot_id,
                'number': best_det['number'],
                'global_coordinates': {
                    'x': x1 + best_det['local_x'], 
                    'y': y1 + best_det['local_y']
                },
                'confidence': best_det['confidence'],
                'methods': methods
            })
            
        # Progress bar
        progress = ((idx + 1) / len(dots)) * 100
        bar_length = 40
        filled = int(bar_length * (idx + 1) / len(dots))
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"\r[{bar}] {progress:.1f}% | Pont {idx+1}/{len(dots)} (ID: {dot_id})", end='', flush=True)

    print("\n" + "="*60)
    logging.info(f"✓ Párosítás kész! Találatok: {len(final_detections)}/{len(dots)}")
    
    # 5. Save results
    logging.info("💾 Eredmények mentése...")
    output_json = config_dir / config['filenames']['global_numbers']
    with open(output_json, 'w') as f:
        json.dump({
            "detection_method": "Dot -> Raw ROI -> Raw OCR -> Closest",
            "total_numbers": len(final_detections),
            "numbers": final_detections
        }, f, indent=2)
    
    logging.info(f"✓ Mentve ide: {output_json}\n")
    
    # 6. Visualize
    logging.info("🎨 Vizualizáció készítése...")
    viz_path = base_path / config['filenames']['main_with_numbers']
    visualize_final_pairs(original_image, dots, final_detections, viz_path)
    logging.info(f"✓ Vizualizáció mentve ide: {viz_path}\n")