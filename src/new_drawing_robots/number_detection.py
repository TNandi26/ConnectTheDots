import cv2
import numpy as np
import json
import logging
import easyocr
import shutil
from paddleocr import PaddleOCR

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
            logging.info(f"EasyOCR initialized (GPU: {gpu_enabled})")
        except Exception as e:
            logging.warning(f"EasyOCR init failed: {e}")
            
    if config['number_detection'].get('use_paddleocr', False):
        if PaddleOCR is None:
            logging.warning("PaddleOCR enabled but not found.")
        else:
            try:
                readers['paddleocr'] = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=gpu_enabled, show_log=False)
                logging.info(f"PaddleOCR initialized (GPU: {gpu_enabled})")
            except Exception as e:
                logging.warning(f"PaddleOCR init failed: {e}")
    return readers


def process_roi_for_ocr(raw_gray_roi, debug_dir=None, dot_id_suffix=""):
    """
    Prepares a RAW grayscale tile (ROI) for OCR.
    Upscales and denoises ONLY (no sharpening).
    """
    if raw_gray_roi.size == 0:
        return None
    h, w = raw_gray_roi.shape
    if h == 0 or w == 0:
        return None
        
    scale_factor = 4
    scaled = cv2.resize(raw_gray_roi, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_CUBIC)
    denoised = cv2.fastNlMeansDenoising(scaled, None, 3, 7, 21) 
    
    if debug_dir and dot_id_suffix:
        try:
            cv2.imwrite(str(debug_dir / f"_DEBUG_dot_{dot_id_suffix}_FOR_OCR_denoised.jpg"), denoised)
        except Exception:
            pass
            
    return { 'image': denoised, 'original': raw_gray_roi, 'scale_factor': scale_factor }

def detect_with_easyocr(processed_data, reader):
    """Detect with EasyOCR, return number, center, and LOCAL BBOX [x1, y1, x2, y2]"""
    if reader is None or processed_data is None:
        return []
    try:
        results = reader.readtext(processed_data['image'], allowlist='0123456789', detail=1, batch_size=4)
        detections = []
        for bbox_scaled, text, conf in results:
            if text.isdigit():
                x_coords_s = [p[0] for p in bbox_scaled]
                y_coords_s = [p[1] for p in bbox_scaled]
                sf = processed_data['scale_factor']
                x1 = int(min(x_coords_s) / sf)
                y1 = int(min(y_coords_s) / sf)
                x2 = int(max(x_coords_s) / sf)
                y2 = int(max(y_coords_s) / sf)
                local_x = (x1 + x2) // 2
                local_y = (y1 + y2) // 2
                detections.append({
                    'number': int(text),
                    'confidence': int(conf*100),
                    'method': 'easyocr',
                    'local_x': local_x,
                    'local_y': local_y,
                    'bbox': [x1, y1, x2, y2]
                })
        return detections
    except Exception:
        return []

def detect_with_paddleocr(processed_data, reader):
    """Detect with PaddleOCR, return number, center, and LOCAL BBOX [x1, y1, x2, y2]"""
    if reader is None or processed_data is None:
        return []
    try:
        results = reader.ocr(processed_data['image'], cls=True)
        detections = []
        if results and results[0]:
            for res in results[0]:
                bbox_scaled = res[0]
                text, conf = res[1]
                if text.isdigit():
                    x_coords_s = [p[0] for p in bbox_scaled]
                    y_coords_s = [p[1] for p in bbox_scaled]
                    sf = processed_data['scale_factor']
                    x1 = int(min(x_coords_s) / sf)
                    y1 = int(min(y_coords_s) / sf)
                    x2 = int(max(x_coords_s) / sf)
                    y2 = int(max(y_coords_s) / sf)
                    local_x = (x1 + x2) // 2
                    local_y = (y1 + y2) // 2
                    detections.append({
                        'number': int(text),
                        'confidence': int(conf*100),
                        'method': 'paddleocr',
                        'local_x': local_x,
                        'local_y': local_y,
                        'bbox': [x1, y1, x2, y2]
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
    """Creates the main_image_with_numbers.jpg visualization"""
    vis = original_image.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    for dot in dots:
        x = dot["global_coordinates"]["x"]
        y = dot["global_coordinates"]["y"]
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

        color = (0, 165, 255) # Orange
        if len(det.get('methods', [])) > 1:
            color = (0, 255, 0) # Green

        cv2.circle(vis, (num_x, num_y), 5, (0, 0, 255), -1)
        label = f"{det['number']}"
        cv2.putText(vis, label, (num_x + 8, num_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(vis, label, (num_x + 8, num_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if dot_coord:
            cv2.line(vis, dot_coord, (num_x, num_y), (150, 150, 150), 1)

    cv2.imwrite(str(output_path), vis)


def run_segment_based_detection(config, picture_name, expected_range=None):
    logging.info("Starting number detection")

    base_path = config['_base_path']
    config_dir = base_path / config['paths']['config_dir']
    debug_dir = base_path / config['paths']['number_debug_dir']

    if debug_dir.exists():
        shutil.rmtree(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    picture_path = base_path / config['paths']['pictures_dir'] / picture_name
    if not picture_path.exists():
        logging.error(f"❌ Image file not found: {picture_path}")
        return

    original_image = cv2.imread(str(picture_path))
    gray_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    h, w = gray_original.shape
    logging.info(f"📷 Original image: {w}x{h} ({picture_name})")

    dots_json_path = config_dir / config['filenames']['global_dots']
    if not dots_json_path.exists():
        logging.error(f"❌ Dot file not found: {dots_json_path}")
        return
    dots = load_dots(dots_json_path)
    if not dots:
        logging.error("❌ No dots found!")
        return

    # --- GLOBAL DOT ERASE (RADIUS + 2) ---
    logging.info("Starting global dot erase (with local median, radius+2)...")
    gray_dots_erased = gray_original.copy()
    erased_count = 0
    neighborhood_radius = 10
    global_fill_color = 200 # Default light gray

    try:
        global_median = int(np.median(gray_original[gray_original > 50]))
        if 100 < global_median < 250:
            global_fill_color = global_median
    except:
        pass

    for dot in dots:
        try:
            gx = dot['global_coordinates']['x']
            gy = dot['global_coordinates']['y']
            radius = dot.get('radius', 3)
            erase_radius = radius + 2 # Radius + 2 as requested

            nx1 = max(0, gx - neighborhood_radius)
            ny1 = max(0, gy - neighborhood_radius)
            nx2 = min(w, gx + neighborhood_radius)
            ny2 = min(h, gy + neighborhood_radius)
            neighborhood = gray_original[ny1:ny2, nx1:nx2]

            if neighborhood.size == 0:
                fill_color = global_fill_color
            else:
                local_gx = gx - nx1
                local_gy = gy - ny1
                mask = np.ones(neighborhood.shape, dtype=np.uint8) * 255
                cv2.circle(mask, (local_gx, local_gy), erase_radius, 0, -1)
                local_median = np.median(neighborhood[mask == 255])
                fill_color = global_fill_color if np.isnan(local_median) else int(local_median)

            cv2.circle(gray_dots_erased, (gx, gy), erase_radius, fill_color, -1)
            erased_count += 1
        except Exception as e:
            logging.warning(f"Dot {dot.get('id','?')}: Error during local color erase: {e}")
    logging.info(f"✓ {erased_count} dots erased (with local median).")
    cv2.imwrite(str(debug_dir / "_DEBUG_gray_dots_erased.jpg"), gray_dots_erased)
    # --- GLOBAL DOT ERASE END ---

    readers = setup_ocr(config)
    logging.info("="*60)
    logging.info("Pairing dot by dot (Iterative Number Erase)")
    logging.info("="*60)

    final_detections = []
    search_roi_radius = config['number_detection']['number_search_roi_radius']
    refined_padding = config['number_detection']['refined_ocr_padding']

    # Important: Sort dots by ID so the processing order is deterministic
    sorted_dots = sorted(dots, key=lambda d: d['id'])

    for idx, dot in enumerate(sorted_dots):
        dot_id = dot['id']
        dot_x = dot['global_coordinates']['x']
        dot_y = dot['global_coordinates']['y']

        # We cut from the 'gray_dots_erased' image, which is UPDATED iteratively!
        sx1 = max(0, dot_x - search_roi_radius)
        sy1 = max(0, dot_y - search_roi_radius)
        sx2 = min(w, dot_x + search_roi_radius)
        sy2 = min(h, dot_y + search_roi_radius)
        search_roi_erased = gray_dots_erased[sy1:sy2, sx1:sx2]
        dot_local_x = dot_x - sx1
        dot_local_y = dot_y - sy1

        try: # Save Debug ROI
             debug_sroi = cv2.cvtColor(search_roi_erased, cv2.COLOR_GRAY2BGR)
             cv2.line(debug_sroi, (dot_local_x - 5, dot_local_y), (dot_local_x + 5, dot_local_y), (0, 0, 255), 1)
             cv2.line(debug_sroi, (dot_local_x, dot_local_y - 5), (dot_local_x, dot_local_y + 5), (0, 0, 255), 1)
             cv2.imwrite(str(debug_dir / f"_DEBUG_dot_{dot_id}_SEARCH_roi_erased.jpg"), debug_sroi)
        except:
            pass

        processed_search_roi = process_roi_for_ocr(search_roi_erased)
        if not processed_search_roi:
            continue

        easy_dets_cand = detect_with_easyocr(processed_search_roi, readers['easyocr'])
        paddle_dets_cand = detect_with_paddleocr(processed_search_roi, readers['paddleocr'])
        all_candidates = easy_dets_cand + paddle_dets_cand
        if not all_candidates:
            continue

        best_candidate = None
        min_dist = float('inf')
        for cand in all_candidates:
            dist = np.sqrt((cand['local_x'] - dot_local_x)**2 + (cand['local_y'] - dot_local_y)**2)
            if dist < search_roi_radius and dist < min_dist:
                min_dist = dist
                best_candidate = cand
        if not best_candidate:
            continue

        # Refined ROI coordinates
        gr_x1, gr_y1, gr_x2, gr_y2 = (0,0,0,0) # Initialization
        try:
            lc_x1, lc_y1, lc_x2, lc_y2 = best_candidate['bbox']
            gr_x1 = max(0, sx1 + lc_x1 - refined_padding)
            gr_y1 = max(0, sy1 + lc_y1 - refined_padding)
            gr_x2 = min(w, sx1 + lc_x2 + refined_padding)
            gr_y2 = min(h, sy1 + lc_y2 + refined_padding)
            refined_roi_erased = gray_dots_erased[gr_y1:gr_y2, gr_x1:gr_x2]
            cv2.imwrite(str(debug_dir / f"_DEBUG_dot_{dot_id}_REFINED_roi_erased.jpg"), refined_roi_erased)
        except Exception as e:
            logging.warning(f"Dot {dot_id}: Refined ROI error. {e}")
            continue

        processed_refined_roi = process_roi_for_ocr(refined_roi_erased, debug_dir, f"{dot_id}_refined")
        if not processed_refined_roi:
            continue

        easy_dets_final = detect_with_easyocr(processed_refined_roi, readers['easyocr'])
        paddle_dets_final = detect_with_paddleocr(processed_refined_roi, readers['paddleocr'])
        all_final_detections = easy_dets_final + paddle_dets_final
        if not all_final_detections:
            continue

        # Select the highest confidence result from the refined ROI
        final_result = max(all_final_detections, key=lambda d: d['confidence'])

        if expected_range:
            min_n, max_n = expected_range
            if not (min_n <= final_result['number'] <= max_n):
                continue

        final_global_x = (gr_x1 + gr_x2) // 2
        final_global_y = (gr_y1 + gr_y2) // 2
        methods = list(set(d['method'] for d in all_final_detections if d['number'] == final_result['number']))

        final_detections.append({
            'dot_id': dot_id,
            'number': final_result['number'],
            'global_coordinates': { 'x': final_global_x, 'y': final_global_y },
            'confidence': final_result['confidence'],
            'methods': methods,
            'distance': min_dist # Save the distance for the matchmaker
        })

        # --- MODIFICATION HERE: ERASE THE FOUND NUMBER ---
        try:
            # Use the 'gr_x1...' coordinates (global coordinates of the refined ROI)
            cv2.rectangle(gray_dots_erased, (gr_x1, gr_y1), (gr_x2, gr_y2), global_fill_color, -1)
            logging.debug(f"Dot {dot_id}: Number '{final_result['number']}' erased from further searching.")
            
            # Save a debug image of the erase (showing the search ROI)
            debug_img_after_erase = gray_dots_erased[sy1:sy2, sx1:sx2].copy()
            cv2.imwrite(str(debug_dir / f"_DEBUG_dot_{dot_id}_AFTER_ERASING_NUMBER_{final_result['number']}.jpg"), debug_img_after_erase)
            
        except Exception as e:
            logging.warning(f"Dot {dot_id}: Failed to erase number: {e}")
        # --- MODIFICATION END ---

        progress = ((idx + 1) / len(sorted_dots)) * 100
        bar_length = 40
        filled = int(bar_length * (idx + 1) / len(sorted_dots))
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"\r[{bar}] {progress:.1f}% | Dot {idx+1}/{len(sorted_dots)} (ID: {dot_id})", end='', flush=True)

    print("\n" + "="*60)
    logging.info(f"✓ Pairing complete! Hits: {len(final_detections)}/{len(dots)}")

    logging.info("💾 Saving results...")
    output_json = config_dir / config['filenames']['global_numbers']
    with open(output_json, 'w') as f:
        json.dump({ "detection_method": "Global Erase + Iterative Number Erase (V9)", "total_numbers": len(final_detections), "numbers": final_detections }, f, indent=2)
    logging.info(f"✓ Saved to: {output_json}\n")

    logging.info("🎨 Creating visualization...")
    viz_path = base_path / config['filenames']['main_with_numbers']
    visualize_final_pairs(original_image, dots, final_detections, viz_path) 
    logging.info(f"✓ Visualization saved to: {viz_path}\n")