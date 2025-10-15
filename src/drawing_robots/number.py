"""
Robust Detection - Target 80%+ accuracy
- Multi-stage preprocessing
- Contrast enhancement
- Multiple search strategies
- Confidence-based retry
"""

import cv2
import numpy as np
import json
import logging
from pathlib import Path
from collections import Counter
import pytesseract
from PIL import Image
import easyocr


def setup_ocr(config):
    """Initialize OCR"""
    pytesseract.pytesseract.tesseract_cmd = config['number_detection']['tesseract_path']
    
    reader = None
    if config['number_detection']['use_easyocr']:
        try:
            reader = easyocr.Reader(['en'], gpu=config['number_detection']['easyocr_gpu'])
            logging.info("EasyOCR initialized")
        except Exception as e:
            logging.warning(f"EasyOCR failed: {e}")
    
    return reader


def load_dot_coordinates(global_dots_json):
    """Load dots"""
    try:
        with open(global_dots_json) as f:
            data = json.load(f)
        
        dots = [
            {"id": c["id"], "x": c["global_coordinates"]["x"], 
             "y": c["global_coordinates"]["y"], "radius": c.get("radius", 3)}
            for c in data.get("circles", [])
            if "global_coordinates" in c and "id" in c
        ]
        
        logging.info(f"Loaded {len(dots)} dots")
        return dots
    except Exception as e:
        logging.error(f"Failed to load dots: {e}")
        return []


def extract_dot_region(image, dot_x, dot_y, search_radius):
    """Extract region"""
    h, w = image.shape[:2]
    x1, y1 = max(0, dot_x - search_radius), max(0, dot_y - search_radius)
    x2, y2 = min(w, dot_x + search_radius), min(h, dot_y + search_radius)
    return image[y1:y2, x1:x2], (x1, y1)


def enhance_contrast(region):
    """
    Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    Helps with low-contrast numbers
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(region)
    return enhanced


def calculate_adaptive_thresholds(region):
    """Calculate smart thresholds"""
    # Otsu
    _, otsu_binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_thresh = _
    
    mean_val = np.mean(region)
    
    # Generate 4 thresholds around Otsu
    if mean_val > 200:
        thresholds = [
            max(100, otsu_thresh - 30),
            otsu_thresh,
            min(200, otsu_thresh + 30)
        ]
    elif mean_val > 150:
        thresholds = [
            max(70, otsu_thresh - 30),
            otsu_thresh,
            min(180, otsu_thresh + 30),
            min(200, otsu_thresh + 50)
        ]
    else:
        thresholds = [
            max(50, otsu_thresh - 20),
            otsu_thresh,
            min(150, otsu_thresh + 30)
        ]
    
    return sorted(set([int(t) for t in thresholds]))


def preprocess_multi_strategy(region, dot_x_local, dot_y_local, dot_radius):
    """
    Multiple preprocessing strategies
    Strategy 1: Normal
    Strategy 2: Contrast enhanced
    Strategy 3: Morphological cleaned
    """
    preprocessed = []
    
    # Erase dot first (all strategies)
    region_no_dot = region.copy()
    cv2.circle(region_no_dot, (dot_x_local, dot_y_local), dot_radius, 255, -1)
    
    # Strategy 1: Normal thresholding
    thresholds = calculate_adaptive_thresholds(region_no_dot)
    
    for thresh in thresholds:
        _, binary = cv2.threshold(region_no_dot, thresh, 255, cv2.THRESH_BINARY)
        
        # Light morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Upscale 4x
        scaled = cv2.resize(binary, (binary.shape[1] * 4, binary.shape[0] * 4), 
                          interpolation=cv2.INTER_CUBIC)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(scaled, None, 5, 7, 21)
        
        preprocessed.append({
            'image': denoised,
            'strategy': 'normal',
            'threshold': thresh,
            'scale_factor': 4
        })
    
    # Strategy 2: Contrast enhanced (for weak numbers)
    enhanced = enhance_contrast(region_no_dot)
    thresholds_enh = calculate_adaptive_thresholds(enhanced)
    
    for thresh in thresholds_enh[:2]:  # Only use 2 best thresholds
        _, binary = cv2.threshold(enhanced, thresh, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        scaled = cv2.resize(binary, (binary.shape[1] * 4, binary.shape[0] * 4), 
                          interpolation=cv2.INTER_CUBIC)
        
        denoised = cv2.fastNlMeansDenoising(scaled, None, 5, 7, 21)
        
        preprocessed.append({
            'image': denoised,
            'strategy': 'enhanced',
            'threshold': thresh,
            'scale_factor': 4
        })
    
    # Strategy 3: Morphological repair (for broken characters)
    # Use median threshold only
    median_thresh = thresholds[len(thresholds)//2]
    _, binary = cv2.threshold(region_no_dot, median_thresh, 255, cv2.THRESH_BINARY)
    
    # More aggressive morphology to connect broken parts
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    scaled = cv2.resize(binary, (binary.shape[1] * 4, binary.shape[0] * 4), 
                      interpolation=cv2.INTER_CUBIC)
    
    denoised = cv2.fastNlMeansDenoising(scaled, None, 5, 7, 21)
    
    preprocessed.append({
        'image': denoised,
        'strategy': 'morphology',
        'threshold': median_thresh,
        'scale_factor': 4
    })
    
    return preprocessed


def detect_tesseract(region_data, min_conf):
    """Tesseract with best PSM modes"""
    detections = []
    
    # PSM 8 (word), 11 (sparse), 13 (raw line)
    psm_modes = [8, 11, 13]
    
    for psm in psm_modes:
        config_str = f"--psm {psm} --oem 3 -c tessedit_char_whitelist=0123456789"
        
        try:
            pil_img = Image.fromarray(region_data['image'])
            data = pytesseract.image_to_data(pil_img, config=config_str, 
                                            output_type=pytesseract.Output.DICT)
            
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                conf = int(data['conf'][i]) if data['conf'][i] != -1 else 0
                
                if text and text.isdigit() and conf > min_conf:
                    sf = region_data['scale_factor']
                    detections.append({
                        'number': int(text),
                        'confidence': conf,
                        'bbox': (data['left'][i]//sf, data['top'][i]//sf, 
                                data['width'][i]//sf, data['height'][i]//sf),
                        'method': 'tesseract',
                        'psm': psm,
                        'strategy': region_data['strategy']
                    })
        except Exception:
            pass
    
    return detections


def detect_easyocr(region_data, reader):
    """EasyOCR with boost for multi-digit"""
    if reader is None:
        return []
    
    detections = []
    
    try:
        results = reader.readtext(region_data['image'], allowlist='0123456789', detail=1)
        
        for bbox, text, conf in results:
            if text.isdigit():
                sf = region_data['scale_factor']
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                
                x = int(min(x_coords)) // sf
                y = int(min(y_coords)) // sf
                w = int(max(x_coords) - min(x_coords)) // sf
                h = int(max(y_coords) - min(y_coords)) // sf
                
                # Boost multi-digit
                adjusted_conf = int(conf * 100)
                if len(text) > 1:
                    adjusted_conf = min(100, int(adjusted_conf * 1.3))
                
                detections.append({
                    'number': int(text),
                    'confidence': adjusted_conf,
                    'bbox': (x, y, w, h),
                    'method': 'easyocr',
                    'strategy': region_data['strategy']
                })
    except Exception:
        pass
    
    return detections


def detect_number_for_dot(image_gray, dot, config, reader):
    """Robust detection for one dot"""
    search_radius = config['number_detection']['search_radius']
    min_conf = config['number_detection']['tesseract_confidence']
    
    dot_x, dot_y, dot_radius = dot['x'], dot['y'], dot.get('radius', 3)
    
    # Extract region
    region, (offset_x, offset_y) = extract_dot_region(image_gray, dot_x, dot_y, search_radius)
    
    if region.size == 0:
        return None
    
    # Local dot position
    dot_x_local = dot_x - offset_x
    dot_y_local = dot_y - offset_y
    
    # Multi-strategy preprocessing
    preprocessed = preprocess_multi_strategy(region, dot_x_local, dot_y_local, dot_radius)
    
    # Detect
    all_detections = []
    
    for region_data in preprocessed:
        tess_dets = detect_tesseract(region_data, min_conf)
        all_detections.extend(tess_dets)
        
        easy_dets = detect_easyocr(region_data, reader)
        all_detections.extend(easy_dets)
    
    if not all_detections:
        return None
    
    # Vote with strategy awareness
    # Give bonus to detections from 'enhanced' and 'morphology' strategies
    for det in all_detections:
        if det['strategy'] in ['enhanced', 'morphology']:
            det['confidence'] = min(100, det['confidence'] + 5)
    
    number_votes = Counter([d['number'] for d in all_detections])
    most_common, vote_count = number_votes.most_common(1)[0]
    
    # Best of winner
    same_number = [d for d in all_detections if d['number'] == most_common]
    best = max(same_number, key=lambda x: x['confidence'])
    
    # Global coords
    x, y, w, h = best['bbox']
    global_x, global_y = x + offset_x, y + offset_y
    center_x, center_y = global_x + w // 2, global_y + h // 2
    distance = np.sqrt((center_x - dot_x)**2 + (center_y - dot_y)**2)
    
    return {
        'dot_id': dot['id'],
        'number': most_common,
        'global_x': center_x,
        'global_y': center_y,
        'bbox': (global_x, global_y, w, h),
        'confidence': best['confidence'],
        'vote_count': vote_count,
        'total_votes': len(all_detections),
        'distance_from_dot': distance,
        'method': best['method'],
        'strategy': best['strategy']
    }


def aggressive_relaxed_pass(image_gray, undetected_dots, config, reader):
    """
    Very aggressive third pass for stubborn dots
    - Larger radius
    - Lower confidence
    - More preprocessing
    """
    if not undetected_dots:
        return []
    
    logging.info(f"\nAGGRESSIVE PASS: {len(undetected_dots)} stubborn dots...")
    
    # Save original
    orig_radius = config['number_detection']['search_radius']
    orig_conf = config['number_detection']['tesseract_confidence']
    
    # VERY relaxed settings
    config['number_detection']['search_radius'] = int(orig_radius * 2)  # 2x radius!
    config['number_detection']['tesseract_confidence'] = 20  # Very low!
    
    aggressive = []
    
    for dot in undetected_dots:
        det = detect_number_for_dot(image_gray, dot, config, reader)
        if det:
            det['aggressive'] = True
            aggressive.append(det)
            logging.info(f"  ✓ Dot {dot['id']}: {det['number']} "
                        f"(conf={det['confidence']}, votes={det['vote_count']}/{det['total_votes']})")
    
    # Restore
    config['number_detection']['search_radius'] = orig_radius
    config['number_detection']['tesseract_confidence'] = orig_conf
    
    logging.info(f"Aggressive pass recovered: {len(aggressive)}")
    return aggressive


def validate_distance(detections):
    """Distance validation using MAD"""
    if len(detections) < 5:
        return detections
    
    distances = [d['distance_from_dot'] for d in detections]
    median = np.median(distances)
    mad = np.median([abs(d - median) for d in distances])
    
    threshold = 3.0 * mad  # Slightly more permissive
    
    filtered = [d for d in detections 
                if abs(d['distance_from_dot'] - median) <= threshold]
    
    if len(filtered) < len(detections):
        logging.info(f"Distance filter: {len(detections)} → {len(filtered)}")
    
    return filtered


def deduplicate_global(detections, threshold=20):
    """Remove duplicates"""
    if len(detections) <= 1:
        return detections
    
    sorted_dets = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    keep = []
    
    for det in sorted_dets:
        too_close = False
        for kept in keep:
            dist = np.sqrt((det['global_x'] - kept['global_x'])**2 + 
                          (det['global_y'] - kept['global_y'])**2)
            if dist < threshold:
                too_close = True
                break
        
        if not too_close:
            keep.append(det)
    
    if len(keep) < len(detections):
        logging.info(f"Deduplication: {len(detections)} → {len(keep)}")
    
    return keep


def visualize(image, detections, dots, output_path):
    """Visualize with strategy colors"""
    vis = image.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    
    dot_dict = {d['id']: d for d in dots}
    
    # Draw dots
    for dot in dots:
        cv2.circle(vis, (dot['x'], dot['y']), 3, (255, 0, 0), -1)
    
    # Draw detections with color by strategy
    for det in detections:
        x, y, w, h = det['bbox']
        
        # Color by strategy
        if det.get('aggressive'):
            color = (0, 0, 255)  # Red = aggressive pass
        elif det.get('strategy') == 'enhanced':
            color = (0, 255, 255)  # Yellow = contrast enhanced
        elif det.get('strategy') == 'morphology':
            color = (255, 0, 255)  # Magenta = morphology
        else:
            color = (0, 255, 0)  # Green = normal
        
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        cv2.circle(vis, (det['global_x'], det['global_y']), 3, (0, 0, 255), -1)
        
        # Line to dot
        dot = dot_dict[det['dot_id']]
        cv2.line(vis, (dot['x'], dot['y']), (det['global_x'], det['global_y']), 
                (200, 200, 200), 1)
        
        # Label
        marker = "!" if det.get('aggressive') else ""
        label = f"{det['number']}{marker}"
        cv2.putText(vis, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, color, 2)
    
    cv2.imwrite(str(output_path), vis)


def run_dot_anchored_detection(config, picture_name, expected_range):
    """Main entry - ROBUST version targeting 80%+"""
    logging.info("="*60)
    logging.info("ROBUST DETECTION - Target 80%+ accuracy")
    logging.info("="*60)
    
    base_path = config['_base_path']
    config_dir = base_path / config['paths']['config_dir']
    
    # Load dots
    dots = load_dot_coordinates(config_dir / config['filenames']['global_dots'])
    if not dots:
        logging.error("No dots!")
        return
    
    logging.info(f"Processing {len(dots)} dots")
    if expected_range:
        logging.info(f"Expected: {expected_range[0]}-{expected_range[1]}")
    
    # Load image
    picture_path = base_path / config['paths']['pictures_dir'] / picture_name
    if not picture_path.exists():
        logging.error("Image not found")
        return
    
    image = cv2.imread(str(picture_path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    reader = setup_ocr(config)
    
    # PHASE 1: Normal detection
    logging.info("\nPHASE 1: Standard detection...")
    all_dets = []
    
    for i, dot in enumerate(dots, 1):
        if i % 25 == 0:
            logging.info(f"  {i}/{len(dots)}...")
        
        det = detect_number_for_dot(gray, dot, config, reader)
        if det:
            all_dets.append(det)
    
    logging.info(f"Pass 1: {len(all_dets)}/{len(dots)} ({len(all_dets)/len(dots)*100:.1f}%)")
    
    # PHASE 2: Validation
    logging.info("\nPHASE 2: Validation...")
    all_dets = validate_distance(all_dets)
    all_dets = deduplicate_global(all_dets, config['number_detection'].get('duplicate_threshold', 20))
    
    # Range filter
    if expected_range:
        min_n, max_n = expected_range
        all_dets = [d for d in all_dets if min_n <= d['number'] <= max_n]
    
    # PHASE 3: Aggressive pass for remaining
    detected_ids = set(d['dot_id'] for d in all_dets)
    undetected = [d for d in dots if d['id'] not in detected_ids]
    
    if undetected:
        aggressive_dets = aggressive_relaxed_pass(gray, undetected, config, reader)
        
        # Validate aggressive detections
        if expected_range:
            aggressive_dets = [d for d in aggressive_dets if min_n <= d['number'] <= max_n]
        
        # Only keep if they don't conflict with existing
        for det in aggressive_dets:
            conflict = False
            for existing in all_dets:
                dist = np.sqrt((det['global_x'] - existing['global_x'])**2 + 
                              (det['global_y'] - existing['global_y'])**2)
                if dist < 20:
                    conflict = True
                    break
            
            if not conflict:
                all_dets.append(det)
    
    # Final results
    all_dets.sort(key=lambda x: x['dot_id'])
    accuracy = (len(all_dets) / len(dots)) * 100 if dots else 0
    
    # Count by strategy
    strategy_counts = Counter([d.get('strategy', 'unknown') for d in all_dets])
    aggressive_count = sum(1 for d in all_dets if d.get('aggressive'))
    
    logging.info(f"\n{'='*60}")
    logging.info(f"FINAL: {len(all_dets)}/{len(dots)} ({accuracy:.1f}%)")
    logging.info(f"Strategy breakdown:")
    for strategy, count in strategy_counts.most_common():
        logging.info(f"  {strategy}: {count}")
    if aggressive_count:
        logging.info(f"  aggressive pass: {aggressive_count}")
    logging.info(f"{'='*60}\n")
    
    # Save
    output_json = config_dir / config['filenames']['global_numbers']
    with open(output_json, 'w') as f:
        json.dump({
            "detection_method": "Robust Multi-Strategy",
            "total_dots": len(dots),
            "total_numbers": len(all_dets),
            "detection_accuracy": round(accuracy, 2),
            "numbers": [
                {
                    "dot_id": d['dot_id'],
                    "number": d['number'],
                    "global_coordinates": {"x": d['global_x'], "y": d['global_y']},
                    "bbox": d['bbox'],
                    "confidence": d['confidence'],
                    "vote_percentage": round((d['vote_count'] / d['total_votes']) * 100, 1),
                    "distance_from_dot": round(d['distance_from_dot'], 2),
                    "method": d['method'],
                    "strategy": d.get('strategy', 'unknown'),
                    "aggressive": d.get('aggressive', False)
                }
                for d in all_dets
            ]
        }, f, indent=2)
    
    logging.info(f"Saved: {output_json}")
    
    # Visualize
    visualize(image, all_dets, dots, base_path / config['filenames']['main_with_numbers'])
    
    # Missing report
    if expected_range:
        detected_nums = set(d['number'] for d in all_dets)
        expected_nums = set(range(expected_range[0], expected_range[1] + 1))
        missing = sorted(expected_nums - detected_nums)
        if missing:
            logging.warning(f"Missing numbers: {missing[:30]}...")
    
    detected_ids = set(d['dot_id'] for d in all_dets)
    missing_dots = sorted([d['id'] for d in dots if d['id'] not in detected_ids])
    if missing_dots:
        logging.warning(f"Missing dots: {missing_dots[:30]}...")
    
    logging.info("Done!\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')