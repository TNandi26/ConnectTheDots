"""
Number Detection - Dynamic ROI per Dot

STRATEGY:
1. Load full image
2. For EACH dot: cut out ROI around it (radius ~70px)
3. Save ROI coordinates for later (to convert to global)
4. OCR on each ROI
5. Convert back to global coordinates
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
    """Initialize OCR engines"""
    pytesseract.pytesseract.tesseract_cmd = config['number_detection']['tesseract_path']
    
    reader = None
    if config['number_detection']['use_easyocr']:
        try:
            gpu_enabled = config['number_detection']['easyocr_gpu']
            import torch
            if gpu_enabled and not torch.cuda.is_available():
                gpu_enabled = False
                logging.warning("CUDA not available, using CPU")
            
            reader = easyocr.Reader(['en'], gpu=gpu_enabled)
            logging.info(f"✓ EasyOCR initialized (GPU: {gpu_enabled})")
        except Exception as e:
            logging.warning(f"EasyOCR failed: {e}")
    
    return reader


def load_dots(global_dots_json):
    """Load dot coordinates"""
    try:
        with open(global_dots_json) as f:
            data = json.load(f)
        
        dots = [
            {
                "id": c["id"], 
                "x": c["global_coordinates"]["x"],
                "y": c["global_coordinates"]["y"], 
                "radius": c.get("radius", 5)
            }
            for c in data.get("circles", [])
        ]
        
        logging.info(f"Loaded {len(dots)} dots")
        return dots
    except Exception as e:
        logging.error(f"Failed to load dots: {e}")
        return []


def count_holes_in_region(binary_region):
    """Count holes for digit 8 detection"""
    if binary_region.size == 0:
        return 0
    
    if len(binary_region.shape) == 3:
        binary_region = cv2.cvtColor(binary_region, cv2.COLOR_BGR2GRAY)
    
    _, binary = cv2.threshold(binary_region, 127, 255, cv2.THRESH_BINARY)
    
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    
    if hierarchy is None:
        return 0
    
    hole_count = sum(1 for h in hierarchy[0] if h[3] != -1)
    return hole_count


def analyze_digit_shape(binary_region):
    """Analyze shape for digit 8"""
    if binary_region.size == 0:
        return {'holes': 0, 'is_8': False}
    
    if len(binary_region.shape) == 3:
        binary_region = cv2.cvtColor(binary_region, cv2.COLOR_BGR2GRAY)
    
    _, binary = cv2.threshold(binary_region, 127, 255, cv2.THRESH_BINARY)
    
    holes = count_holes_in_region(binary)
    is_8 = (holes == 2)
    
    return {'holes': holes, 'is_8': is_8}


def extract_roi_around_dot(image, dot, roi_radius=70):
    """
    Extract ROI around dot
    Returns: roi_image, roi_coordinates
    """
    h, w = image.shape[:2]
    
    # Calculate ROI bounds with dot at CENTER
    x_center = dot['x']
    y_center = dot['y']
    
    x1 = max(0, x_center - roi_radius)
    y1 = max(0, y_center - roi_radius)
    x2 = min(w, x_center + roi_radius)
    y2 = min(h, y_center + roi_radius)
    
    # Extract ROI
    roi = image[y1:y2, x1:x2].copy()
    
    # Calculate dot position within ROI
    dot_local_x = x_center - x1
    dot_local_y = y_center - y1
    
    # Erase dot from ROI
    cv2.circle(roi, (dot_local_x, dot_local_y), dot['radius'] + 3, 255, -1)
    
    roi_info = {
        'roi': roi,
        'global_offset': (x1, y1),  # To convert back to global coords
        'dot_local': (dot_local_x, dot_local_y),
        'dot_id': dot['id'],
        'bounds': (x1, y1, x2, y2)
    }
    
    return roi_info


def process_roi_with_thresholds(roi_info, config, debug_dir):
    """
    Process ROI with multiple thresholds FROM CONFIG
    Returns list of processed images ready for OCR
    """
    roi = roi_info['roi']
    dot_id = roi_info['dot_id']
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(roi)
    
    # Get thresholds from config
    thresholds = config['number_detection']['mask_thresholds']
    
    processed_rois = []
    
    for thresh in thresholds:
        # Binary threshold
        _, binary = cv2.threshold(enhanced, thresh, 255, cv2.THRESH_BINARY)
        
        # Ensure numbers are white on black
        if np.mean(binary) < 127:
            binary = cv2.bitwise_not(binary)
        
        # Character separation
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        separated = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel_erode, iterations=1)
        
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        separated = cv2.morphologyEx(separated, cv2.MORPH_DILATE, kernel_dilate, iterations=1)
        
        # Clean noise
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(separated, cv2.MORPH_OPEN, kernel_open, iterations=1)
        
        # Save debug image
        if debug_dir:
            debug_vis = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
            
            # Mark dot center
            dot_x, dot_y = roi_info['dot_local']
            cv2.line(debug_vis, (dot_x - 8, dot_y), (dot_x + 8, dot_y), (0, 0, 255), 1)
            cv2.line(debug_vis, (dot_x, dot_y - 8), (dot_x, dot_y + 8), (0, 0, 255), 1)
            cv2.putText(debug_vis, f"Dot {dot_id} | Thresh: {thresh}", 
                       (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            cv2.imwrite(str(debug_dir / f"dot{dot_id:03d}_thresh{thresh}.jpg"), debug_vis)
        
        # Upscale for OCR
        h, w = cleaned.shape
        scale_factor = 4
        scaled = cv2.resize(cleaned, (w * scale_factor, h * scale_factor), 
                           interpolation=cv2.INTER_CUBIC)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(scaled, None, 3, 7, 21)
        
        processed_rois.append({
            'image': denoised,
            'original': cleaned,
            'threshold': thresh,
            'scale_factor': scale_factor
        })
    
    return processed_rois


def detect_with_tesseract(processed_data, min_conf):
    """Detect with Tesseract"""
    detections = []
    
    for psm in [6, 11, 12]:
        config_str = f"--psm {psm} --oem 3 -c tessedit_char_whitelist=0123456789"
        
        try:
            pil_img = Image.fromarray(processed_data['image'])
            data = pytesseract.image_to_data(
                pil_img, config=config_str,
                output_type=pytesseract.Output.DICT
            )
            
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                conf = int(data['conf'][i]) if data['conf'][i] != -1 else 0
                
                if text and text.isdigit() and conf > min_conf:
                    sf = processed_data['scale_factor']
                    x = data['left'][i] // sf
                    y = data['top'][i] // sf
                    w = data['width'][i] // sf
                    h = data['height'][i] // sf
                    
                    if w > 0 and h > 0 and w < 200 and h < 100:
                        region = processed_data['original'][y:y+h, x:x+w]
                        shape_info = analyze_digit_shape(region)
                        
                        detections.append({
                            'number': int(text),
                            'local_x': x + w // 2,
                            'local_y': y + h // 2,
                            'confidence': conf,
                            'method': 'tesseract',
                            'shape_info': shape_info
                        })
        except:
            pass
    
    return detections


def detect_with_easyocr(processed_data, reader):
    """Detect with EasyOCR"""
    if reader is None:
        return []
    
    detections = []
    
    try:
        results = reader.readtext(
            processed_data['image'],
            allowlist='0123456789',
            detail=1
        )
        
        for bbox, text, conf in results:
            if text.isdigit():
                sf = processed_data['scale_factor']
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                
                x = int(min(x_coords)) // sf
                y = int(min(y_coords)) // sf
                w = int(max(x_coords) - min(x_coords)) // sf
                h = int(max(y_coords) - min(y_coords)) // sf
                
                if w > 0 and h > 0 and w < 200 and h < 100:
                    region = processed_data['original'][y:y+h, x:x+w]
                    shape_info = analyze_digit_shape(region)
                    
                    adjusted_conf = int(conf * 100)
                    if len(text) > 1:
                        adjusted_conf = min(100, int(adjusted_conf * 1.3))
                    
                    detections.append({
                        'number': int(text),
                        'local_x': x + w // 2,
                        'local_y': y + h // 2,
                        'confidence': adjusted_conf,
                        'method': 'easyocr',
                        'shape_info': shape_info
                    })
    except:
        pass
    
    return detections


def vote_on_detections(all_detections):
    """Vote with digit 8 correction"""
    if not all_detections:
        return None
    
    votes = Counter(d['number'] for d in all_detections)
    
    # Check for digit 8 signature
    has_8_signature = any(d['shape_info']['is_8'] for d in all_detections)
    
    if has_8_signature and 8 in votes:
        winning_number = 8
    else:
        winning_number = votes.most_common(1)[0][0]
    
    # Get best detection
    winners = [d for d in all_detections if d['number'] == winning_number]
    best = max(winners, key=lambda d: d['confidence'])
    
    # Average position
    avg_x = int(np.mean([d['local_x'] for d in all_detections]))
    avg_y = int(np.mean([d['local_y'] for d in all_detections]))
    
    return {
        'number': winning_number,
        'local_x': avg_x,
        'local_y': avg_y,
        'confidence': best['confidence'],
        'votes': votes[winning_number],
        'total_detections': len(all_detections),
        'methods': list(set(d['method'] for d in all_detections)),
        'shape_corrected': has_8_signature and 8 in votes
    }


def process_dot(image, dot, config, reader, debug_dir):
    """
    Process a single dot:
    1. Extract ROI around dot
    2. Apply thresholds
    3. Run OCR
    4. Vote on results
    """
    # Extract ROI
    roi_info = extract_roi_around_dot(image, dot, roi_radius=30)
    
    # Save ROI debug
    if debug_dir:
        roi_vis = cv2.cvtColor(roi_info['roi'], cv2.COLOR_GRAY2BGR)
        dot_x, dot_y = roi_info['dot_local']
        
        # Crosshair at center
        cv2.line(roi_vis, (dot_x - 10, dot_y), (dot_x + 10, dot_y), (0, 0, 255), 2)
        cv2.line(roi_vis, (dot_x, dot_y - 10), (dot_x, dot_y + 10), (0, 0, 255), 2)
        cv2.rectangle(roi_vis, (0, 0), (roi_vis.shape[1]-1, roi_vis.shape[0]-1), (0, 255, 0), 2)
        cv2.putText(roi_vis, f"Dot {dot['id']} - ROI {roi_info['roi'].shape[1]}x{roi_info['roi'].shape[0]}", 
                   (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        cv2.imwrite(str(debug_dir / f"dot{dot['id']:03d}_0_roi.jpg"), roi_vis)
    
    # Process with thresholds
    processed_rois = process_roi_with_thresholds(roi_info, config, debug_dir)
    
    # OCR
    all_detections = []
    
    for processed in processed_rois:
        tess_dets = detect_with_tesseract(processed, 
                                          config['number_detection']['tesseract_confidence'])
        easy_dets = detect_with_easyocr(processed, reader)
        
        all_detections.extend(tess_dets + easy_dets)
    
    if not all_detections:
        return None
    
    # Vote
    result = vote_on_detections(all_detections)
    
    if result:
        # Convert to global coordinates
        offset_x, offset_y = roi_info['global_offset']
        result['global_x'] = result['local_x'] + offset_x
        result['global_y'] = result['local_y'] + offset_y
        result['dot_id'] = dot['id']
    
    return result


def visualize(image, detections, dots, output_path):
    """Create visualization"""
    vis = image.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    
    # Draw dots
    for dot in dots:
        cv2.circle(vis, (dot['x'], dot['y']), 3, (255, 0, 0), -1)
    
    # Draw numbers
    for det in detections:
        x = det['global_x']
        y = det['global_y']
        
        if det.get('shape_corrected'):
            color = (255, 0, 255)
        elif len(det['methods']) > 1:
            color = (0, 255, 0)
        else:
            color = (0, 165, 255)
        
        cv2.circle(vis, (x, y), 5, (0, 0, 255), -1)
        
        # Find corresponding dot
        for dot in dots:
            if dot['id'] == det['dot_id']:
                cv2.line(vis, (dot['x'], dot['y']), (x, y), (150, 150, 150), 1)
                break
        
        label = f"{det['number']}"
        cv2.putText(vis, label, (x + 8, y - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(vis, label, (x + 8, y - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    cv2.imwrite(str(output_path), vis)


def run_segment_based_detection(config, picture_name, expected_range=None):
    """Main - Dynamic ROI per dot"""
    print("\n" + "="*60)
    print("DYNAMIC ROI NUMBER DETECTION")
    print("Strategy: Extract ROI around each dot")
    print("Thresholds: 130, 140, 150")
    print("="*60 + "\n")
    
    base_path = config['_base_path']
    config_dir = base_path / config['paths']['config_dir']
    debug_dir = base_path / config['paths']['number_debug_dir']
    
    # Clear debug
    import shutil
    if debug_dir.exists():
        shutil.rmtree(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dots
    dots = load_dots(config_dir / config['filenames']['global_dots'])
    if not dots:
        print("❌ No dots loaded!")
        return
    
    # Load full image
    picture_path = base_path / config['paths']['pictures_dir'] / picture_name
    if not picture_path.exists():
        print(f"❌ Image not found: {picture_path}")
        return
    
    image = cv2.imread(str(picture_path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    print(f"📷 Image: {gray.shape[1]}x{gray.shape[0]}")
    print(f"🔵 Dots: {len(dots)}\n")
    
    if expected_range:
        print(f"🎯 Expected range: {expected_range[0]}-{expected_range[1]}\n")
    
    # OCR
    reader = setup_ocr(config)
    
    # Process each dot
    print("="*60)
    print("PROCESSING DOTS")
    print("="*60)
    
    detections = []
    total_dots = len(dots)
    
    for idx, dot in enumerate(dots, 1):
        # Progress bar
        progress = (idx / total_dots) * 100
        bar_length = 40
        filled = int(bar_length * idx / total_dots)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f"\r[{bar}] {progress:.1f}% | Dot {idx}/{total_dots} (ID: {dot['id']})", 
              end='', flush=True)
        
        result = process_dot(gray, dot, config, reader, debug_dir)
        
        if result:
            detections.append(result)
    
    print("\n" + "="*60)
    print(f"✓ All dots processed!")
    print("="*60 + "\n")
    
    # Filter by range
    if expected_range:
        min_n, max_n = expected_range
        filtered = [d for d in detections if min_n <= d['number'] <= max_n]
        print(f"🔍 Filtered to [{min_n}, {max_n}]: {len(detections)} → {len(filtered)}\n")
        detections = filtered
    
    # Save
    print("💾 Saving results...")
    output_json = config_dir / config['filenames']['global_numbers']
    with open(output_json, 'w') as f:
        json.dump({
            "detection_method": "Dynamic ROI per Dot",
            "thresholds": [130, 140, 150],
            "total_numbers": len(detections),
            "numbers": [
                {
                    "dot_id": d['dot_id'],
                    "number": d['number'],
                    "global_coordinates": {"x": d['global_x'], "y": d['global_y']},
                    "confidence": d['confidence'],
                    "votes": d['votes'],
                    "methods": d['methods'],
                    "shape_corrected": d.get('shape_corrected', False)
                }
                for d in detections
            ]
        }, f, indent=2)
    
    print(f"✓ Saved to: {output_json.name}\n")
    
    # Visualize
    print("🎨 Creating visualization...")
    viz_path = base_path / config['filenames']['main_with_numbers']
    visualize(image, detections, dots, viz_path)
    print(f"✓ Visualization saved\n")
    
    # Stats
    print("="*60)
    print("📊 FINAL RESULTS")
    print("="*60)
    print(f"Numbers detected: {len(detections)}/{len(dots)}")
    
    both_ocr = sum(1 for d in detections if len(d['methods']) > 1)
    shape_corrected = sum(1 for d in detections if d.get('shape_corrected'))
    
    print(f"Both OCRs agreed: {both_ocr}")
    print(f"Shape-corrected 8s: {shape_corrected}")
    
    if expected_range:
        detected_nums = set(d['number'] for d in detections)
        expected_nums = set(range(expected_range[0], expected_range[1] + 1))
        accuracy = (len(detected_nums) / len(expected_nums)) * 100
        
        print(f"\nAccuracy: {accuracy:.1f}%")
        
        missing = sorted(expected_nums - detected_nums)
        if missing:
            print(f"Missing ({len(missing)}): {missing[:20]}{'...' if len(missing) > 20 else ''}")
    
    print("\n" + "="*60)
    print("🔍 DEBUG IMAGES")
    print("="*60)
    print(f"Location: {debug_dir}/")
    print(f"  - dot001_0_roi.jpg (ROI with center marked)")
    print(f"  - dot001_thresh130.jpg, 140, 150")
    print("="*60 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')