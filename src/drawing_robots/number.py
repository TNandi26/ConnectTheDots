"""
Number Detection - Point-Centered ROI Search

KEY STRATEGY:
- We KNOW where dots are → numbers are ALWAYS near dots
- Search around each dot (radius ~50-80px)
- High thresholds: 130, 140, 150
- Debug images for EVERYTHING
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
                logging.warning("CUDA not available, using CPU for EasyOCR")
            
            reader = easyocr.Reader(['en'], gpu=gpu_enabled)
            logging.info(f"✓ EasyOCR initialized (GPU: {gpu_enabled})")
        except Exception as e:
            logging.warning(f"EasyOCR initialization failed: {e}")
    
    return reader


def load_dots_from_segments(global_dots_json):
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
            if "global_coordinates" in c and "id" in c
        ]
        
        logging.info(f"Loaded {len(dots)} dots from global detection")
        return dots
    except Exception as e:
        logging.error(f"Failed to load dots: {e}")
        return []


def load_segment_mapping(segments_json):
    """Load segment coordinate mapping"""
    try:
        with open(segments_json) as f:
            mapping = json.load(f)
        logging.info(f"Loaded {len(mapping)} segment mappings")
        return mapping
    except Exception as e:
        logging.error(f"Failed to load segment mapping: {e}")
        return {}


def find_dots_in_segment(dots, segment_bounds):
    """Find which dots belong to this segment"""
    x1, y1, x2, y2 = segment_bounds
    
    padding = 30
    x1_padded = max(0, x1 - padding)
    y1_padded = max(0, y1 - padding)
    x2_padded = x2 + padding
    y2_padded = y2 + padding
    
    segment_dots = [
        dot for dot in dots
        if x1_padded <= dot['x'] < x2_padded and y1_padded <= dot['y'] < y2_padded
    ]
    
    return segment_dots


def count_holes_in_region(binary_region):
    """Count enclosed holes (for digit 8 detection)"""
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
    """Analyze shape for digit 8 detection"""
    if binary_region.size == 0:
        return {'holes': 0, 'aspect_ratio': 0, 'solidity': 0}
    
    if len(binary_region.shape) == 3:
        binary_region = cv2.cvtColor(binary_region, cv2.COLOR_BGR2GRAY)
    
    _, binary = cv2.threshold(binary_region, 127, 255, cv2.THRESH_BINARY)
    
    holes = count_holes_in_region(binary)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {'holes': holes, 'aspect_ratio': 0, 'solidity': 0}
    
    main_contour = max(contours, key=cv2.contourArea)
    
    x, y, w, h = cv2.boundingRect(main_contour)
    aspect_ratio = w / h if h > 0 else 0
    
    area = cv2.contourArea(main_contour)
    hull = cv2.convexHull(main_contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    
    return {
        'holes': holes,
        'aspect_ratio': aspect_ratio,
        'solidity': solidity
    }


def is_likely_digit_8(shape_features):
    """Check if shape matches digit 8"""
    if shape_features['holes'] == 2:
        if 0.4 <= shape_features['aspect_ratio'] <= 0.9:
            if shape_features['solidity'] > 0.65:
                return True
    return False


def search_number_around_dot(segment_image, dot, thresholds, debug_dir, segment_name):
    """
    Search for number around a specific dot
    DOT IS AT CENTER OF ROI - number can be anywhere around it
    """
    if len(segment_image.shape) == 3:
        gray = cv2.cvtColor(segment_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = segment_image.copy()
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Define SQUARE search region with DOT AT CENTER
    search_radius = 70  # Larger radius for better coverage
    
    # Calculate ROI with dot at center
    roi_half_size = search_radius
    x_center = dot['x']
    y_center = dot['y']
    
    x1 = max(0, x_center - roi_half_size)
    y1 = max(0, y_center - roi_half_size)
    x2 = min(gray.shape[1], x_center + roi_half_size)
    y2 = min(gray.shape[0], y_center + roi_half_size)
    
    search_region = enhanced[y1:y2, x1:x2].copy()
    
    # Erase dot at CENTER of search region
    dot_local_x = x_center - x1
    dot_local_y = y_center - y1
    
    if 0 <= dot_local_x < search_region.shape[1] and 0 <= dot_local_y < search_region.shape[0]:
        # Mark dot with WHITE circle for debugging
        cv2.circle(search_region, (dot_local_x, dot_local_y), dot['radius'] + 3, 255, -1)
    
    # Save search region debug
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / f"{segment_name}_dot{dot['id']}_search_region.jpg"), search_region)
    
    all_rois = []
    
    for thresh in thresholds:
        # Binary threshold
        _, binary = cv2.threshold(search_region, thresh, 255, cv2.THRESH_BINARY)
        
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
        
        # Save threshold debug
        if debug_dir:
            cv2.imwrite(str(debug_dir / f"{segment_name}_dot{dot['id']}_thresh{thresh}.jpg"), cleaned)
        
        # Upscale for OCR
        h, w = cleaned.shape
        scale_factor = 4
        scaled = cv2.resize(cleaned, (w * scale_factor, h * scale_factor), 
                           interpolation=cv2.INTER_CUBIC)
        
        # Light denoising
        denoised = cv2.fastNlMeansDenoising(scaled, None, 3, 7, 21)
        
        all_rois.append({
            'image': denoised,
            'original': cleaned,
            'threshold': thresh,
            'scale_factor': scale_factor,
            'offset_x': x1,
            'offset_y': y1,
            'dot_id': dot['id']
        })
    
    return all_rois


def detect_with_tesseract(processed_data, min_conf):
    """Detect numbers using Tesseract"""
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
                        shape_features = analyze_digit_shape(region)
                        
                        detections.append({
                            'number': int(text),
                            'x': x + w // 2,
                            'y': y + h // 2,
                            'bbox': (x, y, w, h),
                            'confidence': conf,
                            'method': 'tesseract',
                            'psm': psm,
                            'shape_features': shape_features
                        })
        except Exception as e:
            logging.debug(f"Tesseract PSM {psm} failed: {e}")
    
    return detections


def detect_with_easyocr(processed_data, reader):
    """Detect numbers using EasyOCR"""
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
                    shape_features = analyze_digit_shape(region)
                    
                    adjusted_conf = int(conf * 100)
                    if len(text) > 1:
                        adjusted_conf = min(100, int(adjusted_conf * 1.3))
                    
                    detections.append({
                        'number': int(text),
                        'x': x + w // 2,
                        'y': y + h // 2,
                        'bbox': (x, y, w, h),
                        'confidence': adjusted_conf,
                        'method': 'easyocr',
                        'shape_features': shape_features
                    })
    except Exception as e:
        logging.debug(f"EasyOCR failed: {e}")
    
    return detections


def vote_on_detections(all_detections, position_threshold=15):
    """Voting system with digit 8 correction"""
    if not all_detections:
        return []
    
    groups = []
    
    for det in all_detections:
        placed = False
        
        for group in groups:
            group_center_x = np.mean([d['x'] for d in group])
            group_center_y = np.mean([d['y'] for d in group])
            
            dist = np.sqrt(
                (det['x'] - group_center_x)**2 + 
                (det['y'] - group_center_y)**2
            )
            
            if dist < position_threshold:
                group.append(det)
                placed = True
                break
        
        if not placed:
            groups.append([det])
    
    logging.debug(f"Grouped {len(all_detections)} detections into {len(groups)} positions")
    
    voted_detections = []
    
    for group in groups:
        has_strong_8 = any(
            is_likely_digit_8(d['shape_features']) 
            for d in group
        )
        
        votes = Counter(d['number'] for d in group)
        
        if has_strong_8 and 8 in votes:
            winning_number = 8
            confidence_boost = 20
        else:
            winning_number = votes.most_common(1)[0][0]
            confidence_boost = 0
        
        winners = [d for d in group if d['number'] == winning_number]
        best = max(winners, key=lambda d: d['confidence'])
        
        avg_x = int(np.mean([d['x'] for d in group]))
        avg_y = int(np.mean([d['y'] for d in group]))
        
        vote_agreement = votes[winning_number] / len(group)
        final_confidence = min(100, best['confidence'] + 
                             int(vote_agreement * 15) + confidence_boost)
        
        voted_detections.append({
            'number': winning_number,
            'x': avg_x,
            'y': avg_y,
            'bbox': best['bbox'],
            'confidence': final_confidence,
            'votes': votes[winning_number],
            'total_detections': len(group),
            'methods': list(set(d['method'] for d in group)),
            'shape_corrected': has_strong_8 and 8 in votes
        })
    
    return voted_detections


def process_segment(segment_path, segment_name, segment_bounds, dots, config, reader, debug_dir):
    """Process segment using point-centered search"""
    segment_image = cv2.imread(str(segment_path))
    if segment_image is None:
        return []
    
    x1, y1, x2, y2 = segment_bounds
    segment_dots = find_dots_in_segment(dots, segment_bounds)
    
    # Convert to local coordinates
    local_dots = [
        {
            'id': dot['id'],
            'x': dot['x'] - x1,
            'y': dot['y'] - y1,
            'radius': dot['radius']
        }
        for dot in segment_dots
    ]
    
    if not local_dots:
        return []
    
    logging.debug(f"{segment_name}: {len(local_dots)} dots")
    
    # HIGH THRESHOLDS as requested
    thresholds = [130, 140, 150]
    
    all_detections = []
    
    # Search around EACH dot
    for dot in local_dots:
        rois = search_number_around_dot(segment_image, dot, thresholds, 
                                       debug_dir, segment_name)
        
        # OCR each ROI
        for roi_data in rois:
            tess_dets = detect_with_tesseract(roi_data, 
                                             config['number_detection']['tesseract_confidence'])
            easy_dets = detect_with_easyocr(roi_data, reader)
            
            # Add offset to convert back to segment coordinates
            for det in tess_dets + easy_dets:
                det['x'] += roi_data['offset_x']
                det['y'] += roi_data['offset_y']
                det['dot_id'] = dot['id']
            
            all_detections.extend(tess_dets + easy_dets)
    
    if not all_detections:
        return []
    
    # Vote
    voted = vote_on_detections(all_detections)
    
    # Convert to global
    global_detections = []
    for det in voted:
        global_detections.append({
            'number': det['number'],
            'global_x': det['x'] + x1,
            'global_y': det['y'] + y1,
            'confidence': det['confidence'],
            'votes': det['votes'],
            'total_detections': det['total_detections'],
            'methods': det['methods'],
            'segment': segment_name,
            'shape_corrected': det.get('shape_corrected', False)
        })
    
    logging.debug(f"  Detected {len(global_detections)} numbers")
    return global_detections


def remove_global_duplicates(all_numbers, threshold=20):
    """Remove duplicates from overlapping segments"""
    if len(all_numbers) <= 1:
        return all_numbers
    
    sorted_nums = sorted(all_numbers, key=lambda x: x['confidence'], reverse=True)
    unique = []
    
    for num in sorted_nums:
        is_duplicate = False
        
        for existing in unique:
            dist = np.sqrt(
                (num['global_x'] - existing['global_x'])**2 +
                (num['global_y'] - existing['global_y'])**2
            )
            
            if dist < threshold:
                if num['number'] == existing['number']:
                    is_duplicate = True
                    break
                else:
                    if num['confidence'] > existing['confidence']:
                        unique.remove(existing)
                        break
                    else:
                        is_duplicate = True
                        break
        
        if not is_duplicate:
            unique.append(num)
    
    return unique


def visualize_on_main_image(main_image_path, detections, output_path):
    """Visualize detected numbers"""
    image = cv2.imread(str(main_image_path))
    if image is None:
        return
    
    for det in detections:
        x = det['global_x']
        y = det['global_y']
        number = det['number']
        
        if det.get('shape_corrected'):
            color = (255, 0, 255)
        elif len(det['methods']) > 1:
            color = (0, 255, 0)
        else:
            color = (0, 165, 255)
        
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        
        label = f"{number}"
        cv2.putText(image, label, (x + 8, y - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(image, label, (x + 8, y - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    cv2.imwrite(str(output_path), image)
    logging.info(f"Visualization saved: {output_path}")


def run_segment_based_detection(config, picture_name, expected_range=None):
    """Main entry point - Point-centered search"""
    logging.info("=" * 60)
    logging.info("POINT-CENTERED NUMBER DETECTION")
    logging.info("Thresholds: 130, 140, 150")
    logging.info("=" * 60)
    
    base_path = config['_base_path']
    config_dir = base_path / config['paths']['config_dir']
    segments_dir = base_path / config['paths']['segments_overlap_dir']
    debug_dir = base_path / config['paths']['number_debug_dir']
    
    # Clear debug
    import shutil
    if debug_dir.exists():
        shutil.rmtree(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    # Load
    dots = load_dots_from_segments(config_dir / config['filenames']['global_dots'])
    if not dots:
        logging.error("No dots loaded!")
        return
    
    segments_json = segments_dir / config['filenames']['overlap_segments_meta']
    segment_mapping = load_segment_mapping(segments_json)
    if not segment_mapping:
        logging.error("No segment mapping loaded!")
        return
    
    logging.info(f"Processing {len(dots)} dots across {len(segment_mapping)} segments\n")
    
    # OCR
    reader = setup_ocr(config)
    
    # Process segments
    all_detections = []
    
    for segment_name, segment_info in segment_mapping.items():
        segment_path = segments_dir / segment_name
        
        if not segment_path.exists():
            continue
        
        x1 = segment_info['start']['x']
        y1 = segment_info['start']['y']
        x2 = segment_info['end']['x']
        y2 = segment_info['end']['y']
        
        segment_bounds = (x1, y1, x2, y2)
        
        detections = process_segment(
            segment_path, segment_name, segment_bounds,
            dots, config, reader, debug_dir
        )
        
        all_detections.extend(detections)
    
    logging.info(f"\nTotal detections: {len(all_detections)}")
    
    # Remove duplicates
    logging.info("Removing duplicates...")
    unique_detections = remove_global_duplicates(all_detections)
    
    # Filter by range
    if expected_range:
        min_num, max_num = expected_range
        filtered = [
            d for d in unique_detections
            if min_num <= d['number'] <= max_num
        ]
        logging.info(f"Filtered to [{min_num}-{max_num}]: {len(unique_detections)} → {len(filtered)}")
        unique_detections = filtered
    
    # Save
    output_json = config_dir / config['filenames']['global_numbers']
    with open(output_json, 'w') as f:
        json.dump({
            "detection_method": "Point-Centered Search with High Thresholds",
            "thresholds_used": [130, 140, 150],
            "total_numbers": len(unique_detections),
            "numbers": [
                {
                    "number": d['number'],
                    "global_coordinates": {
                        "x": d['global_x'],
                        "y": d['global_y']
                    },
                    "confidence": d['confidence'],
                    "votes": d['votes'],
                    "total_detections": d['total_detections'],
                    "methods": d['methods'],
                    "shape_corrected": d.get('shape_corrected', False)
                }
                for d in unique_detections
            ]
        }, f, indent=2)
    
    logging.info(f"Saved: {output_json}")
    
    # Visualize
    picture_path = base_path / config['paths']['pictures_dir'] / picture_name
    viz_path = base_path / config['filenames']['main_with_numbers']
    visualize_on_main_image(picture_path, unique_detections, viz_path)
    
    # Stats
    logging.info(f"\n{'=' * 60}")
    logging.info(f"RESULTS: {len(unique_detections)} numbers detected")
    
    shape_corrected = sum(1 for d in unique_detections if d.get('shape_corrected'))
    both_ocr = sum(1 for d in unique_detections if len(d['methods']) > 1)
    
    logging.info(f"OCR agreement: {both_ocr}")
    logging.info(f"Shape-corrected 8s: {shape_corrected}")
    
    if expected_range:
        detected_nums = set(d['number'] for d in unique_detections)
        expected_nums = set(range(expected_range[0], expected_range[1] + 1))
        accuracy = (len(detected_nums) / len(expected_nums)) * 100
        
        logging.info(f"Accuracy: {accuracy:.1f}%")
        
        missing = sorted(expected_nums - detected_nums)
        if missing:
            logging.warning(f"Missing: {missing[:20]}...")
    
    logging.info(f"\n🔍 DEBUG IMAGES: {debug_dir}/")
    logging.info(f"   Look for: *_dot*_search_region.jpg")
    logging.info(f"   And: *_dot*_thresh130.jpg, thresh140.jpg, thresh150.jpg")
    logging.info(f"{'=' * 60}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')