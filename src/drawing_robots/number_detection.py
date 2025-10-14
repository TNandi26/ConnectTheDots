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
    """Initialize OCR engines from config"""
    pytesseract.pytesseract.tesseract_cmd = config['number_detection']['tesseract_path']
    
    use_easyocr = config['number_detection']['use_easyocr']
    easyocr_gpu = config['number_detection']['easyocr_gpu']
    
    reader = None
    if use_easyocr:
        try:
            reader = easyocr.Reader(['en'], gpu=easyocr_gpu)
            logging.info(f"EasyOCR initialized (GPU: {easyocr_gpu})")
        except Exception as e:
            logging.warning(f"EasyOCR failed to initialize: {e}")
            reader = None
    
    return reader


def load_detected_circles_for_segment(segment_name, detected_circles_json):
    """Load circle coordinates for this segment"""
    try:
        with open(detected_circles_json) as f:
            data = json.load(f)
        
        for segment in data.get('segments', []):
            if segment['segment_name'] == segment_name:
                return [(c['pixel_x'], c['pixel_y'], c['radius']) for c in segment.get('circles', [])]
        
        return []
    except Exception as e:
        logging.warning(f"Could not load circles for {segment_name}: {e}")
        return []


def erase_dots_from_image(image, circles, config):
    """Erase dots by filling with white"""
    if not circles:
        return image
    
    result = image.copy()
    margin = config['number_detection']['dot_erase_margin']
    
    for x, y, r in circles:
        cv2.circle(result, (x, y), r + margin, 255, -1)
    
    logging.info(f"Erased {len(circles)} dots from image")
    return result


def save_debug_image(image, filename, config):
    """Save debug image to number_debug folder"""
    debug_dir = config['_base_path'] / config['paths']['number_debug_dir']
    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / filename), image)


def preprocess_for_ocr(gray_image, threshold_value):
    """Preprocess image for OCR"""
    _, binary = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Upscale 2x for better OCR
    scale_factor = 2
    width = int(binary.shape[1] * scale_factor)
    height = int(binary.shape[0] * scale_factor)
    binary = cv2.resize(binary, (width, height), interpolation=cv2.INTER_CUBIC)
    
    denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
    
    return denoised, scale_factor


def detect_numbers_easyocr(image, scale_factor, reader):
    """Detect numbers using EasyOCR"""
    if reader is None:
        return []
    
    try:
        results = reader.readtext(image, allowlist='0123456789', detail=1)
        
        detections = []
        for bbox, text, conf in results:
            if text.isdigit():
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                
                x = int(min(x_coords)) // scale_factor
                y = int(min(y_coords)) // scale_factor
                w = int(max(x_coords) - min(x_coords)) // scale_factor
                h = int(max(y_coords) - min(y_coords)) // scale_factor
                
                detections.append({
                    'number': int(text),
                    'center': (x + w // 2, y + h // 2),
                    'bbox': (x, y, w, h),
                    'confidence': int(conf * 100),
                    'method': 'easyocr'
                })
        
        return detections
    except Exception as e:
        logging.error(f"EasyOCR detection failed: {e}")
        return []


def detect_numbers_tesseract(image, scale_factor, config_str, min_conf):
    """Detect numbers using Tesseract OCR"""
    try:
        pil_image = Image.fromarray(image)
        data = pytesseract.image_to_data(pil_image, config=config_str, output_type=pytesseract.Output.DICT)
        
        detections = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf = int(data['conf'][i])
            
            if text and text.isdigit() and conf > min_conf:
                x = data['left'][i] // scale_factor
                y = data['top'][i] // scale_factor
                w = data['width'][i] // scale_factor
                h = data['height'][i] // scale_factor
                
                detections.append({
                    'number': int(text),
                    'center': (x + w // 2, y + h // 2),
                    'bbox': (x, y, w, h),
                    'confidence': conf,
                    'method': 'tesseract'
                })
        
        return detections
    except Exception as e:
        logging.error(f"Tesseract detection failed: {e}")
        return []


def detect_numbers_multi_threshold(gray_image, segment_name, config, reader):
    """
    Try multiple thresholds with Tesseract (PSM 11 + PSM 6) and EasyOCR
    
    This is the CORE detection function - tries 10 thresholds x 3 methods = 30 passes
    """
    thresholds = config['number_detection']['thresholds']
    min_conf = config['number_detection']['tesseract_confidence']
    all_detections = []
    
    for thresh in thresholds:
        preprocessed, scale_factor = preprocess_for_ocr(gray_image, thresh)
        
        # Method 1: Tesseract PSM 11 (sparse text)
        config_psm11 = '--psm 11 --oem 3 -c tessedit_char_whitelist=0123456789'
        detections_psm11 = detect_numbers_tesseract(preprocessed, scale_factor, config_psm11, min_conf)
        for det in detections_psm11:
            det['threshold'] = thresh
            det['psm'] = 11
        all_detections.extend(detections_psm11)
        
        # Method 2: Tesseract PSM 6 (uniform block)
        config_psm6 = '--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789'
        detections_psm6 = detect_numbers_tesseract(preprocessed, scale_factor, config_psm6, min_conf)
        for det in detections_psm6:
            det['threshold'] = thresh
            det['psm'] = 6
        all_detections.extend(detections_psm6)
        
        # Method 3: EasyOCR
        easyocr_detections = detect_numbers_easyocr(preprocessed, scale_factor, reader)
        for det in easyocr_detections:
            det['threshold'] = thresh
        all_detections.extend(easyocr_detections)
        
        logging.info(f"Threshold {thresh}: Tesseract={len(detections_psm11)+len(detections_psm6)}, "
                    f"EasyOCR={len(easyocr_detections)}")
    
    logging.info(f"Total detections from all methods: {len(all_detections)}")
    return all_detections


def consensus_voting(detections, config):
    """
    Vote on which number is at each position
    Requires min 3 votes and 50% agreement
    """
    if not detections:
        return []
    
    position_tolerance = config['number_detection']['consensus']['position_tolerance']
    min_votes = config['number_detection']['consensus']['min_votes']
    min_percentage = config['number_detection']['consensus']['min_percentage']
    
    # Group by position
    position_groups = []
    
    for det in detections:
        cx, cy = det['center']
        
        found_group = False
        for group in position_groups:
            group_center = group['center']
            dist = np.sqrt((cx - group_center[0])**2 + (cy - group_center[1])**2)
            
            if dist < position_tolerance:
                group['detections'].append(det)
                found_group = True
                break
        
        if not found_group:
            position_groups.append({
                'center': (cx, cy),
                'detections': [det]
            })
    
    # Vote on each position
    consensus_detections = []
    
    for group in position_groups:
        dets = group['detections']
        
        if len(dets) < min_votes:
            logging.debug(f"Skipped position - only {len(dets)} vote(s)")
            continue
        
        number_votes = Counter([d['number'] for d in dets])
        most_common_number, vote_count = number_votes.most_common(1)[0]
        
        vote_percentage = (vote_count / len(dets)) * 100
        
        if vote_percentage < min_percentage:
            logging.debug(f"Skipped - low consensus ({vote_percentage:.0f}%)")
            continue
        
        same_number_dets = [d for d in dets if d['number'] == most_common_number]
        best_det = max(same_number_dets, key=lambda x: x['confidence'])
        
        best_det['vote_count'] = vote_count
        best_det['total_votes'] = len(dets)
        best_det['vote_percentage'] = vote_percentage
        
        consensus_detections.append(best_det)
        logging.info(f"✓ Number {most_common_number} at ({best_det['center'][0]}, {best_det['center'][1]}) "
                    f"- {vote_count}/{len(dets)} votes ({vote_percentage:.0f}%)")
    
    logging.info(f"Consensus voting: {len(detections)} → {len(consensus_detections)} detections")
    return consensus_detections


def filter_by_size(detections, config):
    """Remove detections that are too small (dots) or too large"""
    size_cfg = config['number_detection']['size_filter']
    min_w, min_h = size_cfg['min_width'], size_cfg['min_height']
    max_w, max_h = size_cfg['max_width'], size_cfg['max_height']
    
    filtered = []
    for det in detections:
        x, y, w, h = det['bbox']
        if min_w <= w <= max_w and min_h <= h <= max_h:
            filtered.append(det)
        else:
            logging.debug(f"Filtered out: {det['number']} ({w}x{h})")
    
    logging.info(f"Size filtering: {len(detections)} → {len(filtered)}")
    return filtered


def ensure_unique_numbers(detections, expected_range):
    """Ensure each number appears only once"""
    if not detections:
        return []
    
    number_groups = {}
    for det in detections:
        num = det['number']
        if num not in number_groups:
            number_groups[num] = []
        number_groups[num].append(det)
    
    unique_detections = []
    
    for num, dets in number_groups.items():
        if len(dets) == 1:
            unique_detections.append(dets[0])
        else:
            best = max(dets, key=lambda x: (x.get('vote_percentage', 0), x['confidence']))
            unique_detections.append(best)
            logging.info(f"Number {num}: kept best of {len(dets)} detections")
    
    unique_detections = sorted(unique_detections, key=lambda x: x['number'])
    logging.info(f"Unique filtering: {len(detections)} → {len(unique_detections)}")
    
    # Check for missing numbers
    if expected_range:
        detected_nums = set(d['number'] for d in unique_detections)
        expected_nums = set(range(expected_range[0], expected_range[1] + 1))
        missing_nums = expected_nums - detected_nums
        
        if missing_nums:
            logging.warning(f"Missing numbers: {sorted(missing_nums)}")
    
    return unique_detections


def validate_number_range(detections, expected_range):
    """Filter detections to expected range"""
    if expected_range is None:
        return detections
    
    min_num, max_num = expected_range
    filtered = [d for d in detections if min_num <= d['number'] <= max_num]
    
    logging.info(f"Range filtering ({min_num}-{max_num}): {len(detections)} → {len(filtered)}")
    return filtered


def detect_missing_numbers_relaxed(gray_no_dots, missing_numbers, config):
    """
    Second pass with relaxed settings for missing numbers
    Lower confidence threshold (35 instead of 55)
    """
    if not missing_numbers:
        return []
    
    logging.info(f"Second pass: searching for {missing_numbers}")
    
    relaxed_thresholds = config['number_detection']['relaxed_thresholds']
    relaxed_conf = config['number_detection']['relaxed_confidence']
    all_detections = []
    
    for thresh in relaxed_thresholds:
        preprocessed, scale_factor = preprocess_for_ocr(gray_no_dots, thresh)
        
        config_str = '--psm 11 --oem 3 -c tessedit_char_whitelist=0123456789'
        data = pytesseract.image_to_data(Image.fromarray(preprocessed), config=config_str, output_type=pytesseract.Output.DICT)
        
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf = int(data['conf'][i])
            
            if text and text.isdigit() and int(text) in missing_numbers and conf > relaxed_conf:
                x = data['left'][i] // scale_factor
                y = data['top'][i] // scale_factor
                w = data['width'][i] // scale_factor
                h = data['height'][i] // scale_factor
                
                all_detections.append({
                    'number': int(text),
                    'center': (x + w // 2, y + h // 2),
                    'bbox': (x, y, w, h),
                    'confidence': conf,
                    'method': 'relaxed_pass',
                    'threshold': thresh
                })
    
    logging.info(f"Second pass found {len(all_detections)} candidates")
    return all_detections


def visualize_detected_numbers(image, detections, save_path):
    """Draw detected numbers on image"""
    result_image = image.copy()
    if len(result_image.shape) == 2:
        result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)
    
    for det in detections:
        cx, cy = det['center']
        x, y, w, h = det['bbox']
        number = det['number']
        vote_pct = det.get('vote_percentage', 0)
        
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.circle(result_image, (cx, cy), 3, (0, 0, 255), -1)
        
        label = f"{number} ({vote_pct:.0f}%)"
        cv2.putText(result_image, label, (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imwrite(str(save_path), result_image)
    logging.info(f"Visualization saved: {save_path}")


def process_segment(image_path, config, detected_circles_json, expected_range, reader):
    """Process one segment with full detection pipeline"""

    segment_name = image_path.stem
    logging.info(f"\n{'='*60}")
    logging.info(f"Processing: {segment_name}")
    logging.info(f"{'='*60}")
    
    image = cv2.imread(str(image_path))
    if image is None:
        logging.error(f"Failed to load: {segment_name}")
        return None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    circles = load_detected_circles_for_segment(segment_name, detected_circles_json)
    
    if circles:
        logging.info(f"Erasing {len(circles)} dots...")
        gray_no_dots = erase_dots_from_image(gray, circles, config)
        save_debug_image(gray_no_dots, f"{segment_name}_no_dots.jpg", config)
    else:
        logging.warning("No circles found, processing without dot removal")
        gray_no_dots = gray
    
    logging.info("PHASE 1: Multi-threshold detection...")
    all_detections = detect_numbers_multi_threshold(gray_no_dots, segment_name, config, reader)
    
    if not all_detections:
        logging.warning(f"No numbers found in {segment_name}")
        return None
    
    logging.info("PHASE 2: Consensus voting...")
    consensus_detections = consensus_voting(all_detections, config)
    
    logging.info("PHASE 3: Size filtering...")
    consensus_detections = filter_by_size(consensus_detections, config)
    
    logging.info("PHASE 4: Unique numbers...")
    consensus_detections = ensure_unique_numbers(consensus_detections, expected_range)
    
    # Phase 4.5: Relaxed pass for missing numbers
    if expected_range:
        detected_nums = set(d['number'] for d in consensus_detections)
        expected_nums = set(range(expected_range[0], expected_range[1] + 1))
        missing_nums = sorted(expected_nums - detected_nums)
        
        if missing_nums:
            logging.info(f"PHASE 4.5: Recovering {len(missing_nums)} missing numbers...")
            relaxed_detections = detect_missing_numbers_relaxed(gray_no_dots, missing_nums, config)
            
            size_cfg = config['number_detection']['size_filter']
            for det in relaxed_detections:
                x, y, w, h = det['bbox']
                if (size_cfg['min_width'] <= w <= size_cfg['max_width'] and
                    size_cfg['min_height'] <= h <= size_cfg['max_height']):
                    
                    # Check not duplicate position
                    is_dup = False
                    for existing in consensus_detections:
                        ex, ey = existing['center']
                        cx, cy = det['center']
                        dist = np.sqrt((cx - ex)**2 + (cy - ey)**2)
                        if dist < 15:
                            is_dup = True
                            break
                    
                    if not is_dup:
                        det['vote_percentage'] = 100
                        consensus_detections.append(det)
                        logging.info(f"  ✓ Recovered: {det['number']}")
    
    if expected_range:
        logging.info(f"PHASE 5: Range validation {expected_range}...")
        consensus_detections = validate_number_range(consensus_detections, expected_range)
        
        detected_nums = set(d['number'] for d in consensus_detections)
        expected_nums = set(range(expected_range[0], expected_range[1] + 1))
        still_missing = sorted(expected_nums - detected_nums)
        if still_missing:
            logging.warning(f"  Still missing: {still_missing}")
    
    # Create segment data
    segment_data = {
        "segment_name": segment_name,
        "numbers": [
            {
                "number": det['number'],
                "pixel_x": det['center'][0],
                "pixel_y": det['center'][1],
                "bbox": det['bbox'],
                "confidence": det['confidence'],
                "vote_percentage": det.get('vote_percentage', 100)
            }
            for det in consensus_detections
        ],
        "total_numbers": len(consensus_detections)
    }
    
    viz_dir = config['_base_path'] / config['paths']['number_viz_dir']
    viz_dir.mkdir(parents=True, exist_ok=True)
    viz_path = viz_dir / f"{segment_name}_numbers.jpg"
    visualize_detected_numbers(image, consensus_detections, viz_path)
    
    logging.info(f"{segment_name}: {len(consensus_detections)} numbers detected\n")
    
    return segment_data


def convert_to_global(detected_json, segments_json, output_json, config):
    """Convert to global coordinates with deduplicatiom"""

    with open(detected_json) as f:
        detected_data = json.load(f)
    with open(segments_json) as f:
        segment_mapping = json.load(f)
       
    all_global_numbers = []
    
    for segment in detected_data["segments"]:
        segment_name = segment["segment_name"] + ".jpg"
        
        if segment_name not in segment_mapping:
            logging.warning(f"Segment {segment_name} not in mapping")
            continue
        
        offset = segment_mapping[segment_name]["start"]
        
        for num_data in segment["numbers"]:
            all_global_numbers.append({
                "number": num_data["number"],
                "segment_name": segment_name,
                "local_coordinates": {
                    "x": num_data["pixel_x"],
                    "y": num_data["pixel_y"]
                },
                "global_coordinates": {
                    "x": num_data["pixel_x"] + offset["x"],
                    "y": num_data["pixel_y"] + offset["y"]
                },
                "bbox": num_data["bbox"],
                "confidence": num_data["confidence"],
                "vote_percentage": num_data.get("vote_percentage", 100)
            })
    
    logging.info(f"Total before deduplication: {len(all_global_numbers)}")
    
    # Remove duplicates (keep best vote percentage)
    threshold = config['number_detection']['duplicate_threshold']
    sorted_numbers = sorted(all_global_numbers, key=lambda x: (x.get('vote_percentage', 0), x['confidence']), reverse=True)
    
    unique = []
    
    for num in sorted_numbers:
        global_x = num["global_coordinates"]["x"]
        global_y = num["global_coordinates"]["y"]
        number_value = num["number"]
        
        is_duplicate = False
        duplicate_idx = -1
        
        for i, existing in enumerate(unique):
            ex_x = existing["global_coordinates"]["x"]
            ex_y = existing["global_coordinates"]["y"]
            
            distance = np.sqrt((global_x - ex_x)**2 + (global_y - ex_y)**2)
            
            if distance < threshold:
                if number_value == existing["number"]:
                    is_duplicate = True
                    duplicate_idx = i
                    break
                elif distance < 10:
                    is_duplicate = True
                    duplicate_idx = i
                    logging.info(f"Removing close false positive: {number_value} vs {existing['number']}")
                    break
        
        if is_duplicate:
            if num.get('vote_percentage', 0) > unique[duplicate_idx].get('vote_percentage', 0):
                unique[duplicate_idx] = num
        else:
            unique.append(num)
    
    logging.info(f"After deduplication: {len(unique)}")
    
    with open(output_json, 'w') as f:
        json.dump({
            "detection_method": "Multi-Threshold OCR with Consensus Voting",
            "total_numbers": len(unique),
            "numbers": unique
        }, f, indent=2)
    
    logging.info(f"Saved to {output_json}")
    return unique


def run_detection_for_all_segments(config, picture_name, expected_range, use_combo_ocr):
    """Main entry point for number detection"""

    base_path = config['_base_path']
    segments_dir = base_path / config['paths']['segments_overlap_dir']
    config_dir = base_path / config['paths']['config_dir']
    
    reader = setup_ocr(config) if use_combo_ocr else None
    
    # Check for detected circles
    detected_circles_json = config_dir / config['filenames']['detected_circles']
    if not detected_circles_json.exists():
        logging.error(f"Detected circles JSON not found: {detected_circles_json}")
        logging.error("Please run circle detection first!")
        return
    
    jpg_files = sorted(segments_dir.glob("*.jpg"))
    
    logging.info(f"BATCH PROCESSING: {len(jpg_files)} segments")
    if expected_range:
        logging.info(f"Expected range: {expected_range[0]}-{expected_range[1]}\n")
    
    # Process segments
    all_segments_data = []
    for i, image_file in enumerate(jpg_files, 1):
        logging.info(f"[{i}/{len(jpg_files)}]")
        segment_data = process_segment(image_file, config, detected_circles_json, expected_range, reader)
        all_segments_data.append(segment_data)
    
    # Save segment results
    detected_numbers_json = config_dir / config['filenames']['detected_numbers']
    
    with open(detected_numbers_json, 'w') as f:
        json.dump({
            "detection_method": "Multi-Threshold OCR with Consensus Voting",
            "total_segments": len([s for s in all_segments_data if s]),
            "total_numbers_found": sum(s["total_numbers"] for s in all_segments_data if s),
            "segments": [s for s in all_segments_data if s]
        }, f, indent=2)
    
    logging.info(f"Saved segment results to {detected_numbers_json}\n")
    
    # Convert to global
    segments_json = segments_dir / config['filenames']['overlap_segments_meta']
    global_numbers_json = config_dir / config['filenames']['global_numbers']
    
    if not segments_json.exists():
        logging.error(f"Segments mapping not found: {segments_json}")
        return
    
    global_numbers = convert_to_global(detected_numbers_json, segments_json, global_numbers_json, config)
    
    # Visualize on main image
    picture_path = base_path / config['paths']['pictures_dir'] / picture_name
    
    if picture_path.exists():
        main_image = cv2.imread(str(picture_path))
        
        for num in global_numbers:
            x = num["global_coordinates"]["x"]
            y = num["global_coordinates"]["y"]
            number = num["number"]
            
            cv2.circle(main_image, (x, y), 5, (0, 0, 255), -1)
            
            label = f"{number}"
            cv2.putText(main_image, label, (x + 8, y - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        output_path = base_path / config['filenames']['main_with_numbers']
        cv2.imwrite(str(output_path), main_image)
        logging.info(f"Main image visualization saved to {output_path}")
    else:
        logging.warning(f"Main image not found: {picture_path}")
    
    logging.info("Number detection completed\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)