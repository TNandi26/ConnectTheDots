import cv2
import numpy as np
import os
import logging
import json
import pathlib
from collections import Counter
import pytesseract
from PIL import Image

# Set Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Optional: EasyOCR for combo detection
try:
    import easyocr
    EASYOCR_AVAILABLE = True
    reader = easyocr.Reader(['en'], gpu=False)
except ImportError:
    EASYOCR_AVAILABLE = False
    logging.warning("EasyOCR not available. Install with: pip install easyocr")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_transformed_image(image_path):
    if not os.path.exists(image_path):
        logging.error(f"Input image not found: {image_path}")
        return None
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Could not load image from {image_path}")
        return None
    logging.info(f"Loaded image from {image_path}")
    return image


def save_debug_image(image, filename):
    """Save debug image to output folder"""
    base_path = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_path, "Output_pictures", "number_debug", filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)


def load_detected_circles_for_segment(segment_name, detected_circles_json):
    """Load circle coordinates for this specific segment"""
    try:
        with open(detected_circles_json, 'r') as f:
            data = json.load(f)
        
        # Find this segment's circles
        for segment in data.get('segments', []):
            if segment['segment_name'] == segment_name:
                circles = []
                for circle in segment.get('circles', []):
                    circles.append((circle['pixel_x'], circle['pixel_y'], circle['radius']))
                return circles
        
        return []
    except Exception as e:
        logging.warning(f"Could not load circles for {segment_name}: {e}")
        return []


def erase_dots_from_image(image, circles, margin=3):
    """
    Erase dots from image by filling them with white
    margin: extra pixels around dot to erase
    """
    if not circles:
        return image
    
    result = image.copy()
    
    for x, y, r in circles:
        # Create a mask for this circle
        cv2.circle(result, (x, y), r + margin, 255, -1)  # Fill with white
    
    logging.info(f"Erased {len(circles)} dots from image")
    return result


def preprocess_for_ocr(gray_image, threshold_value, upscale=True):
    """Preprocess image for OCR with specific threshold"""
    # Threshold to binary (black text on white background)
    _, binary = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Remove small noise but keep text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Upscale image for better OCR (small text detection)
    scale_factor = 1
    if upscale:
        scale_factor = 2
        width = int(binary.shape[1] * scale_factor)
        height = int(binary.shape[0] * scale_factor)
        binary = cv2.resize(binary, (width, height), interpolation=cv2.INTER_CUBIC)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
    
    return denoised, scale_factor


def detect_numbers_easyocr(image, scale_factor=1):
    """Detect numbers using EasyOCR"""
    if not EASYOCR_AVAILABLE:
        return []
    
    try:
        results = reader.readtext(image, allowlist='0123456789', detail=1)
        
        detections = []
        for bbox, text, conf in results:
            if text.isdigit():
                # bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                
                x = int(min(x_coords)) // scale_factor
                y = int(min(y_coords)) // scale_factor
                w = int(max(x_coords) - min(x_coords)) // scale_factor
                h = int(max(y_coords) - min(y_coords)) // scale_factor
                
                center_x = x + w // 2
                center_y = y + h // 2
                
                detections.append({
                    'number': int(text),
                    'center': (center_x, center_y),
                    'bbox': (x, y, w, h),
                    'confidence': int(conf * 100),
                    'method': 'easyocr'
                })
        
        return detections
    except Exception as e:
        logging.error(f"EasyOCR detection failed: {e}")
        return []


def detect_numbers_tesseract(image, scale_factor=1, config='--psm 11 --oem 3 -c tessedit_char_whitelist=0123456789'):
    """
    Detect numbers using Tesseract OCR
    """
    try:
        pil_image = Image.fromarray(image)
        data = pytesseract.image_to_data(pil_image, config=config, output_type=pytesseract.Output.DICT)
        
        detections = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf = int(data['conf'][i])
            
            # Filter: must be a number with good confidence to avoid false positives
            if text and text.isdigit() and conf > 55:  # Stricter threshold
                x = data['left'][i] // scale_factor
                y = data['top'][i] // scale_factor
                w = data['width'][i] // scale_factor
                h = data['height'][i] // scale_factor
                
                center_x = x + w // 2
                center_y = y + h // 2
                
                detections.append({
                    'number': int(text),
                    'center': (center_x, center_y),
                    'bbox': (x, y, w, h),
                    'confidence': conf,
                    'method': 'tesseract'
                })
        
        return detections
    except Exception as e:
        logging.error(f"Tesseract detection failed: {e}")
        return []
    """Detect numbers using EasyOCR"""
    if not EASYOCR_AVAILABLE:
        return []
    
    try:
        results = reader.readtext(image, allowlist='0123456789', detail=1)
        
        detections = []
        for bbox, text, conf in results:
            if text.isdigit():
                # bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                
                x = int(min(x_coords)) // scale_factor
                y = int(min(y_coords)) // scale_factor
                w = int(max(x_coords) - min(x_coords)) // scale_factor
                h = int(max(y_coords) - min(y_coords)) // scale_factor
                
                center_x = x + w // 2
                center_y = y + h // 2
                
                detections.append({
                    'number': int(text),
                    'center': (center_x, center_y),
                    'bbox': (x, y, w, h),
                    'confidence': int(conf * 100),
                    'method': 'easyocr'
                })
        
        return detections
    except Exception as e:
        logging.error(f"EasyOCR detection failed: {e}")
        return []
    """
    Detect numbers using Tesseract OCR
    """
    try:
        pil_image = Image.fromarray(image)
        data = pytesseract.image_to_data(pil_image, config=config, output_type=pytesseract.Output.DICT)
        
        detections = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf = int(data['conf'][i])
            
            # Filter: must be a number with good confidence
            if text and text.isdigit() and conf > 50:  # Increased confidence threshold
                x = data['left'][i] // scale_factor
                y = data['top'][i] // scale_factor
                w = data['width'][i] // scale_factor
                h = data['height'][i] // scale_factor
                
                center_x = x + w // 2
                center_y = y + h // 2
                
                detections.append({
                    'number': int(text),
                    'center': (center_x, center_y),
                    'bbox': (x, y, w, h),
                    'confidence': conf
                })
        
        return detections
    except Exception as e:
        logging.error(f"Tesseract detection failed: {e}")
        return []


def detect_numbers_multi_threshold(gray_image, segment_name, use_combo=True):
    """
    Try multiple thresholds with both Tesseract and EasyOCR
    Using similar thresholds as successful dot detection
    """
    mean_val = gray_image.mean()
    std_val = gray_image.std()
    logging.info(f"Image stats: mean={mean_val:.1f}, std={std_val:.1f}")
    
    # Using thresholds similar to successful dot detection
    test_thresholds = [10, 20, 30, 40, 50, 80, 100, 120, 150, 180]
    all_detections = []
    
    for thresh in test_thresholds:
        preprocessed, scale_factor = preprocess_for_ocr(gray_image, thresh, upscale=True)
        
        # Debug images disabled to save space
        # save_debug_image(preprocessed, f"{segment_name}_thresh_{thresh}.jpg")
        
        # Method 1: Tesseract with PSM 11 (sparse text)
        config = '--psm 11 --oem 3 -c tessedit_char_whitelist=0123456789'
        tesseract_detections = detect_numbers_tesseract(preprocessed, scale_factor, config)
        
        for det in tesseract_detections:
            det['threshold'] = thresh
            det['psm'] = 11
        
        all_detections.extend(tesseract_detections)
        
        # Try PSM 6 as well (uniform block) for better coverage
        config_psm6 = '--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789'
        tesseract_detections_psm6 = detect_numbers_tesseract(preprocessed, scale_factor, config_psm6)
        
        for det in tesseract_detections_psm6:
            det['threshold'] = thresh
            det['psm'] = 6
        
        all_detections.extend(tesseract_detections_psm6)
        
        # Method 2: EasyOCR (if available and enabled)
        if use_combo and EASYOCR_AVAILABLE:
            easyocr_detections = detect_numbers_easyocr(preprocessed, scale_factor)
            
            for det in easyocr_detections:
                det['threshold'] = thresh
            
            all_detections.extend(easyocr_detections)
            logging.info(f"Threshold {thresh}: Tesseract={len(tesseract_detections)+len(tesseract_detections_psm6)}, EasyOCR={len(easyocr_detections)}")
        else:
            logging.info(f"Threshold {thresh}: {len(tesseract_detections)+len(tesseract_detections_psm6)} numbers detected")
    
    logging.info(f"Total detections from all methods: {len(all_detections)}")
    return all_detections


def consensus_voting(detections, position_tolerance=15, min_votes=3, min_percentage=50):
    """
    Use voting to decide which number is at each position
    Stricter requirements to reduce false positives
    """
    if not detections:
        return []
    
    # Group detections by position
    position_groups = []
    
    for det in detections:
        cx, cy = det['center']
        
        # Find if this position already exists
        found_group = False
        for group in position_groups:
            # Check if close to any detection in this group
            group_center = group['center']
            dist = np.sqrt((cx - group_center[0])**2 + (cy - group_center[1])**2)
            
            if dist < position_tolerance:
                group['detections'].append(det)
                found_group = True
                break
        
        if not found_group:
            # Create new group
            position_groups.append({
                'center': (cx, cy),
                'detections': [det]
            })
    
    # For each position group, vote on the number
    consensus_detections = []
    
    for group in position_groups:
        dets = group['detections']
        
        # Need at least min_votes detections at this position
        if len(dets) < min_votes:
            logging.debug(f"Skipped position - only {len(dets)} vote(s)")
            continue
        
        # Count votes for each number at this position
        number_votes = Counter([d['number'] for d in dets])
        most_common_number, vote_count = number_votes.most_common(1)[0]
        
        # Calculate vote percentage
        vote_percentage = (vote_count / len(dets)) * 100
        
        # Require reasonable agreement (40% or more)
        if vote_percentage < min_percentage:
            logging.debug(f"Skipped - low consensus ({vote_percentage:.0f}%). Votes: {dict(number_votes)}")
            continue
        
        # Get the detection with highest confidence for this number
        same_number_dets = [d for d in dets if d['number'] == most_common_number]
        best_det = max(same_number_dets, key=lambda x: x['confidence'])
        
        # Add vote info
        best_det['vote_count'] = vote_count
        best_det['total_votes'] = len(dets)
        best_det['vote_percentage'] = vote_percentage
        
        consensus_detections.append(best_det)
        logging.info(f"✓ Number {most_common_number} at ({best_det['center'][0]}, {best_det['center'][1]}) "
                    f"- {vote_count}/{len(dets)} votes ({vote_percentage:.0f}%)")
    
    logging.info(f"Consensus voting: {len(detections)} -> {len(consensus_detections)} detections")
    return consensus_detections


def ensure_unique_numbers(detections, expected_range=None):
    """
    Ensure each number appears only once in the final results
    Keep the detection with highest vote percentage, then confidence
    """
    if not detections:
        return []
    
    # Group by number value
    number_groups = {}
    for det in detections:
        num = det['number']
        if num not in number_groups:
            number_groups[num] = []
        number_groups[num].append(det)
    
    unique_detections = []
    
    for num, dets in number_groups.items():
        if len(dets) == 1:
            # Only one detection, keep it
            unique_detections.append(dets[0])
        else:
            # Multiple detections of same number - keep the best one
            # Sort by vote percentage, then confidence
            best = max(dets, key=lambda x: (x.get('vote_percentage', 0), x['confidence']))
            unique_detections.append(best)
            logging.info(f"Number {num}: kept best of {len(dets)} detections (vote={best.get('vote_percentage', 0):.0f}%, conf={best['confidence']})")
    
    # Sort by number for readability
    unique_detections = sorted(unique_detections, key=lambda x: x['number'])
    
    logging.info(f"Unique number filtering: {len(detections)} -> {len(unique_detections)} detections")
    
    # Check for missing numbers in sequence
    if expected_range:
        detected_nums = set(d['number'] for d in unique_detections)
        expected_nums = set(range(expected_range[0], expected_range[1] + 1))
        missing_nums = expected_nums - detected_nums
        
        if missing_nums:
            logging.warning(f"Missing numbers in this segment: {sorted(missing_nums)}")
    
    return unique_detections
    """Remove detections that are too small (likely dots)"""
    filtered = []
    
    for det in detections:
        x, y, w, h = det['bbox']
        
        if w >= min_width and h >= min_height:
            filtered.append(det)
        else:
            logging.debug(f"Filtered out small detection: {det['number']} ({w}x{h})")
    
    logging.info(f"Dot filtering: {len(detections)} -> {len(filtered)} detections")
    return filtered


def validate_number_range(detections, expected_range=None):
    """Filter detections to expected range"""
    if expected_range is None:
        return detections
    
    min_num, max_num = expected_range
    filtered = [d for d in detections if min_num <= d['number'] <= max_num]
    
    logging.info(f"Range filtering ({min_num}-{max_num}): {len(detections)} -> {len(filtered)}")
    return filtered


# ============================================================================
# VISUALIZATION
# ============================================================================

def filter_out_dots(detections, min_width=8, min_height=6, max_width=40, max_height=30):
    """
    Remove detections that are too small (likely dots) or too large (likely false detections)
    """
    filtered = []
    
    for det in detections:
        x, y, w, h = det['bbox']
        
        # Check size constraints
        if min_width <= w <= max_width and min_height <= h <= max_height:
            filtered.append(det)
        else:
            logging.debug(f"Filtered out detection: {det['number']} ({w}x{h}) - size out of range")
    
    logging.info(f"Size filtering: {len(detections)} -> {len(filtered)} detections")
    return filtered


def validate_number_range(detections, expected_range=None):
    """Filter detections to expected range"""
    if expected_range is None:
        return detections
    
    min_num, max_num = expected_range
    filtered = [d for d in detections if min_num <= d['number'] <= max_num]
    
    logging.info(f"Range filtering ({min_num}-{max_num}): {len(detections)} -> {len(filtered)}")
    return filtered


def visualize_detected_numbers(image, detections, save_path):
    """Draw detected numbers on the image"""
    result_image = image.copy()
    if len(result_image.shape) == 2:
        result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)
    
    for det in detections:
        cx, cy = det['center']
        x, y, w, h = det['bbox']
        number = det['number']
        conf = det['confidence']
        vote_pct = det.get('vote_percentage', 0)
        
        # Draw bounding box in blue
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Draw center point in red
        cv2.circle(result_image, (cx, cy), 3, (0, 0, 255), -1)
        
        # Put number text with vote percentage
        label = f"{number} ({vote_pct:.0f}%)"
        cv2.putText(result_image, label, (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imwrite(save_path, result_image)
    logging.info(f"Number visualization saved: {save_path}")


# ============================================================================
# SEGMENT PROCESSING
# ============================================================================

def detect_missing_numbers_relaxed(gray_no_dots, segment_name, missing_numbers):
    """
    Second pass with relaxed settings to find specific missing numbers
    Only searches for numbers we know should exist
    """
    if not missing_numbers:
        return []
    
    logging.info(f"Second pass: searching for missing numbers {missing_numbers}")
    
    # Relaxed thresholds for second pass
    test_thresholds = [20, 40, 60, 100, 140, 180]
    all_detections = []
    
    for thresh in test_thresholds:
        preprocessed, scale_factor = preprocess_for_ocr(gray_no_dots, thresh, upscale=True)
        
        # Lower confidence for second pass
        config = '--psm 11 --oem 3 -c tessedit_char_whitelist=0123456789'
        data = pytesseract.image_to_data(Image.fromarray(preprocessed), config=config, 
                                         output_type=pytesseract.Output.DICT)
        
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf = int(data['conf'][i])
            
            # Only accept if it's a missing number and has reasonable confidence
            if text and text.isdigit() and int(text) in missing_numbers and conf > 35:
                x = data['left'][i] // scale_factor
                y = data['top'][i] // scale_factor
                w = data['width'][i] // scale_factor
                h = data['height'][i] // scale_factor
                
                center_x = x + w // 2
                center_y = y + h // 2
                
                all_detections.append({
                    'number': int(text),
                    'center': (center_x, center_y),
                    'bbox': (x, y, w, h),
                    'confidence': conf,
                    'method': 'relaxed_pass',
                    'threshold': thresh
                })
    
    logging.info(f"Second pass found {len(all_detections)} candidates for missing numbers")
    return all_detections


def process_single_segment(image_path, output_base_path, viz_dir, detected_circles_json, expected_range=None, use_combo=True):
    """Process one segment and return detected numbers"""
    segment_name = pathlib.Path(image_path).stem
    logging.info(f"\n{'='*60}")
    logging.info(f"Processing: {segment_name}")
    logging.info(f"{'='*60}")
    
    # Load image
    image = load_transformed_image(image_path)
    if image is None:
        logging.error(f"Failed to load: {segment_name}")
        return None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    
    # Load detected circles for this segment
    circles = load_detected_circles_for_segment(segment_name, detected_circles_json)
    
    # Erase dots from image
    if circles:
        logging.info(f"Erasing {len(circles)} dots from image...")
        gray_no_dots = erase_dots_from_image(gray, circles, margin=2)  # Reduced margin from 3 to 2
        save_debug_image(gray_no_dots, f"{segment_name}_no_dots.jpg")
    else:
        logging.warning("No circles found for this segment, processing without dot removal")
        gray_no_dots = gray
    
    # Phase 1: Detect numbers with multiple thresholds
    logging.info("PHASE 1: Multi-threshold number detection...")
    all_detections = detect_numbers_multi_threshold(gray_no_dots, segment_name, use_combo=use_combo)
    
    if not all_detections:
        logging.warning(f"No numbers found in {segment_name}")
        return None
    
    # Phase 2: Consensus voting (stricter to eliminate false positives)
    logging.info("PHASE 2: Consensus voting...")
    consensus_detections = consensus_voting(all_detections, position_tolerance=15, min_votes=3, min_percentage=50)
    
    # Phase 3: Filter out dots and oversized detections
    logging.info("PHASE 3: Filtering by size...")
    consensus_detections = filter_out_dots(consensus_detections, min_width=8, min_height=6, max_width=40, max_height=30)
    
    # Phase 4: Ensure each number appears only once
    logging.info("PHASE 4: Ensuring unique numbers...")
    consensus_detections = ensure_unique_numbers(consensus_detections, expected_range)
    
    # Phase 4.5: Second pass for missing numbers (if expected range provided)
    if expected_range:
        detected_nums = set(d['number'] for d in consensus_detections)
        expected_nums = set(range(expected_range[0], expected_range[1] + 1))
        missing_nums = sorted(expected_nums - detected_nums)
        
        if missing_nums:
            logging.info(f"PHASE 4.5: Attempting to recover {len(missing_nums)} missing numbers...")
            relaxed_detections = detect_missing_numbers_relaxed(gray_no_dots, segment_name, missing_nums)
            
            # Filter and add only valid ones
            for det in relaxed_detections:
                x, y, w, h = det['bbox']
                if 8 <= w <= 40 and 6 <= h <= 30:
                    # Check if not a duplicate position
                    is_dup = False
                    for existing in consensus_detections:
                        ex, ey = existing['center']
                        cx, cy = det['center']
                        dist = np.sqrt((cx - ex)**2 + (cy - ey)**2)
                        if dist < 15:
                            is_dup = True
                            break
                    
                    if not is_dup:
                        det['vote_percentage'] = 100  # Mark as second pass
                        consensus_detections.append(det)
                        logging.info(f"  ✓ Recovered missing number: {det['number']}")
    
    # Phase 5: Validate range if provided
    if expected_range:
        logging.info(f"PHASE 5: Validating range {expected_range}...")
        consensus_detections = validate_number_range(consensus_detections, expected_range)
        
        # Final missing check
        detected_nums = set(d['number'] for d in consensus_detections)
        expected_nums = set(range(expected_range[0], expected_range[1] + 1))
        still_missing = sorted(expected_nums - detected_nums)
        if still_missing:
            logging.warning(f"  Still missing after recovery: {still_missing}")
    
    # Create segment data
    segment_data = {
        "segment_name": segment_name,
        "segment_path": str(image_path),
        "image_dimensions": {"width": image.shape[1], "height": image.shape[0]},
        "numbers": [],
        "total_numbers": len(consensus_detections)
    }
    
    # Add number data
    for det in consensus_detections:
        cx, cy = det['center']
        segment_data["numbers"].append({
            "number": det['number'],
            "pixel_x": cx,
            "pixel_y": cy,
            "bbox": det['bbox'],
            "confidence": det['confidence'],
            "vote_percentage": det.get('vote_percentage', 100)
        })
    
    # Save visualization
    viz_path = os.path.join(viz_dir, f"{segment_name}_numbers.jpg")
    visualize_detected_numbers(image, consensus_detections, viz_path)
    
    logging.info(f"✓ {segment_name}: {len(consensus_detections)} numbers detected\n")
    
    return segment_data


# ============================================================================
# COORDINATE CONVERSION
# ============================================================================

def load_segment_mapping(segments_json_path):
    """Load the segment mapping JSON"""
    with open(segments_json_path, 'r') as f:
        return json.load(f)


def convert_to_global_coordinates(detected_numbers_json_path, segments_json_path, 
                                  output_json_path="global_numbers.json"):
    """Convert segment-local coordinates to global coordinates"""
    with open(detected_numbers_json_path, 'r') as f:
        detected_data = json.load(f)
    
    segment_mapping = load_segment_mapping(segments_json_path)
    
    logging.info("\n" + "="*60)
    logging.info("CONVERTING NUMBERS TO GLOBAL COORDINATES")
    logging.info("="*60)
    
    all_global_numbers = []
    
    for segment in detected_data["segments"]:
        segment_name = segment["segment_name"] + ".jpg"
        
        if segment_name not in segment_mapping:
            logging.warning(f"Segment {segment_name} not found in mapping, skipping")
            continue
        
        offset_x = segment_mapping[segment_name]["start"]["x"]
        offset_y = segment_mapping[segment_name]["start"]["y"]
        
        logging.info(f"Processing {segment_name}: offset ({offset_x}, {offset_y})")
        
        for num_data in segment["numbers"]:
            local_x = num_data["pixel_x"]
            local_y = num_data["pixel_y"]
            
            global_x = local_x + offset_x
            global_y = local_y + offset_y
            
            global_number = {
                "number": num_data["number"],
                "segment_name": segment_name,
                "local_coordinates": {"x": local_x, "y": local_y},
                "global_coordinates": {"x": global_x, "y": global_y},
                "bbox": num_data["bbox"],
                "confidence": num_data["confidence"],
                "vote_percentage": num_data.get("vote_percentage", 100)
            }
            all_global_numbers.append(global_number)
    
    logging.info(f"Total numbers before deduplication: {len(all_global_numbers)}")
    
    # Remove duplicates from overlapping segments
    unique_numbers = remove_global_duplicates(all_global_numbers)
    
    logging.info(f"Total numbers after deduplication: {len(unique_numbers)}")
    
    # Save to JSON
    output_data = {
        "detection_method": "Multi-Threshold OCR with Consensus Voting",
        "coordinate_space": "global (main image)",
        "total_numbers": len(unique_numbers),
        "numbers": unique_numbers
    }
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    full_output_path = os.path.join(base_path, "Output_pictures", output_json_path)
    
    with open(full_output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logging.info(f"Global coordinates saved: {full_output_path}")
    logging.info("="*60 + "\n")
    
    return unique_numbers


def remove_global_duplicates(numbers, distance_threshold=20):
    """
    Remove duplicate numbers from overlapping segments
    Keep the one with highest vote percentage, then confidence
    """
    if not numbers:
        return []
    
    # Sort by vote percentage, then confidence
    sorted_numbers = sorted(numbers, key=lambda x: (x.get('vote_percentage', 0), x['confidence']), reverse=True)
    
    unique = []
    
    for num in sorted_numbers:
        global_x = num["global_coordinates"]["x"]
        global_y = num["global_coordinates"]["y"]
        number_value = num["number"]
        
        is_duplicate = False
        duplicate_index = -1
        
        for i, existing in enumerate(unique):
            ex_x = existing["global_coordinates"]["x"]
            ex_y = existing["global_coordinates"]["y"]
            
            distance = np.sqrt((global_x - ex_x)**2 + (global_y - ex_y)**2)
            
            # Same number within distance threshold OR different numbers very close together (false positive)
            if distance < distance_threshold:
                if number_value == existing["number"]:
                    # Duplicate of same number
                    is_duplicate = True
                    duplicate_index = i
                    break
                elif distance < 10:
                    # Different numbers too close together - likely false positive, keep better one
                    is_duplicate = True
                    duplicate_index = i
                    logging.info(f"Removing close false positive: {number_value} vs {existing['number']} at distance {distance:.1f}px")
                    break
        
        if is_duplicate:
            # Keep the one with higher vote percentage
            if num.get('vote_percentage', 0) > unique[duplicate_index].get('vote_percentage', 0):
                unique[duplicate_index] = num
                logging.debug(f"Replaced duplicate {number_value}")
        else:
            unique.append(num)
    
    logging.info(f"Removed {len(numbers) - len(unique)} duplicate numbers from overlaps")
    return unique


def visualize_on_main_image(main_image_path, global_numbers, output_path="main_image_with_numbers.jpg"):
    """Draw all detected numbers on the main image"""
    logging.info("\n" + "="*60)
    logging.info("VISUALIZING NUMBERS ON MAIN IMAGE")
    logging.info("="*60)
    
    main_image = cv2.imread(main_image_path)
    if main_image is None:
        logging.error(f"Failed to load main image: {main_image_path}")
        return
    
    logging.info(f"Main image dimensions: {main_image.shape[1]}x{main_image.shape[0]}")
    
    for num in global_numbers:
        x = num["global_coordinates"]["x"]
        y = num["global_coordinates"]["y"]
        number = num["number"]
        vote_pct = num.get("vote_percentage", 100)
        
        # Draw center point in red
        cv2.circle(main_image, (x, y), 5, (0, 0, 255), -1)
        
        # Draw number text
        label = f"{number}"
        cv2.putText(main_image, label, (x + 8, y - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    full_output_path = os.path.join(base_path, "Output_pictures", output_path)
    cv2.imwrite(full_output_path, main_image)
    
    logging.info(f"Main image visualization saved: {full_output_path}")
    logging.info(f"Total numbers drawn: {len(global_numbers)}")
    logging.info("="*60 + "\n")


def save_final_json(all_segments_data, output_path, filename="detected_numbers.json"):
    """Save all segment data to ONE JSON file"""
    total_numbers = sum(seg["total_numbers"] for seg in all_segments_data if seg is not None)
    
    output_data = {
        "detection_method": "Multi-Threshold OCR with Consensus Voting",
        "total_segments": len([s for s in all_segments_data if s is not None]),
        "total_numbers_found": total_numbers,
        "segments": [s for s in all_segments_data if s is not None]
    }
    
    full_output_path = os.path.join(output_path, filename)
    with open(full_output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logging.info(f"\n{'='*60}")
    logging.info(f"FINAL JSON SAVED: {full_output_path}")
    logging.info(f"Total segments: {output_data['total_segments']}")
    logging.info(f"Total numbers: {total_numbers}")
    logging.info(f"{'='*60}\n")


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def run_detection_for_all_segments(expected_range=None, use_combo_ocr=True):
    """
    Process all segments in the folder
    use_combo_ocr: if True, uses both Tesseract + EasyOCR for better accuracy
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    segments_path = os.path.join(base_path, "Segments/SegmentsOverlap")
    output_path = os.path.join(base_path, "Output_pictures")
    viz_dir = os.path.join(output_path, "number_visualizations")
    detected_circles_json = os.path.join(output_path, "detected_circles.json")
    
    os.makedirs(viz_dir, exist_ok=True)
    
    # Check if circles are detected
    if not os.path.exists(detected_circles_json):
        logging.error(f"Detected circles JSON not found: {detected_circles_json}")
        logging.error("Please run circle detection first!")
        return
    
    folder = pathlib.Path(segments_path)
    jpg_files = sorted(list(folder.glob("*.jpg")))
    
    logging.info(f"\n{'='*60}")
    logging.info(f"BATCH PROCESSING: {len(jpg_files)} segments")
    if expected_range:
        logging.info(f"Expected number range: {expected_range[0]}-{expected_range[1]}")
    if use_combo_ocr and EASYOCR_AVAILABLE:
        logging.info("Using COMBO: Tesseract + EasyOCR")
    elif use_combo_ocr and not EASYOCR_AVAILABLE:
        logging.warning("EasyOCR not available, using Tesseract only")
    else:
        logging.info("Using Tesseract only")
    logging.info(f"{'='*60}\n")
    
    all_segments_data = []
    for i, image_file in enumerate(jpg_files, 1):
        logging.info(f"[{i}/{len(jpg_files)}]")
        segment_data = process_single_segment(image_file, output_path, viz_dir, 
                                             detected_circles_json, expected_range, use_combo_ocr)
        all_segments_data.append(segment_data)
    
    detected_numbers_json = os.path.join(output_path, "detected_numbers.json")
    save_final_json(all_segments_data, output_path, "detected_numbers.json")
    
    logging.info("\n✓✓✓ SEGMENT PROCESSING COMPLETE ✓✓✓\n")
    
    segments_json_path = os.path.join(segments_path, "segments.json")
    
    if os.path.exists(segments_json_path):
        global_numbers = convert_to_global_coordinates(
            detected_numbers_json,
            segments_json_path,
            "global_numbers.json"
        )
        
        main_image_path = os.path.join(base_path, "Pictures/lo.jpg")
        if os.path.exists(main_image_path):
            visualize_on_main_image(main_image_path, global_numbers, "main_image_with_numbers.jpg")
        else:
            logging.warning(f"Main image not found at: {main_image_path}")
    else:
        logging.warning(f"Segments mapping not found at: {segments_json_path}")
    
    logging.info("\n✓✓✓ ALL NUMBER DETECTION COMPLETE ✓✓✓\n")


if __name__ == "__main__":
    # Run with combo OCR (Tesseract + EasyOCR) for best accuracy
    run_detection_for_all_segments(expected_range=(1, 100), use_combo_ocr=True)
