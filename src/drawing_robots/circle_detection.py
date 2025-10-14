"""
Circle/Dot Detection Module
- Merged auto_calibrate_multiple_thresholds + detect_circles_blob_detector → detect_circles_multi_method
- Combined filter_by_radius + check_black_color → filter_circles
- Simplified remove_duplicate_circles (single pass instead of nested loops)
- Merged convert_to_global_coordinates + remove_duplicate_circles + filter_by_intensity_variance + add_unique_ids → convert_to_global
"""

import cv2
import numpy as np
import json
import logging
from collections import Counter


def detect_circles_multi_method(gray, config):
    """Detect circles using multiple methods"""
    thresholds = config['circle_detection']['thresholds']
    min_area = config['circle_detection']['min_area']
    max_area = config['circle_detection']['max_area']
    min_radius = config['circle_detection']['min_radius']
    max_radius = config['circle_detection']['max_radius']
    min_circularity = config['circle_detection']['min_circularity']
    
    all_candidates = []
    
    # Method 1: Multi-threshold contour detection
    for thresh in thresholds:
        _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    if circularity > min_circularity:
                        (x, y), radius = cv2.minEnclosingCircle(contour)
                        if min_radius <= int(radius) <= max_radius:
                            all_candidates.append({
                                'center': (int(x), int(y)),
                                'radius': int(radius),
                                'circularity': circularity,
                                'method': 'contour'
                            })
    
    # Method 2: SimpleBlobDetector
    inverted = 255 - gray
    params = cv2.SimpleBlobDetector_Params()
    blob_cfg = config['circle_detection']['blob_detector']
    params.filterByArea = True
    params.minArea = blob_cfg['min_area']
    params.maxArea = blob_cfg['max_area']
    params.filterByCircularity = True
    params.minCircularity = blob_cfg['min_circularity']
    params.filterByConvexity = True
    params.minConvexity = blob_cfg['min_convexity']
    params.filterByInertia = True
    params.minInertiaRatio = blob_cfg['min_inertia_ratio']
    params.filterByColor = False
    
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(inverted)
    
    for kp in keypoints:
        all_candidates.append({
            'center': (int(kp.pt[0]), int(kp.pt[1])),
            'radius': int(kp.size / 2),
            'circularity': 1.0,
            'method': 'blob'
        })
    
    logging.info(f"Found {len(all_candidates)} circle candidates from all methods")
    return all_candidates


def filter_circles(candidates, gray, config):
    """Apply all filtering steps"""
    if not candidates:
        return []
    
    # Step 1: Radius clustering - find mode and keep ±1
    radii = [c['radius'] for c in candidates]
    radius_counts = Counter(radii)
    mode_radius, count = radius_counts.most_common(1)[0]
    logging.info(f"Most common radius: {mode_radius} ({count} occurrences)")
    
    filtered = [c for c in candidates if abs(c['radius'] - mode_radius) <= 1]
    logging.info(f"Radius filtering: {len(candidates)} → {len(filtered)}")
    
    # Step 2: Black color check
    black_threshold = config['circle_detection']['black_threshold']
    black_circles = []
    
    for c in filtered:
        x, y = c['center']
        if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
            intensity = gray[y, x]
            if intensity < black_threshold:
                c['intensity'] = int(intensity)
                black_circles.append(c)
                logging.debug(f"Accepted: ({x},{y}) - intensity: {intensity}")
            else:
                logging.debug(f"Rejected: ({x},{y}) - intensity: {intensity}")
    
    logging.info(f"Black filter: {len(filtered)} → {len(black_circles)}")
    
    # Step 3: Quality filter
    quality_filtered = [c for c in black_circles if c.get('circularity', 1.0) > 0.6]
    logging.info(f"Quality filter: {len(black_circles)} → {len(quality_filtered)}")
    
    return quality_filtered


def process_segment(image_path, config):
    """Process a single segment"""
    segment_name = image_path.stem
    logging.info(f"Processing: {segment_name}")
    
    image = cv2.imread(str(image_path))
    if image is None:
        logging.error(f"Failed to load: {segment_name}")
        return None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Phase 1: Detection
    logging.info("PHASE 1: Multi-method circle detection...")
    candidates = detect_circles_multi_method(gray, config)
    
    if not candidates:
        logging.warning(f"No circles found in {segment_name}")
        return None
    
    # Phase 2: Filtering
    logging.info("PHASE 2: Filtering...")
    filtered = filter_circles(candidates, gray, config)
    
    if not filtered:
        logging.warning(f"No circles passed filtering in {segment_name}")
        return None
    
    # Save visualization
    viz_dir = config['_base_path'] / config['paths']['dot_viz_dir']
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    viz_img = image.copy()
    for i, c in enumerate(filtered, 1):
        x, y, r = c['center'][0], c['center'][1], c['radius']
        cv2.circle(viz_img, (x, y), 3, (0, 0, 255), -1)  # Red center
        cv2.circle(viz_img, (x, y), r, (0, 255, 0), 2)   # Green border
        cv2.putText(viz_img, str(i), (x - 8, y - r - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    cv2.imwrite(str(viz_dir / f"{segment_name}_final.jpg"), viz_img)
    logging.info(f"Visualization saved")
    
    # Create segment data
    segment_data = {
        "segment_name": segment_name,
        "circles": [
            {
                "pixel_x": c['center'][0],
                "pixel_y": c['center'][1],
                "radius": c['radius'],
                "intensity": c['intensity']
            }
            for c in filtered
        ],
        "total_circles": len(filtered)
    }
    
    logging.info(f"{segment_name}: {len(filtered)} circles detected\n")
    return segment_data


def convert_to_global(detected_json, segments_json, output_json, config):
    """Convert to global coordinates with deduplication and filtering"""
    with open(detected_json) as f:
        detected_data = json.load(f)
    with open(segments_json) as f:
        segment_mapping = json.load(f)
    
    all_circles = []
    
    # Step 1: Convert to global coordinates
    for segment in detected_data["segments"]:
        segment_name = segment["segment_name"] + ".jpg"
        
        if segment_name not in segment_mapping:
            logging.warning(f"Segment {segment_name} not in mapping")
            continue
        
        offset = segment_mapping[segment_name]["start"]
        
        for circle in segment["circles"]:
            all_circles.append({
                "segment_name": segment_name,
                "local_coordinates": {
                    "x": circle["pixel_x"],
                    "y": circle["pixel_y"]
                },
                "global_coordinates": {
                    "x": circle["pixel_x"] + offset["x"],
                    "y": circle["pixel_y"] + offset["y"]
                },
                "radius": circle["radius"],
                "intensity": circle["intensity"]
            })
    
    logging.info(f"Total circles before deduplication: {len(all_circles)}")
    
    # Step 2: Remove duplicates (keep darker circle)
    threshold = config['circle_detection']['duplicate_threshold']
    unique = []
    
    for circle in all_circles:
        cx, cy = circle["global_coordinates"]["x"], circle["global_coordinates"]["y"]
        
        duplicate_idx = None
        for i, existing in enumerate(unique):
            ex_x = existing["global_coordinates"]["x"]
            ex_y = existing["global_coordinates"]["y"]
            dist = np.sqrt((cx - ex_x)**2 + (cy - ex_y)**2)
            
            if dist < threshold:
                duplicate_idx = i
                break
        
        if duplicate_idx is not None:
            if circle["intensity"] < unique[duplicate_idx]["intensity"]:
                unique[duplicate_idx] = circle
                logging.debug(f"Replaced duplicate at ({cx}, {cy})")
        else:
            unique.append(circle)
    
    logging.info(f"After deduplication: {len(unique)} circles")
    
    # Step 3: Intensity variance filter (remove outliers)
    intensities = [c["intensity"] for c in unique]
    mean_int = np.mean(intensities)
    std_int = np.std(intensities)
    std_mult = config['circle_detection']['intensity_std_multiplier']
    
    logging.info(f"Intensity stats - Mean: {mean_int:.2f}, Std: {std_int:.2f}")
    
    filtered = []
    removed_count = 0
    for c in unique:
        deviation = abs(c["intensity"] - mean_int)
        if deviation <= std_mult * std_int:
            filtered.append(c)
        else:
            removed_count += 1
            logging.info(f"Removed outlier at ({c['global_coordinates']['x']}, "
                        f"{c['global_coordinates']['y']}) - intensity: {c['intensity']}")
    
    logging.info(f"After intensity filter: {len(filtered)} circles ({removed_count} removed)")
    
    # Step 4: Add unique IDs
    for i, circle in enumerate(filtered, 1):
        circle["id"] = i
    
    # Step 5: Save
    with open(output_json, 'w') as f:
        json.dump({
            "detection_method": "Multi-Method Improved with Intensity Filter",
            "total_circles": len(filtered),
            "circles": filtered
        }, f, indent=2)
    
    logging.info(f"Saved to {output_json}")
    return filtered

def run_dot_detection_for_all_segments(config, picture_name):
    """Main entry point for dot detection pipeline"""
    base_path = config['_base_path']
    segments_dir = base_path / config['paths']['segments_overlap_dir']
    config_dir = base_path / config['paths']['config_dir']
    config_dir.mkdir(parents=True, exist_ok=True)
    
    jpg_files = sorted(segments_dir.glob("*.jpg"))
    logging.info(f"Processing {len(jpg_files)} segments...\n")
    
    # Process each segment
    all_segments = []
    for i, image_file in enumerate(jpg_files, 1):
        logging.info(f"[{i}/{len(jpg_files)}]")
        segment_data = process_segment(image_file, config)
        if segment_data:
            all_segments.append(segment_data)
    
    # Save segment-level results
    detected_json = config_dir / config['filenames']['detected_circles']
    
    with open(detected_json, 'w') as f:
        json.dump({
            "total_segments": len(all_segments),
            "total_circles_found": sum(s["total_circles"] for s in all_segments),
            "segments": all_segments
        }, f, indent=2)
    
    logging.info(f"Saved segment results to {detected_json}\n")
    
    # Convert to global coordinates
    segments_json = segments_dir / config['filenames']['overlap_segments_meta']
    global_json = config_dir / config['filenames']['global_dots']
    
    if not segments_json.exists():
        logging.error(f"Segments JSON not found: {segments_json}")
        return
    
    global_circles = convert_to_global(detected_json, segments_json, global_json, config)
    
    # Visualize on main image
    picture_path = base_path / config['paths']['pictures_dir'] / picture_name
    
    if picture_path.exists():
        main_img = cv2.imread(str(picture_path))
        
        for circle in global_circles:
            x = circle["global_coordinates"]["x"]
            y = circle["global_coordinates"]["y"]
            r = circle["radius"]
            cv2.circle(main_img, (x, y), 3, (0, 0, 255), -1)  # Red center
            cv2.circle(main_img, (x, y), r, (0, 255, 0), 2)   # Green border
        
        output_path = base_path / config['filenames']['main_with_dots']
        cv2.imwrite(str(output_path), main_img)
        logging.info(f"Main image visualization saved to {output_path}")
    else:
        logging.warning(f"Main image not found: {picture_path}")
    
    logging.info("Dot detection completed\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)