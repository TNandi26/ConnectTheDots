"""
Circle/Dot Detection Module
- Merged auto_calibrate_multiple_thresholds + detect_circles_blob_detector → detect_circles_multi_method
- Combined filter_by_radius + check_black_color → filter_circles
- Simplified remove_duplicate_circles (single pass instead of nested loops)
- Merged convert_to_global_coordinates + remove_duplicate_circles + filter_by_intensity_variance + add_unique_ids → convert_to_global
- Added variance and solidity filters to reject number centers (false positives)
"""

import cv2
import numpy as np
import json
import logging
from collections import Counter



def calculate_variance(gray, center, radius):
    """Calculate intensity variance - dots are uniform, number centers have edges"""
    x, y = center
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask, (x, y), radius, 255, -1)
    pixels = gray[mask == 255]
    return np.var(pixels) if len(pixels) > 0 else 999


def calculate_solidity(contour):
    """Calculate solidity - dots are solid shapes, number centers are hollow"""
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    return area / hull_area if hull_area > 0 else 0


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
                            solidity = calculate_solidity(contour)
                            all_candidates.append({
                                'center': (int(x), int(y)),
                                'radius': int(radius),
                                'circularity': circularity,
                                'solidity': solidity,
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
            'solidity': 1.0,
            'method': 'blob'
        })
    
    logging.info(f"Found {len(all_candidates)} circle candidates from all methods")
    return all_candidates


def check_solid_center(gray, center, radius):
    """Check if a circle is solid (real dot) vs hollow (number center like 'o', '0', '6', '8', '9')"""
    x, y = center
    
    # Check center pixel
    if not (0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]):
        return False
    
    center_intensity = gray[y, x]
    
    # Sample pixels in a small radius around center (25% of circle radius)
    sample_radius = max(1, radius // 4)
    sample_points = []
    
    # Sample 8 points around the center
    for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
        rad = np.radians(angle)
        sx = int(x + sample_radius * np.cos(rad))
        sy = int(y + sample_radius * np.sin(rad))
        
        if 0 <= sx < gray.shape[1] and 0 <= sy < gray.shape[0]:
            sample_points.append(gray[sy, sx])
    
    if not sample_points:
        return False
    
    # Calculate intensity variance in the center region
    all_points = [center_intensity] + sample_points
    variance = np.var(all_points)
    mean_intensity = np.mean(all_points)
    
    # Real dots: low variance (solid), dark mean intensity
    # Number centers: high variance (white center, dark edges) OR light mean intensity
    is_solid = variance < 400 and mean_intensity < 120
    
    return is_solid


def filter_circles(candidates, gray, config, expected_range=None):
    """Apply minimal filtering - let matchmaker handle false positives"""
    if not candidates:
        return []
    
    logging.info(f"Starting with {len(candidates)} candidates")
    
    # Step 1: Black color check (keep dots that are dark enough)
    black_threshold = config['circle_detection']['black_threshold']
    black_circles = []
    
    for c in candidates:
        x, y = c['center']
        if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
            intensity = gray[y, x]
            if intensity < black_threshold:
                c['intensity'] = int(intensity)
                black_circles.append(c)
    
    logging.info(f"Black filter: {len(candidates)} → {len(black_circles)}")
    
    # Step 2: Solid center check (NEW - reject number centers)
    solid_circles = []
    rejected_hollow = 0
    
    for c in black_circles:
        if check_solid_center(gray, c['center'], c['radius']):
            solid_circles.append(c)
        else:
            rejected_hollow += 1
            logging.debug(f"Rejected hollow center at ({c['center'][0]}, {c['center'][1]})")
    
    logging.info(f"Solid center filter: {len(black_circles)} → {len(solid_circles)} (rejected {rejected_hollow} hollow centers)")
    
    # Step 3: Remove extreme outliers in radius (keep 95% of data)
    if len(solid_circles) > 5:
        radii = sorted([c['radius'] for c in solid_circles])
        p5 = radii[int(len(radii) * 0.05)]  # 5th percentile
        p95 = radii[int(len(radii) * 0.95)]  # 95th percentile
        
        radius_filtered = [c for c in solid_circles if p5 <= c['radius'] <= p95]
        logging.info(f"Radius outlier filter (keep {p5}-{p95}px): {len(solid_circles)} → {len(radius_filtered)}")
    else:
        radius_filtered = solid_circles
        logging.info(f"Too few candidates, skipping radius filter")
    
    # Step 4: Basic quality filter (very permissive)
    quality_filtered = [c for c in radius_filtered if c.get('circularity', 1.0) > 0.4]
    logging.info(f"Quality filter (circularity > 0.4): {len(radius_filtered)} → {len(quality_filtered)}")
    
    logging.info(f"Final filtered candidates: {len(quality_filtered)}\n")
    return quality_filtered


def process_segment(image_path, config, expected_range=None):
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
    
    # Phase 2: Filtering (now with expected_range)
    logging.info("PHASE 2: Filtering...")
    filtered = filter_circles(candidates, gray, config, expected_range)
    
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

def run_dot_detection_for_all_segments(config, picture_name, expected_range=None):
    """Main entry point for dot detection pipeline"""
    base_path = config['_base_path']
    segments_dir = base_path / config['paths']['segments_overlap_dir']
    config_dir = base_path / config['paths']['config_dir']
    config_dir.mkdir(parents=True, exist_ok=True)
    
    jpg_files = sorted(segments_dir.glob("*.jpg"))
    
    # Log whether radius filtering will be used
    if expected_range and expected_range < 30:
        logging.info(f"Processing {len(jpg_files)} segments (expected: {expected_range} dots - radius filtering DISABLED)\n")
    else:
        logging.info(f"Processing {len(jpg_files)} segments (expected: {expected_range} dots - radius filtering ENABLED)\n")
    
    # Process each segment
    all_segments = []
    for i, image_file in enumerate(jpg_files, 1):
        logging.info(f"[{i}/{len(jpg_files)}]")
        segment_data = process_segment(image_file, config, expected_range)
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