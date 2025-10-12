import cv2
import numpy as np
import os
import logging
import json
import pathlib
from collections import Counter

def auto_calibrate_multiple_thresholds(gray_image):
    """Test multiple thresholds and collect candidates from each"""

    test_thresholds = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    all_candidates = []
    threshold_stats = []
    
    for thresh in test_thresholds:
        _, binary = cv2.threshold(gray_image, thresh, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 15 <= area <= 900:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.5:
                        (x, y), radius = cv2.minEnclosingCircle(contour)
                        center = (int(x), int(y))
                        radius = int(radius)
                        if 3 <= radius <= 25:
                            candidate = {
                                'center': center,
                                'radius': radius,
                                'area': area,
                                'circularity': circularity,
                                'threshold': thresh
                            }
                            candidates.append(candidate)
        
        all_candidates.extend(candidates)
        threshold_stats.append((thresh, len(contours), len(candidates)))
        logging.info(f"Threshold {thresh}: {len(contours)} contours, {len(candidates)} circle candidates")
    
    logging.info(f"Total candidates from all thresholds: {len(all_candidates)}")
    return all_candidates, threshold_stats


def detect_circles_blob_detector(image):
    """SimpleBlobDetector optimized for filled black circles"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    inverted = 255 - gray
    
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 20
    params.maxArea = 800
    params.filterByCircularity = True
    params.minCircularity = 0.6
    params.filterByConvexity = True
    params.minConvexity = 0.7
    params.filterByInertia = True
    params.minInertiaRatio = 0.4
    params.filterByColor = False
    
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(inverted)
    
    candidates = []
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        radius = int(kp.size / 2)
        confidence = kp.response
        candidate = {
            'center': (x, y),
            'radius': radius,
            'area': np.pi * radius * radius,
            'circularity': 1.0,
            'confidence': confidence,
            'method': 'blob'
        }
        candidates.append(candidate)
    
    logging.info(f"SimpleBlobDetector found {len(candidates)} circles")
    return candidates


def cluster_radius_groups(candidates, n_groups=2):
    """Use clustering to find multiple radius groups"""
    if len(candidates) < 3:
        return candidates
    
    radii = [c['radius'] for c in candidates]
    radius_counts = Counter(radii)
    most_common = radius_counts.most_common(n_groups)
    
    if len(most_common) == 1:
        target_radius = most_common[0][0]
        tolerance = max(2, target_radius // 4)
        filtered = [c for c in candidates if abs(c['radius'] - target_radius) <= tolerance]
    else:
        filtered = []
        for target_radius, count in most_common:
            if count >= 2:
                tolerance = max(2, target_radius // 4)
                group_circles = [c for c in candidates if abs(c['radius'] - target_radius) <= tolerance]
                filtered.extend(group_circles)
        
        seen_positions = set()
        unique_filtered = []
        for c in filtered:
            pos = c['center']
            if pos not in seen_positions:
                seen_positions.add(pos)
                unique_filtered.append(c)
        filtered = unique_filtered
    
    logging.info(f"Radius clustering: {len(candidates)} -> {len(filtered)} circles")
    return filtered


def detect_circles_combined_improved(gray):
    """Improved combined detection with multiple methods"""
    all_candidates = []
    
    logging.info("Running multi-threshold contour detection...")
    contour_candidates, threshold_stats = auto_calibrate_multiple_thresholds(gray)
    
    for c in contour_candidates:
        c['method'] = 'contour'
    all_candidates.extend(contour_candidates)
    
    logging.info("Running SimpleBlobDetector...")
    blob_candidates = detect_circles_blob_detector(gray)
    all_candidates.extend(blob_candidates)
    
    logging.info(f"Total candidates from all methods: {len(all_candidates)}")
    
    if not all_candidates:
        return [], None
    
    filtered_candidates = cluster_radius_groups(all_candidates)
    
    quality_filtered = []
    for c in filtered_candidates:
        if c.get('circularity', 0.8) > 0.6:
            quality_filtered.append(c)
    
    logging.info(f"After quality filtering: {len(quality_filtered)} circles")
    
    detected_circles = [(c['center'][0], c['center'][1], c['radius']) for c in quality_filtered]
    
    if threshold_stats:
        best_thresh = max(threshold_stats, key=lambda x: x[2])[0]
        _, debug_binary = cv2.threshold(gray, best_thresh, 255, cv2.THRESH_BINARY_INV)
    else:
        debug_binary = None
    
    return detected_circles, debug_binary


# ============================================================================
# FILTERING FUNCTIONS
# ============================================================================

def filter_by_radius(circles):
    """Filter circles by most common radius ±1"""
    if not circles:
        return []
    
    radii = [r for _, _, r in circles]
    radius_counts = Counter(radii)
    mode_radius, count = radius_counts.most_common(1)[0]
    logging.info(f"Most common radius: {mode_radius} (occurrences: {count}x)")
    
    filtered = []
    for x, y, r in circles:
        if mode_radius - 1 <= r <= mode_radius + 1:
            filtered.append((x, y, r))
    
    logging.info(f"Radius filtering: {len(circles)} -> {len(filtered)} circles")
    return filtered


def check_black_color(circles, gray_image, black_threshold=120):
    """Filter circles that are on black pixels"""
    black_circles = []
    rejected_circles = []
    
    for x, y, r in circles:
        if 0 <= x < gray_image.shape[1] and 0 <= y < gray_image.shape[0]:
            center_intensity = gray_image[y, x]
            
            if center_intensity < black_threshold:
                black_circles.append((x, y, r, int(center_intensity)))
                logging.info(f"ACCEPTED: ({x},{y}) - intensity: {center_intensity}")
            else:
                rejected_circles.append((x, y, r, int(center_intensity)))
                logging.info(f"REJECTED: ({x},{y}) - intensity: {center_intensity}")
        else:
            rejected_circles.append((x, y, r, -1))
            logging.info(f"REJECTED: ({x},{y}) - outside bounds")
    
    logging.info(f"Black filter: {len(circles)} -> {len(black_circles)} accepted, {len(rejected_circles)} rejected")
    return black_circles, rejected_circles


def filter_by_intensity_variance(circles, std_multiplier=2.0):
    """
    Filter out circles whose intensity differs significantly from the average.
    
    Args:
        circles: List of circles with intensity data
        std_multiplier: Number of standard deviations to use as threshold (default: 2.0)
    
    Returns:
        Filtered list of circles
    """
    if not circles:
        return []
    
    intensities = [circle.get("intensity", 0) for circle in circles]
    
    mean_intensity = np.mean(intensities)
    std_intensity = np.std(intensities)
    
    logging.info(f"Intensity stats - Mean: {mean_intensity:.2f}, Std: {std_intensity:.2f}")
    
    # Filter circles
    filtered_circles = []
    removed_count = 0
    
    for circle in circles:
        intensity = circle.get("intensity", 0)
        deviation = abs(intensity - mean_intensity)
        
        if deviation <= std_multiplier * std_intensity:
            filtered_circles.append(circle)
        else:
            removed_count += 1
            logging.info(f"Removed circle at ({circle.get('global_coordinates', {}).get('x', 'N/A')}, "
                        f"{circle.get('global_coordinates', {}).get('y', 'N/A')}) - "
                        f"intensity: {intensity} (deviation: {deviation:.2f})")
    
    logging.info(f"Intensity filtering: {len(circles)} -> {len(filtered_circles)} circles "
                f"({removed_count} removed)")
    
    return filtered_circles


def add_unique_ids(circles):
    for i, circle in enumerate(circles, start=1):
        circle["id"] = i
    return circles


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_final_circles(image, circles, save_path):
    """Draw final detected circles"""
    result_image = image.copy()
    if len(result_image.shape) == 2:
        result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)
    
    for i, circle_data in enumerate(circles):
        x, y, r = circle_data[0], circle_data[1], circle_data[2]
        cv2.circle(result_image, (x, y), 3, (0, 0, 255), -1)  # Red center
        cv2.circle(result_image, (x, y), r, (0, 255, 0), 2)   # Green border
        cv2.putText(result_image, str(i + 1), (x - 8, y - r - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    cv2.imwrite(save_path, result_image)
    logging.info(f"Final visualization saved: {save_path}")


# ============================================================================
# SEGMENT PROCESSING
# ============================================================================

def process_single_segment(image_path, viz_dir):
    """Process one segment and return its data"""
    segment_name = pathlib.Path(image_path).stem
    logging.info(f"\n{'='*60}")
    logging.info(f"Processing: {segment_name}")
    logging.info(f"{'='*60}")
    
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Failed to load: {segment_name}")
        return None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    
    logging.info("PHASE 1: Circle detection...")
    detected_circles, binary_debug = detect_circles_combined_improved(gray)
    
    if not detected_circles:
        logging.warning(f"No circles found in {segment_name}")
        return None
    
    logging.info(f"Phase 1: {len(detected_circles)} circles detected")
    
    logging.info("PHASE 2: Radius filtering...")
    filtered_circles = filter_by_radius(detected_circles)
    
    logging.info("PHASE 3: Black color check...")
    black_circles, rejected_circles = check_black_color(filtered_circles, gray)
    
    segment_data = {
        "segment_name": segment_name,
        "segment_path": str(image_path),
        "image_dimensions": {"width": image.shape[1], "height": image.shape[0]},
        "circles": [],
        "total_circles": len(black_circles)
    }
    
    # Add circle data
    for i, (x, y, r, intensity) in enumerate(black_circles):
        segment_data["circles"].append({
            "local_id": i + 1,
            "pixel_x": x,
            "pixel_y": y,
            "radius": r,
            "intensity": intensity
        })
    
    viz_path = os.path.join(viz_dir, f"{segment_name}_final.jpg")
    visualize_final_circles(image, black_circles, viz_path)
    
    logging.info(f"{segment_name}: {len(black_circles)} circles detected\n")
    
    return segment_data

# ============================================================================
# COORDINATE CONVERSION AND VISUALIZATION ON MAIN IMAGE
# ============================================================================

def load_segment_mapping(segments_json_path):
    """Load the segment mapping JSON that defines where each segment belongs on the main image"""
    with open(segments_json_path, 'r') as f:
        return json.load(f)


def convert_to_global_coordinates(detected_circles_json_path, segments_json_path, output_json_path):
    """
    Convert segment-local coordinates to global coordinates on the main image
    Also removes duplicates from overlapping segments
    """
    with open(detected_circles_json_path, 'r') as f:
        detected_data = json.load(f)
    
    segment_mapping = load_segment_mapping(segments_json_path)
    all_global_circles = []
    
    for segment in detected_data["segments"]:
        segment_name = segment["segment_name"] + ".jpg"
        
        # Get the offset for this segment from the mapping
        if segment_name not in segment_mapping:
            logging.warning(f"Segment {segment_name} not found in mapping, skipping")
            continue
        
        offset_x = segment_mapping[segment_name]["start"]["x"]
        offset_y = segment_mapping[segment_name]["start"]["y"]
        
        logging.info(f"Processing {segment_name}: offset ({offset_x}, {offset_y})")
        
        # Convert each circle's coordinates
        for circle in segment["circles"]:
            local_x = circle["pixel_x"]
            local_y = circle["pixel_y"]
            global_x = local_x + offset_x
            global_y = local_y + offset_y
            
            global_circle = {
                "segment_name": segment_name,
                "local_coordinates": {"x": local_x, "y": local_y},
                "global_coordinates": {"x": global_x, "y": global_y},
                "radius": circle["radius"],
                "intensity": circle["intensity"]
            }
            all_global_circles.append(global_circle)
    
    logging.info(f"Total circles before deduplication: {len(all_global_circles)}")
    
    unique_circles = remove_duplicate_circles(all_global_circles)
    
    logging.info(f"Total circles after deduplication: {len(unique_circles)}")
    
    logging.info("Filtering by intensity variance...")
    filtered_circles = filter_by_intensity_variance(unique_circles, std_multiplier=2.0)
    
    filtered_circles = add_unique_ids(filtered_circles)
    
    output_data = {
        "detection_method": "Multi-Method Improved with Intensity Filter",
        "coordinate_space": "global (main image)",
        "total_circles": len(filtered_circles),
        "circles": filtered_circles
    }
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    full_output_path = os.path.join(base_path, "Output_pictures/config", output_json_path)
    
    with open(full_output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logging.info(f"Global coordinates saved: {full_output_path}")
    
    return filtered_circles


def remove_duplicate_circles(circles, distance_threshold=15):
    """
    Remove duplicate circles detected in overlapping segments
    Keeps the circle with higher intensity (darker = lower value = better)
    """
    if not circles:
        return []
    
    unique = []
    
    for circle in circles:
        global_x = circle["global_coordinates"]["x"]
        global_y = circle["global_coordinates"]["y"]
        
        # Check if this circle is too close to any existing unique circle
        is_duplicate = False
        duplicate_index = -1
        
        for i, existing in enumerate(unique):
            ex_x = existing["global_coordinates"]["x"]
            ex_y = existing["global_coordinates"]["y"]
            
            distance = np.sqrt((global_x - ex_x)**2 + (global_y - ex_y)**2)
            
            if distance < distance_threshold:
                is_duplicate = True
                duplicate_index = i
                break
        
        if is_duplicate:
            # Keep the one with lower intensity (darker/blacker)
            if circle["intensity"] < unique[duplicate_index]["intensity"]:
                unique[duplicate_index] = circle
                logging.debug(f"Replaced duplicate at ({global_x}, {global_y})")
        else:
            unique.append(circle)
    
    logging.info(f"Removed {len(circles) - len(unique)} duplicate circles from overlaps")
    return unique


def visualize_on_main_image(main_image_path, global_circles, output_path="main_image_with_dots.jpg"):
    """
    Draw all detected circles on the main image using global coordinates
    """    
    main_image = cv2.imread(main_image_path)
    if main_image is None:
        logging.error(f"Failed to load main image: {main_image_path}")
        return
        
    # Draw circles
    for i, circle in enumerate(global_circles):
        x = circle["global_coordinates"]["x"]
        y = circle["global_coordinates"]["y"]
        r = circle["radius"]
        
        # Draw red center point
        cv2.circle(main_image, (x, y), 3, (0, 0, 255), -1)
        # Draw green circle border
        cv2.circle(main_image, (x, y), r, (0, 255, 0), 2)

    full_output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Output_pictures", output_path)
    cv2.imwrite(full_output_path, main_image)
    
    logging.info(f"Main image visualization saved: {full_output_path}")
    logging.info(f"Total circles drawn: {len(global_circles)}")


def save_final_json(all_segments_data, output_path, filename):
    """Save all segment data to ONE JSON file with unique IDs"""
    total_circles = sum(seg["total_circles"] for seg in all_segments_data if seg is not None)
    
    global_id = 1
    for segment in all_segments_data:
        if segment is not None:
            for circle in segment["circles"]:
                circle["global_id"] = global_id
                global_id += 1
    
    output_data = {
        "total_segments": len([s for s in all_segments_data if s is not None]),
        "total_circles_found": total_circles,
        "segments": [s for s in all_segments_data if s is not None]
    }
    
    full_output_path = os.path.join(output_path, filename)
    with open(full_output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logging.info(f"Final json saved to {full_output_path}")
    logging.info(f"Total segments: {output_data['total_segments']}")
    logging.info(f"Total circles: {total_circles}")


def run_dot_detection_for_all_segments(picture_name):
    """Process all segments in the folder"""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_path, "Output_pictures")
    segments_path = os.path.join(output_path, "Segments/SegmentsOverlap")
    viz_dir = os.path.join(output_path, "Segments/Segment_dot_visualizations")
        
    folder = pathlib.Path(segments_path)
    jpg_files = sorted(list(folder.glob("*.jpg")))
    
    logging.info(f"Batch processing: {len(jpg_files)} segments")
    
    all_segments_data = []
    for i, image_file in enumerate(jpg_files, 1):
        logging.info(f"[{i}/{len(jpg_files)}]")
        segment_data = process_single_segment(image_file, viz_dir)
        all_segments_data.append(segment_data)
    
    save_final_json(all_segments_data, os.path.join(output_path, "config"), "detected_circles.json")
    
    logging.info("\nSegment process is completed\n")
    
    # Convert to global coordinates and visualize on main image
    segments_json_path = os.path.join(segments_path, "overlap_segments.json")
    
    if os.path.exists(segments_json_path):
        logging.info("Converting to global coordinates")
        
        global_circles = convert_to_global_coordinates(
            os.path.join(os.path.join(output_path, "config"), "detected_circles.json"),
            segments_json_path,
            "global_dot_coordinates.json"
        )
        
        main_image_path = os.path.join(base_path, f"../../Pictures/{picture_name}")
        if os.path.exists(main_image_path):
            visualize_on_main_image(main_image_path, global_circles, "main_image_with_dots.jpg")
        else:
            logging.warning(f"Main image not found at: {main_image_path}")
    else:
        logging.warning(f"Segments mapping not found at: {segments_json_path}")
    
    logging.info("All process has been completed\n")


if __name__ == "__main__":
    run_dot_detection_for_all_segments()