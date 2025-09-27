import cv2
import numpy as np
import os
import logging
import json
from typing import List, Tuple
from collections import Counter
import matplotlib.pyplot as plt


def load_transformed_image(input_folder="Output_pictures", step_name="perspective_adaptive_gray", step_number=13):
    """Load the transformed image from the previous processing script"""
    base_path = os.path.dirname(os.path.abspath(__file__))
    filename = "step_114_perspective_adaptive_gray.jpg"
    image_path = os.path.join(base_path, input_folder, filename)
    
    if not os.path.exists(image_path):
        logging.error(f"Input image not found: {image_path}")
        return None
        
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Could not load image from {image_path}")
        return None
    
    logging.info(f"Loaded transformed image: {image.shape} from {filename}")
    return image


def auto_calibrate_multiple_thresholds(gray_image):
    """
    Test multiple thresholds and collect candidates from each
    """
    # Calculate image statistics
    mean_val = gray_image.mean()
    std_val = gray_image.std()
    
    logging.info(f"Image stats: mean={mean_val:.1f}, std={std_val:.1f}")
    
    # Test multiple thresholds
    test_thresholds = [50, 80, 100, 120, 150, 180]
    all_candidates = []
    threshold_stats = []
    
    for thresh in test_thresholds:
        _, binary = cv2.threshold(gray_image, thresh, 255, cv2.THRESH_BINARY_INV)
        
        # Gentler morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Save debug binary for this threshold
        save_debug_image(binary, f"debug_threshold_{thresh}.jpg")
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Save contour visualization
        contour_viz = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contour_viz, contours, -1, (0, 255, 0), 1)
        save_debug_image(contour_viz, f"debug_contours_{thresh}.jpg")
        
        # Collect circle candidates from this threshold
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 15 <= area <= 900:  # Reasonable area range
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.5:  # Relaxed circularity
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
    """
    SimpleBlobDetector optimized for filled black circles
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    
    # Invert for blob detection (blobs should be dark)
    inverted = 255 - gray
    
    # Setup SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()
    
    # Filter by Area
    params.filterByArea = True
    params.minArea = 20
    params.maxArea = 800
    
    # Filter by Circularity - more relaxed
    params.filterByCircularity = True
    params.minCircularity = 0.6  # Relaxed from 0.7
    
    # Filter by Convexity - more relaxed
    params.filterByConvexity = True
    params.minConvexity = 0.7  # Relaxed from 0.8
    
    # Filter by Inertia - more relaxed
    params.filterByInertia = True
    params.minInertiaRatio = 0.4  # Relaxed from 0.5
    
    # Don't filter by color
    params.filterByColor = False
    
    # Create detector
    detector = cv2.SimpleBlobDetector_create(params)
    
    # Detect blobs
    keypoints = detector.detect(inverted)
    
    candidates = []
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        radius = int(kp.size / 2)  # SimpleBlobDetector returns diameter as size
        confidence = kp.response  # Use response as confidence
        
        candidate = {
            'center': (x, y),
            'radius': radius,
            'area': np.pi * radius * radius,
            'circularity': 1.0,  # Assume high circularity from blob detector
            'confidence': confidence,
            'method': 'blob'
        }
        candidates.append(candidate)
    
    logging.info(f"SimpleBlobDetector found {len(candidates)} circles")
    return candidates


def cluster_radius_groups(candidates, n_groups=2):
    """
    Use clustering to find multiple radius groups instead of just one
    """
    if len(candidates) < 3:
        return candidates
    
    radii = [c['radius'] for c in candidates]
    
    # Simple approach: find two most common radii
    radius_counts = Counter(radii)
    most_common = radius_counts.most_common(n_groups)
    
    if len(most_common) == 1:
        # Only one significant radius group
        target_radius = most_common[0][0]
        tolerance = max(2, target_radius // 4)
        filtered = [c for c in candidates if abs(c['radius'] - target_radius) <= tolerance]
    else:
        # Multiple radius groups
        filtered = []
        for target_radius, count in most_common:
            if count >= 2:  # At least 2 circles of this size
                tolerance = max(2, target_radius // 4)
                group_circles = [c for c in candidates if abs(c['radius'] - target_radius) <= tolerance]
                filtered.extend(group_circles)
        
        # Remove duplicates
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


def detect_circles_combined_improved(image):
    """
    Improved combined detection with multiple methods
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    all_candidates = []
    
    # Method 1: Multi-threshold contour detection
    logging.info("Running multi-threshold contour detection...")
    contour_candidates, threshold_stats = auto_calibrate_multiple_thresholds(gray)
    
    # Add method tag
    for c in contour_candidates:
        c['method'] = 'contour'
    
    all_candidates.extend(contour_candidates)
    
    # Method 2: SimpleBlobDetector
    logging.info("Running SimpleBlobDetector...")
    blob_candidates = detect_circles_blob_detector(image)
    all_candidates.extend(blob_candidates)
    
    # Method 3: HoughCircles backup (multiple parameter sets)
    logging.info("Running HoughCircles backup...")
    hough_candidates = detect_circles_hough_multiple(gray)
    
    # Add method tag
    for c in hough_candidates:
        c['method'] = 'hough'
    
    all_candidates.extend(hough_candidates)
    
    logging.info(f"Total candidates from all methods: {len(all_candidates)}")
    
    if not all_candidates:
        return [], None
    
    # Plot radius histogram for debugging
    plot_radius_histogram(all_candidates)
    
    # Filter by multiple radius groups
    filtered_candidates = cluster_radius_groups(all_candidates)
    
    # Enhanced circularity filtering - more relaxed
    quality_filtered = []
    for c in filtered_candidates:
        if c.get('circularity', 0.8) > 0.6:  # More relaxed threshold
            quality_filtered.append(c)
    
    logging.info(f"After quality filtering: {len(quality_filtered)} circles")
    
    # Remove overlapping detections
    final_candidates = remove_overlapping_detections_improved(quality_filtered)
    
    # Convert to expected format
    detected_circles = [(c['center'][0], c['center'][1], c['radius']) for c in final_candidates]
    
    # Create debug binary (using best threshold from stats)
    if threshold_stats:
        best_thresh = max(threshold_stats, key=lambda x: x[2])[0]
        _, debug_binary = cv2.threshold(gray, best_thresh, 255, cv2.THRESH_BINARY_INV)
    else:
        debug_binary = None
    
    return detected_circles, debug_binary


def detect_circles_hough_multiple(gray):
    """
    HoughCircles with multiple parameter sets
    """
    # Apply blur
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)
    
    candidates = []
    
    # Multiple parameter sets
    param_sets = [
        {'dp': 1, 'minDist': 15, 'param1': 50, 'param2': 20, 'minR': 4, 'maxR': 20},
        {'dp': 1, 'minDist': 12, 'param1': 40, 'param2': 18, 'minR': 5, 'maxR': 18},
        {'dp': 1, 'minDist': 18, 'param1': 60, 'param2': 25, 'minR': 3, 'maxR': 22},
    ]
    
    for params in param_sets:
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=params['dp'],
            minDist=params['minDist'],
            param1=params['param1'],
            param2=params['param2'],
            minRadius=params['minR'],
            maxRadius=params['maxR']
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                candidate = {
                    'center': (x, y),
                    'radius': r,
                    'area': np.pi * r * r,
                    'circularity': 0.8,  # Assume good circularity from Hough
                    'confidence': params['param2'] / 30.0  # Normalize confidence
                }
                candidates.append(candidate)
    
    logging.info(f"HoughCircles found {len(candidates)} total candidates")
    return candidates


def remove_overlapping_detections_improved(circles, min_distance=10):
    """
    Improved overlap removal with priority for better methods
    """
    if not circles:
        return []
    
    # Priority order: blob > contour > hough
    method_priority = {'blob': 3, 'contour': 2, 'hough': 1}
    
    # Sort by method priority, then by circularity/confidence
    def sort_key(c):
        method_score = method_priority.get(c.get('method', 'contour'), 1)
        quality_score = c.get('confidence', c.get('circularity', 0.5))
        return method_score * quality_score
    
    sorted_circles = sorted(circles, key=sort_key, reverse=True)
    
    filtered_circles = []
    
    for current_circle in sorted_circles:
        x1, y1 = current_circle['center']
        is_too_close = False
        
        # Check distance to already accepted circles
        for existing_circle in filtered_circles:
            x2, y2 = existing_circle['center']
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            
            if distance < min_distance:
                is_too_close = True
                break
        
        if not is_too_close:
            filtered_circles.append(current_circle)
    
    logging.info(f"After improved overlap removal: {len(filtered_circles)} unique circles")
    return filtered_circles


def plot_radius_histogram(candidates):
    """
    Plot radius distribution for debugging
    """
    radii = [c['radius'] for c in candidates]
    
    plt.figure(figsize=(10, 6))
    plt.hist(radii, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Radius (pixels)')
    plt.ylabel('Count')
    plt.title('Radius Distribution of Circle Candidates')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    base_path = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(base_path, "Output_pictures", "radius_histogram.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Radius histogram saved to: radius_histogram.png")


def visualize_detected_circles_detailed(image, circles, save_path=None):
    """
    Enhanced visualization with method information
    """
    result_image = image.copy()
    if len(result_image.shape) == 1:
        result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)
    
    for i, (x, y, r) in enumerate(circles):
        # Draw red filled circle at center (small)
        cv2.circle(result_image, (x, y), 3, (0, 0, 255), -1)  # Red center point
        # Draw circle boundary in blue
        cv2.circle(result_image, (x, y), r, (255, 0, 0), 1)   # Blue boundary
        # Add index number
        cv2.putText(result_image, str(i+1), (x-8, y-r-8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    if save_path:
        cv2.imwrite(save_path, result_image)
        logging.info(f"Detailed visualization saved to: {save_path}")
    
    return result_image


def save_circle_coordinates(circles, output_path="circle_coordinates.json"):
    """
    Save circle coordinates to JSON file
    """
    circle_data = {
        "detection_method": "Multi-Method Improved",
        "circles": [
            {
                "id": i + 1,
                "center_x": int(x),
                "center_y": int(y),
                "radius": int(r),
                "coordinates": [int(x), int(y)]
            }
            for i, (x, y, r) in enumerate(circles)
        ],
        "total_circles": len(circles),
        "average_radius": float(np.mean([r for _, _, r in circles])) if circles else 0,
        "radius_std": float(np.std([r for _, _, r in circles])) if circles else 0
    }
    
    # Save to JSON
    base_path = os.path.dirname(os.path.abspath(__file__))
    full_output_path = os.path.join(base_path, "Output_pictures", output_path)
    
    with open(full_output_path, 'w') as f:
        json.dump(circle_data, f, indent=2)
    
    logging.info(f"Circle coordinates saved to: {full_output_path}")
    return circle_data


def save_debug_image(image, filename):
    """Save debug image to output folder"""
    base_path = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_path, "Output_pictures", filename)
    cv2.imwrite(output_path, image)


def filter_json_by_radius(json_path):
    """JSON sugÃ¡r alapÃº szÅ±rÃ©s"""
    with open(json_path, "r") as f:
        data = json.load(f)

    circles = data.get("circles", [])

    if not circles:
        logging.info("Nincsenek kÃ¶rÃ¶k a fÃ¡jlban.")
        return

    # Radii kigyÅ±jtÃ©se
    radii = [c["radius"] for c in circles]

    # Leggyakoribb sugÃ¡r meghatÃ¡rozÃ¡sa
    radius_counts = Counter(radii)
    mode_radius, count = radius_counts.most_common(1)[0]
    logging.info(f"Leggyakoribb sugÃ¡r: {mode_radius} (elÅ‘fordulÃ¡s: {count}x)")

    # Csak azok maradjanak, amiknek a sugara megegyezik a leggyakoribb sugÃ¡rral (Â±1)
    filtered_circles = []
    for c in circles:
        r = c["radius"]
        if r == mode_radius or r == mode_radius - 1 or r == mode_radius + 1:
            filtered_circles.append(c)

    # FrissÃ­tett adatok
    data["circles"] = filtered_circles
    data["total_circles"] = len(filtered_circles)
    data["average_radius"] = mode_radius if filtered_circles else 0
    data["mode_radius"] = mode_radius

    # FelÃ¼lÃ­rÃ¡s
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    logging.info(f"JSON frissÃ­tve: {json_path}")
    logging.info(f"Maradt kÃ¶rÃ¶k szÃ¡ma: {data['total_circles']}")


def check_coordinates_black_color(json_path, gray_image, black_threshold=120):
    """
    ÃšJ FÃœGGVÃ‰NY: EllenÅ‘rzi a JSON-ban lÃ©vÅ‘ koordinÃ¡tÃ¡k helyÃ©n a pixelintenzitÃ¡st
    """
    logging.info("=== KOORDINÃTÃK FEKETESÃ‰G ELLENÅRZÃ‰SE ===")
    
    # JSON betÃ¶ltÃ©se
    with open(json_path, "r") as f:
        data = json.load(f)
    
    circles = data.get("circles", [])
    if not circles:
        logging.info("Nincsenek kÃ¶rÃ¶k a JSON fÃ¡jlban.")
        return [], []
    
    black_circles = []
    rejected_circles = []
    
    for circle in circles:
        x = int(circle["center_x"])
        y = int(circle["center_y"])
        circle_id = circle["id"]
        
        # EllenÅ‘rzÃ©s hogy a koordinÃ¡tÃ¡k a kÃ©pen belÃ¼l vannak-e
        if 0 <= x < gray_image.shape[1] and 0 <= y < gray_image.shape[0]:
            # Pixel intenzitÃ¡s a kÃ¶zÃ©ppontban
            center_intensity = gray_image[y, x]
            
            # Fekete teszt
            is_black = center_intensity < black_threshold
            
            if is_black:
                black_circles.append(circle)
                logging.info(f"ELFOGADVA: KÃ¶r #{circle_id} ({x},{y}) - intenzitÃ¡s: {center_intensity} < {black_threshold}")
            else:
                rejected_circles.append(circle)
                logging.info(f"ELUTASÃTVA: KÃ¶r #{circle_id} ({x},{y}) - intenzitÃ¡s: {center_intensity} >= {black_threshold}")
        else:
            rejected_circles.append(circle)
            logging.info(f"ELUTASÃTVA: KÃ¶r #{circle_id} ({x},{y}) - kÃ©p hatÃ¡rain kÃ­vÃ¼l")
    
    logging.info(f"FeketesÃ©g teszt eredmÃ©ny: {len(circles)} -> {len(black_circles)} elfogadott, {len(rejected_circles)} elutasÃ­tott")
    
    return black_circles, rejected_circles


def save_black_filtered_json(black_circles, output_path="black_filtered_coordinates.json"):
    """
    Elmenti a feketesÃ©g teszt utÃ¡n maradt kÃ¶rÃ¶ket Ãºj JSON fÃ¡jlba
    """
    filtered_data = {
        "detection_method": "Multi-Method + Radius Filter + Black Color Filter",
        "circles": black_circles,
        "total_circles": len(black_circles),
        "average_radius": float(np.mean([c["radius"] for c in black_circles])) if black_circles else 0,
        "filtering_steps": [
            "1. Multi-method circle detection",
            "2. Radius uniformity filtering",  
            "3. Black color intensity check"
        ]
    }
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    full_output_path = os.path.join(base_path, "Output_pictures", output_path)
    
    with open(full_output_path, "w") as f:
        json.dump(filtered_data, f, indent=2)
    
    logging.info(f"Fekete szÅ±rÃ©s utÃ¡ni koordinÃ¡tÃ¡k mentve: {full_output_path}")
    return filtered_data


def visualize_black_filter_results(image, accepted_circles, rejected_circles, save_path):
    """
    VizualizÃ¡lja a fekete szÅ±rÃ©s eredmÃ©nyeit
    """
    result_image = image.copy()
    if len(result_image.shape) == 1:
        result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)
    
    # Elfogadott kÃ¶rÃ¶k - zÃ¶ld
    for circle in accepted_circles:
        x = int(circle["center_x"])
        y = int(circle["center_y"])
        r = int(circle["radius"])
        circle_id = circle["id"]
        
        cv2.circle(result_image, (x, y), 4, (0, 255, 0), -1)  # ZÃ¶ld kÃ¶zÃ©ppont
        cv2.circle(result_image, (x, y), r, (0, 255, 0), 2)   # ZÃ¶ld keret
        cv2.putText(result_image, f"OK{circle_id}", (x-15, y-r-8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    # ElutasÃ­tott kÃ¶rÃ¶k - piros
    for circle in rejected_circles:
        x = int(circle["center_x"])
        y = int(circle["center_y"])
        r = int(circle["radius"])
        circle_id = circle["id"]
        
        cv2.circle(result_image, (x, y), 4, (0, 0, 255), -1)  # Piros kÃ¶zÃ©ppont
        cv2.circle(result_image, (x, y), r, (0, 0, 255), 1)   # Piros keret
        cv2.putText(result_image, f"X{circle_id}", (x-15, y+r+15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    cv2.imwrite(save_path, result_image)
    logging.info(f"Fekete szÅ±rÃ©s vizualizÃ¡ciÃ³ mentve: {save_path}")


def visualize_circles(json_path, image, output_path="output_with_circles.jpg"):
    """JSON koordinÃ¡tÃ¡k alapjÃ¡n kÃ¶rÃ¶k berajzolÃ¡sa"""
    with open(json_path, "r") as f:
        data = json.load(f)

    circles = data.get("circles", [])
    if not circles:
        logging.info("Nincsenek kÃ¶rÃ¶k a JSON-ban.")
        return

    # KÃ¶rÃ¶k berajzolÃ¡sa
    for c in circles:
        x = int(c["center_x"])
        y = int(c["center_y"])
        r = int(c["radius"])

        # kÃ¶zÃ©ppont pirossal
        cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
        # teljes kÃ¶r zÃ¶lddel
        cv2.circle(image, (x, y), r, (0, 255, 0), 2)

    # MentÃ©s
    cv2.imwrite(output_path, image)
    logging.info(f"KÃ¶rÃ¶k vizualizÃ¡lva: {output_path}")


if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    logging.info("=== ENHANCED CIRCLE DETECTION + BLACK COLOR FILTER ===")
    
    # Load the perspective-corrected image
    image = load_transformed_image()
    
    if image is None:
        logging.error("Failed to load input image. Make sure image_process.py has been run first.")
        exit(1)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    
    # PHASE 1: Run improved combined detection
    logging.info("PHASE 1: Starting improved multi-method circle detection...")
    detected_circles, binary_debug = detect_circles_combined_improved(image)
    
    # Save debug binary image if available
    if binary_debug is not None:
        save_debug_image(binary_debug, "step_350_improved_binary.jpg")
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_path, "Output_pictures")
    json_path = os.path.join(output_path, "circle_coordinates.json")
    
    if detected_circles:
        logging.info(f"PHASE 1 eredmÃ©ny: {len(detected_circles)} kÃ¶r detektÃ¡lva")
        
        # Print detailed results
        radii = [r for _, _, r in detected_circles]
        logging.info(f"Ãtlag sugÃ¡r: {np.mean(radii):.1f} (szÃ³rÃ¡s: {np.std(radii):.1f})")
        
        # Visualize results
        visualization_path = os.path.join(output_path, "step_351_improved_result.jpg")
        visualize_detected_circles_detailed(image, detected_circles, visualization_path)
        
        # Save coordinates to JSON
        save_circle_coordinates(detected_circles)
        
        # PHASE 2: Filter by radius
        logging.info("PHASE 2: SugÃ¡r alapÃº szÅ±rÃ©s...")
        filter_json_by_radius(json_path)
        
        # PHASE 3: Check black color at coordinates
        logging.info("PHASE 3: Fekete szÃ­n ellenÅ‘rzÃ©s a koordinÃ¡tÃ¡knÃ¡l...")
        black_circles, rejected_circles = check_coordinates_black_color(json_path, gray)
        
        # Save black filtered results
        black_filtered_path = os.path.join(output_path, "black_filtered_coordinates.json")
        save_black_filtered_json(black_circles, "black_filtered_coordinates.json")
        
        # Visualize black filter results
        black_filter_viz_path = os.path.join(output_path, "step_352_black_filter_result.jpg")
        visualize_black_filter_results(image, black_circles, rejected_circles, black_filter_viz_path)
        
        # Final visualization with accepted circles only
        final_viz_path = os.path.join(output_path, "step_353_final_result.jpg")
        visualize_circles(black_filtered_path, image.copy(), final_viz_path)
        
        # Final results summary
        logging.info("=== VÃ‰GSÅ EREDMÃ‰NYEK ===")
        logging.info(f"Phase 1 (Kezdeti detektÃ¡lÃ¡s): {len(detected_circles)} kÃ¶r")
        logging.info(f"Phase 2 (SugÃ¡r szÅ±rÃ©s): {len(json.load(open(json_path))['circles'])} kÃ¶r")
        logging.info(f"Phase 3 (Fekete szÅ±rÃ©s): {len(black_circles)} kÃ¶r")
        
        if black_circles:
            logging.info("Elfogadott kÃ¶rÃ¶k:")
            for circle in black_circles:
                x, y = int(circle["center_x"]), int(circle["center_y"])
                intensity = gray[y, x]
                logging.info(f"  KÃ¶r #{circle['id']}: ({x},{y}), R={circle['radius']}, IntenzitÃ¡s={intensity}")
        
        if rejected_circles:
            logging.info("ElutasÃ­tott kÃ¶rÃ¶k (nem feketÃ©k):")
            for circle in rejected_circles:
                x, y = int(circle["center_x"]), int(circle["center_y"])
                intensity = gray[y, x] if 0 <= y < gray.shape[0] and 0 <= x < gray.shape[1] else "N/A"
                logging.info(f"  KÃ¶r #{circle['id']}: ({x},{y}), R={circle['radius']}, IntenzitÃ¡s={intensity}")
        
        logging.info("=== FELDOLGOZÃS BEFEJEZVE ===")
        logging.info("Mentett fÃ¡jlok:")
        logging.info("  - step_351: Kezdeti detektÃ¡lÃ¡s")
        logging.info("  - step_352: Fekete szÅ±rÃ©s eredmÃ©ny (zÃ¶ld=jÃ³, piros=rossz)")
        logging.info("  - step_353: VÃ©gsÅ‘ eredmÃ©ny (csak elfogadott kÃ¶rÃ¶k)")
        logging.info("  - black_filtered_coordinates.json: Csak a fekete kÃ¶rÃ¶k koordinÃ¡tÃ¡i")
        
    else:
        logging.warning("Nincs kÃ¶r detektÃ¡lva az elsÅ‘ fÃ¡zisban!")
        
        # Save empty visualization
        visualization_path = os.path.join(output_path, "step_351_improved_result.jpg")
        visualize_detected_circles_detailed(image, [], visualization_path)