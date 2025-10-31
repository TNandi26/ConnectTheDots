import cv2
import numpy as np
import json
import logging

def preprocess_full_image_for_dots(gray_image, config, debug_dir):
    """
    Applies normalization and adaptive threshold to the entire image
    based on 'circle_detection' parameters.
    """
    try:
        cfg = config['circle_detection']
        bSize = cfg['adaptive_blockSize']
        C_val = cfg['adaptive_C']
        
        blur_k = config.get('number_detection', {}).get('full_image_blur_kernel', 51)
        
        if blur_k % 2 == 0:
            blur_k += 1
            
        logging.info(f"Image process for dots... (Blur: {blur_k}, BlockSize: {bSize}, C: {C_val})")
        
        background = cv2.GaussianBlur(gray_image, (blur_k, blur_k), 0)
        normalized_image = cv2.divide(gray_image, background, scale=255.0)

        clean_full_image = cv2.adaptiveThreshold(
            normalized_image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            bSize, C_val
        )
        
        #Save to debug dir
        (debug_dir.parent / "config").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir.parent / "config" / "_full_dot_preprocessed.jpg"), clean_full_image)
        return clean_full_image

    except Exception as e:
        logging.error(f"Error occured during preprocess: {e}")
        return None

def detect_dots_from_binary(binary_image, gray_image, config):
    """
    Using blob detector to find dots with the given configuration
    """
    try:
        cfg = config['circle_detection']['blob_detector']
    except KeyError:
        logging.error("'blob_detector' section is missing from the config.json")
        return []
    binary_inv = cv2.bitwise_not(binary_image)
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = cfg.get('filterByArea', True)
    params.minArea = cfg.get('minArea', 15)
    params.maxArea = cfg.get('maxArea', 250)
    params.filterByCircularity = cfg.get('filterByCircularity', True)
    params.minCircularity = cfg.get('minCircularity', 0.70)
    params.filterByConvexity = cfg.get('filterByConvexity', True)
    params.minConvexity = cfg.get('minConvexity', 0.80)
    params.filterByInertia = cfg.get('filterByInertia', True)
    params.minInertiaRatio = cfg.get('minInertiaRatio', 0.50)
    params.filterByColor = False 
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(binary_inv)
    all_candidates = []
    for kp in keypoints:
        center = (int(kp.pt[0]), int(kp.pt[1]))
        radius = int(kp.size / 2)
        intensity = 0
        if 0 <= center[1] < gray_image.shape[0] and 0 <= center[0] < gray_image.shape[1]:
            intensity = int(gray_image[center[1], center[0]])
        all_candidates.append({ 'center': center, 'radius': radius, 'intensity': intensity })
    return all_candidates


def process_segment(image_path, config, expected_range=None):
    segment_name = image_path.stem
    logging.info(f"Processing: {segment_name}")
    image = cv2.imread(str(image_path))
    if image is None:
        logging.error(f"Failed to load: {segment_name}")
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    logging.info("PHASE 1: Normalizing and Binarizing...")
    background = cv2.GaussianBlur(gray, (51, 51), 0)
    normalized_image = cv2.divide(gray, background, scale=255.0)
    normalized_image_uint8 = normalized_image.astype(np.uint8)
    cfg_thresh = config['circle_detection']
    try:
        blockSize = cfg_thresh['adaptive_blockSize']
        C = cfg_thresh['adaptive_C']
    except KeyError as e:
        logging.error(f"Error:'{e.args[0]}' key is missing from the config.json 'circle_detection' section")
        return None 
    logging.info(f"Binary image: blockSize={blockSize}, C={C}")
    final_binary_image = cv2.adaptiveThreshold(
        normalized_image_uint8, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize, C
    )
    logging.info("PHASE 2: BlobDetector-based dot detection...")
    candidates = detect_dots_from_binary(final_binary_image, gray, config)
    if not candidates:
        logging.warning(f"No circles found in {segment_name}")
        return None
    filtered = candidates
    viz_dir = config['_base_path'] / config['paths']['dot_viz_dir']
    viz_dir.mkdir(parents=True, exist_ok=True)
    viz_img = cv2.cvtColor(final_binary_image, cv2.COLOR_GRAY2BGR)
    for i, c in enumerate(filtered, 1):
        x, y, r = c['center'][0], c['center'][1], c['radius']
        cv2.circle(viz_img, (x, y), 3, (0, 0, 255), -1)
        cv2.circle(viz_img, (x, y), r, (0, 255, 0), 2)
        cv2.putText(viz_img, str(i), (x - 8, y - r - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.imwrite(str(viz_dir / f"{segment_name}_final_DEBUG.jpg"), viz_img)
    logging.info(f"DEBUG Visualization saved (on binary image)")
    segment_data = {
        "segment_name": segment_name,
        "circles": [
            { "pixel_x": c['center'][0], "pixel_y": c['center'][1], "radius": c['radius'], "intensity": c['intensity'] }
            for c in filtered
        ],
        "total_circles": len(filtered)
    }
    logging.info(f"{segment_name}: {len(filtered)} circles detected\n")
    return segment_data


def convert_to_global(detected_json, segments_json, output_json, config):
    with open(detected_json) as f:
        detected_data = json.load(f)
    with open(segments_json) as f:
        segment_mapping = json.load(f)
    all_circles = []
    for segment in detected_data["segments"]:
        segment_name = segment["segment_name"] + ".jpg"
        if segment_name not in segment_mapping:
            logging.warning(f"Segment {segment_name} not in mapping")
            continue
        offset = segment_mapping[segment_name]["start"]
        for circle in segment["circles"]:
            all_circles.append({
                "segment_name": segment_name,
                "local_coordinates": { "x": circle["pixel_x"], "y": circle["pixel_y"] },
                "global_coordinates": { "x": circle["pixel_x"] + offset["x"], "y": circle["pixel_y"] + offset["y"] },
                "radius": circle["radius"], "intensity": circle["intensity"]
            })
    logging.info(f"Total circles before deduplication: {len(all_circles)}")
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
        else:
            unique.append(circle)
    logging.info(f"After deduplication: {len(unique)} circles")
    intensities = [c["intensity"] for c in unique if c.get("intensity") is not None]
    if not intensities:
        logging.warning("No valid intensities found for filtering.")
        filtered = unique
    else:
        mean_int = np.mean(intensities)
        std_int = np.std(intensities)
        std_mult = config['circle_detection']['intensity_std_multiplier']
        logging.info(f"Intensity stats - Mean: {mean_int:.2f}, Std: {std_int:.2f}")
        filtered = []
        removed_count = 0
        for c in unique:
            if c.get("intensity") is None: deviation = 0
            else: deviation = abs(c["intensity"] - mean_int)
            if deviation <= std_mult * std_int:
                filtered.append(c)
            else:
                removed_count += 1
        logging.info(f"After intensity filter: {len(filtered)} circles ({removed_count} removed)")
    for i, circle in enumerate(filtered, 1):
        circle["id"] = i
    with open(output_json, 'w') as f:
        json.dump({
            "detection_method": "Normalized Adaptive + BlobDetector",
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
    
    logging.info(f"Processing {len(jpg_files)} segments...")
    
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
    
    logging.info("Creating final 'main_with_dots.jpg' on processed image...")
    picture_path = base_path / config['paths']['pictures_dir'] / picture_name
    
    if picture_path.exists():
        original_img = cv2.imread(str(picture_path))
        gray_original = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        
        dot_friendly_image = preprocess_full_image_for_dots(
            gray_original, config, config_dir.parent / "number_debug"
        )
        
        if dot_friendly_image is not None:
            main_img_base = cv2.cvtColor(dot_friendly_image, cv2.COLOR_GRAY2BGR)
        else:
            logging.warning("Failed to create dot-friendly image, drawing on original.")
            main_img_base = original_img

        for circle in global_circles:
            x = circle["global_coordinates"]["x"]
            y = circle["global_coordinates"]["y"]
            r = circle["radius"]
            cv2.circle(main_img_base, (x, y), 3, (0, 0, 255), -1)
            cv2.circle(main_img_base, (x, y), r, (0, 255, 0), 2)
        
        output_path = config['_base_path'] / config['filenames']['main_with_dots']
        cv2.imwrite(str(output_path), main_img_base)
        logging.info(f"Main image visualization saved to {output_path} (based on processed dot image)")
    else:
        logging.warning(f"Main image not found: {picture_path}")
    
    logging.info("Dot detection completed\n")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)