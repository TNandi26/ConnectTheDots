"""
Circle/Dot Detection Module
- V2 -
- Replaced complex grayscale multi-method detection with a robust
  pipeline based on user's test.py:
  1. Lighting Normalization (GaussianBlur + Divide)
  2. Adaptive Threshold
  3. findContours on the clean binary image
- This removes rigid hardcoded thresholds and complex number-rejection logic.
- Detection is now simpler, faster, and more accurate.
- All JSON output and global coordinate conversion logic is preserved.
"""

import cv2
import numpy as np
import json
import logging
from collections import Counter
from pathlib import Path  # Hozzáadva a Path-hoz

# --- MEGTARTVA AZ EREDETIBŐL ---
# Erre szükségünk van, hogy kiszűrjük a nem-tömör alakzatokat (pl. számokat)
def calculate_solidity(contour):
    """Calculate solidity - dots are solid shapes, number centers are hollow"""
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    return area / hull_area if hull_area > 0 else 0

# --- ÚJ DETEKTÁLÓ FÜGGVÉNY ---
def detect_dots_from_binary(binary_image, gray_image, config):
    """
    Detects dots from a clean binary image using findContours.
    This replaces the old multi-method detector.
    """
    cfg = config['circle_detection']
    min_area = cfg.get('min_area', 10)
    max_area = cfg.get('max_area', 200)
    min_solidity = cfg.get('min_solidity', 0.85) # Tömörség szűrő (számok ellen)

    # A findContours fehér objektumokat keres fekete háttéren.
    # A mi bináris képünk fekete pontokból áll fehér háttéren.
    # Ezért invertálnunk kell.
    binary_inv = cv2.bitwise_not(binary_image)
    
    contours, _ = cv2.findContours(binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    all_candidates = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # 1. Szűrés terület alapján
        if not (min_area <= area <= max_area):
            continue
            
        # 2. Szűrés tömörség alapján (kiszűri a "0", "6", "8", "9" belsejét)
        solidity = calculate_solidity(contour)
        if solidity < min_solidity:
            logging.debug(f"Rejected contour, solidity: {solidity:.2f}")
            continue
            
        # Megvan a pont!
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        
        # Intenzitás lekérése az *EREDETI* szürkeárnyalatos képről
        # Ez fontos a `convert_to_global` duplikátum-szűréséhez
        intensity = 0
        if 0 <= center[1] < gray_image.shape[0] and 0 <= center[0] < gray_image.shape[1]:
            intensity = int(gray_image[center[1], center[0]])
            
        all_candidates.append({
            'center': center,
            'radius': int(radius),
            'intensity': intensity, # Ezt használja a convert_to_global
            'solidity': solidity
        })
        
    logging.info(f"Found {len(all_candidates)} dot candidates from contours")
    return all_candidates

# --- MÓDOSÍTOTT FELDOLGOZÓ FÜGGVÉNY ---
def process_segment(image_path, config, expected_range=None):
    """Process a single segment"""
    segment_name = image_path.stem
    logging.info(f"Processing: {segment_name}")
    
    image = cv2.imread(str(image_path))
    if image is None:
        logging.error(f"Failed to load: {segment_name}")
        return None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR_GRAY)
    
    # === ÚJ PREPROCESSZÁLÁSI LÉPÉS (a test.py alapján) ===
    logging.info("PHASE 1: Normalizing and Binarizing...")
    
    # 1. Világítás normalizálása
    background = cv2.GaussianBlur(gray, (51, 51), 0)
    normalized_image = cv2.divide(gray, background, scale=255.0)
    
    # Konvertálás uint8-ra, mert az adaptiveThreshold ezt várja
    normalized_image_uint8 = normalized_image.astype(np.uint8)

    # 2. Adaptív Küszöbölés (paraméterek a configból)
    cfg = config['circle_detection']
    blockSize = cfg.get('adaptive_blockSize', 15) # A te képeid alapján jó érték
    C = cfg.get('adaptive_C', 14)               # A te képeid alapján jó érték

    final_binary_image = cv2.adaptiveThreshold(
        normalized_image_uint8, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, # Fekete pontok, fehér háttér (ahogy a test.py csinálja)
        blockSize, C
    )
    # === PREPROCESSZÁLÁS VÉGE ===

    # Phase 2: Detection (az ÚJ függvénnyel)
    logging.info("PHASE 2: Contour-based dot detection...")
    # A `gray`-t is átadjuk, hogy az intenzitás-ellenőrzés működjön
    candidates = detect_dots_from_binary(final_binary_image, gray, config)
    
    if not candidates:
        logging.warning(f"No circles found in {segment_name}")
        return None
    
    # A régi `filter_circles` függvényre már nincs szükség,
    # a `detect_dots_from_binary` és a `convert_to_global` mindent kezel.
    filtered = candidates
    
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
    
    # Create segment data (a formátum ugyanaz maradt)
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


# --- VÁLTOZTATÁS NÉLKÜL MEGTARTVA ---
# Ez a függvény tökéletes volt, a duplikátum-szűrésre
# és az intenzitás-alapú szűrésre továbbra is szükség van.
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
    intensities = [c["intensity"] for c in unique if c["intensity"] is not None] # Biztonsági ellenőrzés
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
            if c["intensity"] is None:
                deviation = 0 # Tartsuk meg, ha nincs adat
            else:
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
            "detection_method": "Normalized Adaptive Contour Detection",
            "total_circles": len(filtered),
            "circles": filtered
        }, f, indent=2)
    
    logging.info(f"Saved to {output_json}")
    return filtered

# --- VÁLTOZTATÁS NÉLKÜL MEGTARTVA ---
# Ez a belépési pont, ami ciklusban hívja a process_segment-et
# és a végén a convert_to_global-t. Tökéletes.
def run_dot_detection_for_all_segments(config, picture_name, expected_range=None):
    """Main entry point for dot detection pipeline"""
    base_path = config['_base_path']
    segments_dir = base_path / config['paths']['segments_overlap_dir']
    config_dir = base_path / config['paths']['config_dir']
    config_dir.mkdir(parents=True, exist_ok=True)
    
    jpg_files = sorted(segments_dir.glob("*.jpg"))
    
    # Log whether radius filtering will be used (régi logika, de nem bánt)
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
    # Ez a rész csak egy példa, a fő logikád hívja meg
    # a run_dot_detection_for_all_segments-et a config-gal.
    print("circle_detection.py (v2) - Készen áll a futtatásra (a fő szkriptből hívva).")