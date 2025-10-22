"""
Dot-Number Matching Module
Simple and reliable: filter duplicates/outliers, then match by distance
"""

import json
import math
import logging
import cv2
import numpy as np
from pathlib import Path
from scipy.optimize import linear_sum_assignment


def dot_to_dict(dot_path):
    """Convert dot JSON to dictionary {id: {x, y}}"""
    try:
        with open(dot_path) as f:
            data = json.load(f)
        
        result = {}
        for circle in data.get("circles", []):
            id_val = circle.get("id")
            global_coords = circle.get("global_coordinates", {})
            x = global_coords.get("x")
            y = global_coords.get("y")
            
            if id_val is not None and x is not None and y is not None:
                result[id_val] = {"x": x, "y": y}
        
        logging.info(f"Loaded {len(result)} dots")
        return result
    
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error loading dots: {e}")
        return {}


def num_to_dict(num_path, expected_range):
    """
    Simple filtering:
    1. Remove large bounding boxes (outliers)
    2. Remove out-of-range numbers
    3. Remove duplicates (keep first occurrence)
    """
    try:
        with open(num_path) as f:
            data = json.load(f)
        
        # Collect all numbers with their bboxes
        all_numbers = []
        for num in data.get("numbers", []):
            bbox = num.get("bbox")
            number = num.get("number")
            coords = num.get("global_coordinates", {})
            x = coords.get("x")
            y = coords.get("y")
            
            if bbox and len(bbox) == 4 and number is not None and x is not None and y is not None:
                w, h = bbox[2], bbox[3]
                all_numbers.append({
                    "number": number,
                    "x": x,
                    "y": y,
                    "bbox": bbox,
                    "area": w * h
                })
        
        if not all_numbers:
            logging.warning("No valid numbers found")
            return {}
        
        logging.info(f"Loaded {len(all_numbers)} number detections")
        
        # Step 1: Calculate bbox area statistics
        areas = [item["area"] for item in all_numbers]
        avg_area = np.mean(areas)
        std_area = np.std(areas)
        area_threshold = avg_area + 2 * std_area
        
        logging.info(f"Bbox area - Avg: {avg_area:.1f}, Std: {std_area:.1f}, Threshold: {area_threshold:.1f}")
        
        # Step 2: Filter by bbox size
        filtered = []
        removed_large = 0
        for item in all_numbers:
            if item["area"] <= area_threshold:
                filtered.append(item)
            else:
                removed_large += 1
                logging.debug(f"Removed {item['number']} - large bbox area: {item['area']:.0f}")
        
        logging.info(f"After bbox filter: {len(filtered)} (removed {removed_large} large bboxes)")
        
        # Step 3: Filter by expected range
        in_range = []
        removed_range = 0
        for item in filtered:
            if 1 <= item["number"] <= expected_range:
                in_range.append(item)
            else:
                removed_range += 1
                logging.debug(f"Removed {item['number']} - out of range")
        
        logging.info(f"After range filter: {len(in_range)} (removed {removed_range} out-of-range)")
        
        # Step 4: Remove duplicates - keep first occurrence
        seen = set()
        result = {}
        removed_dupes = 0
        
        for item in in_range:
            num = item["number"]
            if num not in seen:
                seen.add(num)
                result[num] = {"x": item["x"], "y": item["y"]}
            else:
                removed_dupes += 1
                logging.debug(f"Removed duplicate: {num}")
        
        logging.info(f"After duplicate removal: {len(result)} (removed {removed_dupes} duplicates)")
        
        # Summary
        missing = [n for n in range(1, expected_range + 1) if n not in result]
        logging.info(f"Missing numbers: {len(missing)}/{expected_range}")
        if len(missing) <= 20:
            logging.info(f"Missing: {missing}")
        
        return result
    
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error loading numbers: {e}")
        return {}


def hungarian_pairing(dot_dict, num_dict, max_distance=25):
    """
    Hungarian algorithm for optimal matching within distance threshold
    """
    if not dot_dict or not num_dict:
        logging.warning("Cannot pair - empty dot or number dict")
        return []
    
    logging.info(f"\nMatching {len(dot_dict)} dots with {len(num_dict)} numbers...")
    
    # Create sorted lists
    dot_ids = sorted(dot_dict.keys())
    num_ids = sorted(num_dict.keys())
    
    # Build cost matrix
    n_dots = len(dot_ids)
    n_nums = len(num_ids)
    LARGE_COST = 1e9
    
    cost_matrix = np.full((n_dots, n_nums), LARGE_COST)
    
    for i, dot_id in enumerate(dot_ids):
        dot_x, dot_y = dot_dict[dot_id]["x"], dot_dict[dot_id]["y"]
        
        for j, num_id in enumerate(num_ids):
            num_x, num_y = num_dict[num_id]["x"], num_dict[num_id]["y"]
            dist = math.hypot(num_x - dot_x, num_y - dot_y)
            
            if dist <= max_distance:
                cost_matrix[i, j] = dist
    
    # Run Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Extract valid pairs
    pairs = []
    
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] < LARGE_COST:
            dot_id = dot_ids[i]
            num_id = num_ids[j]
            dist = cost_matrix[i, j]
            
            pairs.append({
                "dot_id": dot_id,
                "dot_coord": {"x": dot_dict[dot_id]["x"], "y": dot_dict[dot_id]["y"]},
                "num_value": num_id,
                "num_coord": {"x": num_dict[num_id]["x"], "y": num_dict[num_id]["y"]},
                "distance": round(dist, 3)
            })
    
    # Sort by dot_id for clean output
    pairs.sort(key=lambda p: p["dot_id"])
    
    # Statistics
    if pairs:
        distances = [p["distance"] for p in pairs]
        logging.info(f"Successfully paired: {len(pairs)}")
        logging.info(f"Distance range: {min(distances):.1f} - {max(distances):.1f} px (avg: {np.mean(distances):.1f})")
    else:
        logging.warning("No pairs found within distance threshold!")
    
    unpaired_dots = len(dot_dict) - len(pairs)
    unpaired_nums = len(num_dict) - len(pairs)
    
    if unpaired_dots > 0:
        logging.info(f"Unpaired dots: {unpaired_dots}")
    if unpaired_nums > 0:
        logging.info(f"Unpaired numbers: {unpaired_nums}")
    
    return pairs


def visualize_pairs(pairs, output_path, main_image_path=None):
    """Simple visualization - green dots and lines"""
    if main_image_path and main_image_path.exists():
        img = cv2.imread(str(main_image_path))
    else:
        if not pairs:
            logging.warning("No pairs to visualize")
            return
        max_x = max(p["dot_coord"]["x"] for p in pairs) + 100
        max_y = max(p["dot_coord"]["y"] for p in pairs) + 100
        img = 255 * np.ones((max_y, max_x, 3), dtype=np.uint8)
    
    for pair in pairs:
        dot_x = int(pair["dot_coord"]["x"])
        dot_y = int(pair["dot_coord"]["y"])
        num_x = int(pair["num_coord"]["x"])
        num_y = int(pair["num_coord"]["y"])
        num_value = pair["num_value"]
        
        # Green dot
        cv2.circle(img, (dot_x, dot_y), 5, (0, 255, 0), -1)
        
        # Red number center
        cv2.circle(img, (num_x, num_y), 3, (0, 0, 255), -1)
        
        # Gray line
        cv2.line(img, (dot_x, dot_y), (num_x, num_y), (150, 150, 150), 1)
        
        # Label
        cv2.putText(img, str(num_value), (dot_x + 10, dot_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    cv2.imwrite(str(output_path), img)
    logging.info(f"Visualization saved to {output_path}")


def matchmaker_main(config, picture_name, expected_range):
    """Main entry point - simple and clean"""
    logging.info("="*60)
    logging.info("DOT-NUMBER MATCHING")
    logging.info("="*60)
    
    base_path = config['_base_path']
    config_dir = base_path / config['paths']['config_dir']
    
    dot_path = config_dir / config['filenames']['global_dots']
    num_path = config_dir / config['filenames']['global_numbers']
    pairing_path = config_dir / config['filenames']['pairing']
    
    # Load dots
    logging.info("\n1. Loading dots...")
    dot_dict = dot_to_dict(dot_path)
    
    if not dot_dict:
        logging.error("No dots loaded!")
        return
    
    # Load and filter numbers
    logging.info("\n2. Loading and filtering numbers...")
    num_dict = num_to_dict(num_path, expected_range)
    
    if not num_dict:
        logging.error("No valid numbers found!")
        return
    
    # Match using Hungarian algorithm
    logging.info("\n3. Optimal matching (Hungarian algorithm)...")
    max_dist = config['number_detection'].get('max_match_distance', 25)
    pairs = hungarian_pairing(dot_dict, num_dict, max_distance=max_dist)
    
    if not pairs:
        logging.error("No pairs created!")
        return
    
    # Save results
    with open(pairing_path, "w") as f:
        json.dump(pairs, f, indent=4)
    logging.info(f"\nSaved {len(pairs)} pairs to {pairing_path}")
    
    # Visualize
    logging.info("\n4. Creating visualization...")
    picture_path = base_path / config['paths']['pictures_dir'] / picture_name
    viz_path = base_path / config['filenames']['pairing_viz']
    visualize_pairs(pairs, viz_path, picture_path if picture_path.exists() else None)
    
    # Final summary
    accuracy = (len(pairs) / expected_range) * 100
    
    logging.info("\n" + "="*60)
    logging.info("SUMMARY")
    logging.info("="*60)
    logging.info(f"Expected: {expected_range}")
    logging.info(f"Dots detected: {len(dot_dict)}")
    logging.info(f"Numbers detected: {len(num_dict)}")
    logging.info(f"Successfully paired: {len(pairs)}")
    logging.info(f"Accuracy: {accuracy:.1f}%")
    logging.info("="*60 + "\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    try:
        config_path = Path(__file__).parent / "config.json"
        with open(config_path) as f:
            config = json.load(f)
        
        config['_base_path'] = Path(__file__).parent
        
        picture_name = input("Enter picture name: ").strip()
        expected_range = int(input("Enter expected upper limit: ").strip())
        
        matchmaker_main(config, picture_name, expected_range)
        
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)