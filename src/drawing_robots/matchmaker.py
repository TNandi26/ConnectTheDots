"""
Dot-Number Matching Module
Pairs detected dots with detected numbers using distance-based algorithm
Predicts missing numbers using neighbor analysis
"""

import json
import math
import logging
import cv2
import numpy as np


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
        
        return result
    
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error loading dots: {e}")
        return {}


def num_to_dict(num_path, expected_range):
    """Convert number JSON to dictionary"""
    try:
        with open(num_path) as f:
            data = json.load(f)
        
        result = {}
        next_available = expected_range + 1
        outliers = {}
        seen_counts = {}
        
        # Count occurrences
        for num in data.get("numbers", []):
            number = num.get("number")
            if number is not None:
                seen_counts[number] = seen_counts.get(number, 0) + 1
        
        # Process numbers
        for num in data.get("numbers", []):
            number = num.get("number")
            global_coords = num.get("global_coordinates", {})
            x = global_coords.get("x")
            y = global_coords.get("y")
            
            if number is None or x is None or y is None:
                continue
            
            # Check if outlier (out of range or duplicate)
            if number > expected_range or seen_counts[number] > 1:
                new_key = next_available
                result[new_key] = {"x": x, "y": y, "outlier": True, "original": number}
                outliers[number] = outliers.get(number, []) + [new_key]
                next_available += 1
                continue
            
            result[number] = {"x": x, "y": y}
        
        # Find missing numbers
        missing_numbers = []
        for n in range(1, expected_range + 1):
            if n not in seen_counts or seen_counts[n] != 1:
                missing_numbers.append(n)
        
        if outliers:
            logging.info(f"Outliers detected: {outliers}")
        if missing_numbers:
            logging.info(f"Missing numbers: {missing_numbers}")
        
        return result, outliers, missing_numbers
    
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error loading numbers: {e}")
        return {}, {}, []


def distance_based_pairing(dot_dict, num_dict, expected_range, missing_numbers, output_path):
    """
    Pair dots with numbers using greedy distance-based algorithm
    Then predict missing numbers using neighbor analysis
    """
    print(f"Dot detection accuracy: {((expected_range - abs(len(dot_dict) - expected_range)) / expected_range)*100:.2f}%")
    print(f"Number detection accuracy: {((expected_range - abs(len(num_dict) - expected_range)) / expected_range)*100:.2f}%")
    
    pairs = []
    unpaired_dots = set(dot_dict.keys())
    unpaired_nums = set(num_dict.keys())
    
    # Greedy pairing: repeatedly find closest dot-number pair
    while unpaired_dots and unpaired_nums:
        min_dist = None
        best_dot = None
        best_num = None
        
        for dot_id in unpaired_dots:
            dot_x, dot_y = dot_dict[dot_id]["x"], dot_dict[dot_id]["y"]
            
            for num_id in unpaired_nums:
                if num_id > expected_range:
                    continue
                
                num_x, num_y = num_dict[num_id]["x"], num_dict[num_id]["y"]
                dist = math.hypot(num_x - dot_x, num_y - dot_y)
                
                if min_dist is None or dist < min_dist:
                    min_dist = dist
                    best_dot = dot_id
                    best_num = num_id
        
        if best_dot is None:
            break
        
        unpaired_dots.remove(best_dot)
        unpaired_nums.remove(best_num)
        
        pairs.append({
            "dot_id": best_dot,
            "dot_coord": {"x": dot_dict[best_dot]["x"], "y": dot_dict[best_dot]["y"]},
            "num_value": best_num,
            "num_coord": {"x": num_dict[best_num]["x"], "y": num_dict[best_num]["y"]},
            "distance": round(min_dist, 3)
        })
        
        logging.info(f"Paired: Dot {best_dot} ↔ Number {best_num} (distance: {min_dist:.3f}px)")
    
    # Predict missing numbers using neighbor analysis
    if unpaired_dots and missing_numbers:
        print("\nPredicting numbers for unpaired dots...")
        missing_numbers = list(missing_numbers)
        
        while unpaired_dots and missing_numbers:
            best_dot = None
            best_num = None
            best_score = float('inf')
            
            for dot_id in unpaired_dots:
                dot_x, dot_y = dot_dict[dot_id]["x"], dot_dict[dot_id]["y"]
                
                # Find 3 nearest paired neighbors
                neighbor_distances = []
                for pair in pairs:
                    neighbor_x, neighbor_y = pair["dot_coord"]["x"], pair["dot_coord"]["y"]
                    dist = math.hypot(neighbor_x - dot_x, neighbor_y - dot_y)
                    neighbor_distances.append((dist, pair["num_value"]))
                
                neighbor_distances.sort()
                neighbors = [num for _, num in neighbor_distances[:3]]
                
                # Score each missing number based on neighbors
                for missing_num in missing_numbers:
                    score = 0
                    
                    # Check if missing number is between any two neighbors
                    for i in range(len(neighbors) - 1):
                        if min(neighbors[i], neighbors[i+1]) < missing_num < max(neighbors[i], neighbors[i+1]):
                            score = abs(missing_num - neighbors[i]) + abs(missing_num - neighbors[i+1])
                            break
                    
                    # If not between neighbors, use minimum distance + penalty
                    if score == 0:
                        score = min(abs(missing_num - n) for n in neighbors) + 1000
                    
                    if score < best_score:
                        best_score = score
                        best_dot = dot_id
                        best_num = missing_num
            
            if best_dot is None:
                break
            
            unpaired_dots.remove(best_dot)
            missing_numbers.remove(best_num)
            
            pairs.append({
                "dot_id": best_dot,
                "dot_coord": {"x": dot_dict[best_dot]["x"], "y": dot_dict[best_dot]["y"]},
                "num_value": best_num,
                "num_coord": None,
                "distance": None,
                "outlier": True
            })
            
            print(f"Predicted: Dot {best_dot} → Number {best_num} (score: {best_score:.1f})")
    
    # Sort by number
    pairs.sort(key=lambda p: p["num_value"] if isinstance(p["num_value"], int) else 9999)
    
    # Save
    with open(output_path, "w") as f:
        json.dump(pairs, f, indent=4)
    
    logging.info(f"Saved {len(pairs)} pairs to {output_path}")
    return pairs


def visualize_pairs(pairs, output_path, main_image_path=None):
    """
    Create visualization of paired dots and numbers
    Green dots = paired, Orange dots = predicted
    """
    if main_image_path and main_image_path.exists():
        img = cv2.imread(str(main_image_path))
    else:
        max_x = max(p["dot_coord"]["x"] for p in pairs) + 100
        max_y = max(p["dot_coord"]["y"] for p in pairs) + 100
        img = 255 * np.ones((max_y, max_x, 3), dtype=np.uint8)
    
    for pair in pairs:
        dot_x = int(pair["dot_coord"]["x"])
        dot_y = int(pair["dot_coord"]["y"])
        num_value = pair["num_value"]
        predicted = pair.get("outlier", False)
        
        # Color: green if paired, orange if predicted
        color_dot = (0, 255, 0) if not predicted else (0, 165, 255)
        cv2.circle(img, (dot_x, dot_y), 5, color_dot, -1)
        
        # Draw line to number if it was detected
        if pair["num_coord"] is not None:
            num_x = int(pair["num_coord"]["x"])
            num_y = int(pair["num_coord"]["y"])
            cv2.circle(img, (num_x, num_y), 3, (0, 0, 255), -1)
            cv2.line(img, (dot_x, dot_y), (num_x, num_y), (200, 200, 200), 1)
        
        # Label with asterisk if predicted
        label = f"{num_value}{'*' if predicted else ''}"
        cv2.putText(img, label, (dot_x + 10, dot_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    cv2.imwrite(str(output_path), img)
    logging.info(f"Visualization saved to {output_path}")


def matchmaker_main(config, picture_name, expected_range):
    """Main entry point for dot-number matching"""
    logging.basicConfig(level=logging.INFO)

 
    base_path = config['_base_path']
    config_dir = base_path / config['paths']['config_dir']
    
    dot_path = config_dir / config['filenames']['global_dots']
    num_path = config_dir / config['filenames']['global_numbers']
    pairing_path = config_dir / config['filenames']['pairing']
    
    dot_dict = dot_to_dict(dot_path)
    num_dict, outliers, missing_numbers = num_to_dict(num_path, expected_range)
    
    pairs = distance_based_pairing(dot_dict, num_dict, expected_range, missing_numbers, pairing_path)
    
    picture_path = base_path / config['paths']['pictures_dir'] / picture_name
    viz_path = base_path / config['filenames']['pairing_viz']
    
    visualize_pairs(pairs, viz_path, picture_path if picture_path.exists() else None)
    
    logging.info("Matchmaking completed\n")


if __name__ == "__main__":
    matchmaker_main()