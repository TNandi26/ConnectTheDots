import os
import logging
import json
import math
import cv2
import numpy as np


def visualize_pairs(pairs, output_path, main_image_path=None):
    """
    Create visualization of paired dots and numbers.
    pairs: list of dicts containing:
        dot_id, dot_coord {x, y}, num_value, num_coord {x, y}, distance
    main_image_path: optional image file to use as background
    """
    # Prepare base image
    if main_image_path and os.path.exists(main_image_path):
        img = cv2.imread(main_image_path)
    else:
        # Calculate canvas size from coordinates with padding
        max_x = max(p["dot_coord"]["x"] for p in pairs) + 100
        max_y = max(p["dot_coord"]["y"] for p in pairs) + 100
        img = 255 * np.ones((max_y, max_x, 3), dtype=np.uint8)

    for pair in pairs:
        dot_x = int(pair["dot_coord"]["x"])
        dot_y = int(pair["dot_coord"]["y"])
        num_value = pair["num_value"]
        predicted = pair.get("outlier", False)

        color_dot = (0, 255, 0) if not predicted else (0, 165, 255)
        cv2.circle(img, (dot_x, dot_y), 5, color_dot, -1)

        if pair["num_coord"] is not None:
            num_x = int(pair["num_coord"]["x"])
            num_y = int(pair["num_coord"]["y"])
            cv2.circle(img, (num_x, num_y), 3, (0, 0, 255), -1)
            cv2.line(img, (dot_x, dot_y), (num_x, num_y), (200, 200, 200), 1)

        label = f"{num_value}{'*' if predicted else ''}"
        cv2.putText(img, label, (dot_x + 10, dot_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.imwrite(output_path, img)
    logging.info(f"Visualization saved to: {output_path}")


def distance_based_pairing(dot_dict, num_dict, expected_range, missing_numbers, output_path):

    print(f"The dot detection was {((expected_range - abs(len(dot_dict) - expected_range)) / expected_range)*100:.2f}% accurate")
    print(f"The num detection was {((expected_range - abs(len(num_dict) - expected_range)) / expected_range)*100:.2f}% accurate")

    pairs = [] 
    unpaired_dots = set(dot_dict.keys())
    unpaired_nums = set(num_dict.keys())

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

        logging.info(f"Dot {best_dot} paired with number {best_num}, distance: {min_dist:.3f} pixels")

    # Handling outliers and predicting the numbers, it's based on the neighouring numbers
    if unpaired_dots and missing_numbers:
        print("\nPredicting numbers for outliers based on neighbors...")
        missing_numbers = list(missing_numbers)
        
        while unpaired_dots and missing_numbers:
            best_dot = None
            best_num = None
            best_score = float('inf')
            
            for dot_id in unpaired_dots:
                dot_x, dot_y = dot_dict[dot_id]["x"], dot_dict[dot_id]["y"]
                
                neighbor_distances = []
                for pair in pairs:
                    neighbor_x, neighbor_y = pair["dot_coord"]["x"], pair["dot_coord"]["y"]
                    dist = math.hypot(neighbor_x - dot_x, neighbor_y - dot_y)
                    neighbor_distances.append((dist, pair["num_value"]))
                
                neighbor_distances.sort()
                neighbors = [num for _, num in neighbor_distances[:3]]
                
                for missing_num in missing_numbers:
                    score = 0
                    for i in range(len(neighbors) - 1):
                        if min(neighbors[i], neighbors[i+1]) < missing_num < max(neighbors[i], neighbors[i+1]):
                            score = abs(missing_num - neighbors[i]) + abs(missing_num - neighbors[i+1])
                            break
                    
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
            
            print(f"  Predicted: Dot {best_dot} → Number {best_num} (score: {best_score:.1f})")
    
    pairs.sort(key=lambda p: p["num_value"] if isinstance(p["num_value"], int) else 9999)

    with open(output_path, "w") as f:
        json.dump(pairs, f, indent=4)

    return pairs


def num_to_dict(num_path, expected_range):
    try:
        with open(num_path, 'r') as file:
            data = json.load(file)

        result = {}
        next_available = expected_range + 1
        outliers = {}
        seen_counts = {}

        for num in data.get("numbers", []):
            number = num.get("number")
            if number is not None:
                seen_counts[number] = seen_counts.get(number, 0) + 1

        for num in data.get("numbers", []):
            number = num.get("number")
            global_coords = num.get("global_coordinates", {})
            x = global_coords.get("x")
            y = global_coords.get("y")

            if number is None or x is None or y is None:
                continue

            if number > expected_range or seen_counts[number] > 1:
                new_key = next_available
                result[new_key] = {"x": x, "y": y, "outlier": True, "original": number}
                outliers[number] = outliers.get(number, []) + [new_key]
                next_available += 1
                continue

            result[number] = {"x": x, "y": y}

        missing_numbers = []
        for n in range(1, expected_range + 1):
            if n not in seen_counts or seen_counts[n] != 1:
                missing_numbers.append(n)

        if outliers:
            logging.info(f"Outliers detected: {outliers}")
        else:
            logging.info("No outliers found.")

        if missing_numbers:
            logging.info(f"Missing numbers: {missing_numbers}")
        else:
            logging.info(f"No missing numbers between 1-{expected_range}")
        return result, outliers, missing_numbers

    except FileNotFoundError:
        logging.error(f"File not found:  {num_path}")
        return {}
    except json.JSONDecodeError:
        logging.error(f"File not found:  {num_path}")
        return {}

def dot_to_dict(dot_path):
    try:
        with open(dot_path, 'r') as file:
            data = json.load(file)

        result = {}

        for circle in data.get("circles", []):
            id = circle.get("id")
            global_coords = circle.get("global_coordinates", {})
            x = global_coords.get("x")
            y = global_coords.get("y")
            if id is not None and x is not None and y is not None:
                result[id] = {"x": x, "y": y}

        return result

    except FileNotFoundError:
        logging.error(f"File not found:  {dot_path}")
        return {}
    except json.JSONDecodeError:
        logging.error(f"File not found {dot_path}")
        return {}
    
def matchmaker_main(picture_name, expected_range):

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    base_path = os.path.dirname(os.path.abspath(__file__))
    dot_path = os.path.join(base_path, "Output_pictures/config/global_dot_coordinates.json")
    num_path = os.path.join(base_path, "Output_pictures/config/global_numbers.json")
    base_image_path = os.path.join(base_path, f"../../Pictures/{picture_name}")

    dot_dict = dot_to_dict(dot_path)
    num_dict, outliars, missing_numbers = num_to_dict(num_path, expected_range)

    pairs = distance_based_pairing(dot_dict, num_dict, expected_range, missing_numbers, os.path.join(base_path, "Output_pictures/config/pairing.json"))

    visualize_pairs(pairs, os.path.join(base_path, "Output_pictures/pairing_visualization.jpg"), base_image_path)

if __name__ == "__main__":
    matchmaker_main()