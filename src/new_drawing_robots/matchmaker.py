"""
Dot-Number Matchmaker (V7 - Smart Filter by Confidence + "Rule 1")

This script loads the pre-paired numbers from number_detection.py (V9+)
and applies the user's requested filters:
1. "Rule 1": Finds all "1"s, keeps the one with the HIGHEST CONFIDENCE.
   This "1" is protected and ignores the min_confidence_threshold.
2. Duplicate Number Filter: For all other numbers,
   keeps only the one with the HIGHEST CONFIDENCE.
3. Confidence Filter: Discards all remaining numbers below
   'min_confidence_threshold'.
"""

import cv2
import numpy as np
import json
import logging
from pathlib import Path
import sys

def load_json(file_path):
    """Loads JSON data from a file."""
    if not file_path.exists():
        logging.error(f"❌ File not found: {file_path}")
        return None
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logging.error(f"Error loading JSON ({file_path.name}): {e}")
        return None

def visualize_pairing(image_path, dots_map, pairings, output_path):
    """Creates the pairing visualization image."""
    image = cv2.imread(str(image_path))
    if image is None:
        logging.error(f"❌ Failed to load image for visualization: {image_path}")
        return

    # Draw the dots
    for dot_id, dot_info in dots_map.items():
         x = dot_info["global_coordinates"]["x"]
         y = dot_info["global_coordinates"]["y"]
         cv2.circle(image, (x, y), 3, (255, 0, 0), -1) # Blue dot

    # Draw the numbers and lines
    for pair in pairings:
        dot_id = pair['dot_id']
        num_info = pair['number_info']
        
        num_x = num_info['global_coordinates']['x']
        num_y = num_info['global_coordinates']['y']
        
        dot_coord = None
        if dot_id in dots_map:
             dot_coord = (dots_map[dot_id]["global_coordinates"]["x"], dots_map[dot_id]["global_coordinates"]["y"])

        color = (0, 165, 255) # Orange
        if len(num_info.get('methods', [])) > 1:
            color = (0, 255, 0) # Green

        # Draw number
        cv2.circle(image, (num_x, num_y), 5, (0, 0, 255), -1) # Red circle
        label = f"{num_info['number']}"
        cv2.putText(image, label, (num_x + 8, num_y - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(image, label, (num_x + 8, num_y - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw line
        if dot_coord:
            cv2.line(image, dot_coord, (num_x, num_y), (150, 150, 150), 1)

    cv2.imwrite(str(output_path), image)
    logging.info(f"✓ Pairing visualization saved to: {output_path}")


def pair_numbers_to_dots(config, picture_name):
    """
    Main function to FILTER pairings based on Confidence and "Rule 1".
    """
    logging.info("\n" + "="*60)
    logging.info("STEP 4: Dot-Number Matching (Filter by Confidence + 'Rule 1')")
    logging.info("="*60 + "\n")

    base_path = config['_base_path']
    config_dir = base_path / config['paths']['config_dir']

    # Read confidence level from config
    min_conf = config['number_detection'].get('min_confidence_threshold', 60) 
    
    logging.info(f"Minimum confidence threshold (min_confidence_threshold): {min_conf}")

    # 1. Load Dots
    dots_json_path = config_dir / config['filenames']['global_dots']
    dots_data = load_json(dots_json_path)
    if not dots_data or 'circles' not in dots_data:
        logging.error("❌ Error: Failed to load dot data.")
        return
        
    dots_map = {dot['id']: dot for dot in dots_data['circles']}
    logging.info(f"✓ {len(dots_map)} dots loaded.")

    # 2. Load Numbers (already paired by number_detection)
    numbers_json_path = config_dir / config['filenames']['global_numbers']
    numbers_data = load_json(numbers_json_path)
    if not numbers_data or 'numbers' not in numbers_data:
        logging.error("❌ Error: Failed to load number data (with pairings).")
        return
        
    paired_numbers_from_file = numbers_data['numbers']
    logging.info(f"✓ {len(paired_numbers_from_file)} raw pairings loaded (before filtering).")

    # 3. "Rule 1" and Duplicate Filtering (By Confidence)
    best_pair_for_number = {}
    best_one_pair = None
    best_one_conf = -1
    
    # First, search for "1"s
    for num_info in paired_numbers_from_file:
        if num_info['number'] == 1:
            confidence = num_info.get('confidence', 0)
            if confidence > best_one_conf:
                best_one_conf = confidence
                best_one_pair = num_info
                
    if best_one_pair:
        logging.info(f"✓ 'Rule 1': Best '1' selected (Dot: {best_one_pair['dot_id']}, Conf: {best_one_conf}). This pair is protected.")
        best_pair_for_number[1] = best_one_pair
    else:
        logging.info("No '1' detection found.")

    # Now filter OTHER numbers (confidence + duplicate)
    for num_info in paired_numbers_from_file:
        number = num_info['number']
        
        # The '1' has already been handled
        if number == 1:
            continue
            
        confidence = num_info.get('confidence', 0)
        dot_id = num_info['dot_id']
        
        # Confidence filter
        if confidence < min_conf:
            logging.info(f"  -> DISCARDED (low confidence): "
                         f"Dot {dot_id} -> '{number}' (Conf: {confidence} < {min_conf})")
            continue
        
        # Duplicate filter (higher confidence wins)
        if number not in best_pair_for_number:
            best_pair_for_number[number] = num_info
        else:
            existing_pair = best_pair_for_number[number]
            existing_confidence = existing_pair.get('confidence', 0)
            
            if confidence > existing_confidence:
                logging.warning(f"  -> CONFLICT: The number {number} for dot {dot_id} (Conf: {confidence}) REPLACED "
                                f"dot {existing_pair['dot_id']} (Conf: {existing_confidence})")
                best_pair_for_number[number] = num_info
            else:
                logging.warning(f"  -> CONFLICT: The number {number} for dot {dot_id} (Conf: {confidence}) LOST "
                                f"to dot {existing_pair['dot_id']} (Conf: {existing_confidence})")

    final_paired_numbers = list(best_pair_for_number.values())
    logging.info(f"✓ {len(final_paired_numbers)} unique pairings remain after filtering.")

    # 4. Create Pairing List
    pairing_list = []
    found_dot_ids = set()
    for num_info in final_paired_numbers:
        dot_id = num_info.get('dot_id')
        if dot_id not in dots_map:
            logging.warning(f"Number ({num_info['number']}) with invalid dot_id ({dot_id})!")
            continue
        dot_info = dots_map[dot_id]
        pairing_list.append({
            "dot_id": dot_id,
            "dot_coordinates": dot_info["global_coordinates"],
            "number_info": num_info
        })
        found_dot_ids.add(dot_id)
        
    missing_dots = len(dots_map) - len(found_dot_ids)
    logging.info(f"Total of {len(pairing_list)} FINAL pairings created.")
    if missing_dots > 0:
        logging.warning(f"{missing_dots} dots remain unpaired.")

    # 5. Save pairing.json
    pairing_json_path = config_dir / config['filenames']['pairing']
    try:
        with open(pairing_json_path, 'w') as f:
            json.dump({
                "pairing_method": "Iterative Erase (V9) + Conf Filter (V7)",
                "min_confidence_threshold_used": min_conf,
                "total_pairs": len(pairing_list),
                "pairings": pairing_list
            }, f, indent=2)
        logging.info(f"✓ Final pairing list saved to: {pairing_json_path}")
    except Exception as e:
        logging.error(f"Error saving pairing.json: {e}")

    # 6. Create Visualization
    image_path = base_path / config['paths']['pictures_dir'] / picture_name
    viz_output_path = base_path / config['filenames']['pairing_viz']
    visualize_pairing(image_path, dots_map, pairing_list, viz_output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("matchmaker.py (V7 - Smart Filter by Conf + Rule 1) - Ready to be run from orchestrator.py.")