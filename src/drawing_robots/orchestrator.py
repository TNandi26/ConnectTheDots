"""
Main orchestrator for Connect The Dots pipeline
Handles configuration loading, directory setup, and pipeline execution
"""

import json
import logging
from pathlib import Path
import torch
from PIL import Image
from circle_detection import run_dot_detection_for_all_segments
from segment_merge import segment_and_merge_image
from number_detection import run_detection_for_all_segments
from matchmaker import matchmaker_main


def load_config(config_file="config.json"):
    """Load configuration from JSON file"""
    base_path = Path(__file__).parent
    config_path = base_path / config_file
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path) as f:
        config = json.load(f)
    
    config['_base_path'] = base_path
    return config


def get_path(config, key, create=False):
    """Get absolute path from config"""
  
    base_path = config['_base_path']
    rel_path = config['paths'][key]
    abs_path = base_path / rel_path
    
    if create:
        abs_path.mkdir(parents=True, exist_ok=True)
    
    return abs_path


def get_file_path(config, filename_key, subdir='config', create_dir=True):
    """Get full path for a file in output directory """
    

    base_path = config['_base_path']
    filename = config['filenames'][filename_key]
    
    if subdir:
        full_path = base_path / "Output_pictures" / subdir / filename
    else:
        full_path = base_path / "Output_pictures" / filename
    
    if create_dir:
        full_path.parent.mkdir(parents=True, exist_ok=True)
    
    return full_path

def preview_segmentation(config, image_path):
    """Show segment count and allow adjustment of all parameters"""
    
    while True:
        img = Image.open(image_path)
        w, h = img.size
        cols = config['segmentation']['overlap_cols']
        overlap_x = config['segmentation']['overlap_x']
        overlap_y = config['segmentation']['overlap_y']
        
        seg_w = int(w / cols)
        seg_h = int(seg_w * (h / w))
        step_x = int(seg_w * (1 - overlap_x))
        step_y = int(seg_h * (1 - overlap_y))
        
        total = len(range(0, w, step_x)) * len(range(0, h, step_y))
        
        print(f"Segmentation Preview: {total} segments")
        print(f"1. Columns: {cols}")
        print(f"2. Overlap X: {overlap_x*100:.0f}%")
        print(f"3. Overlap Y: {overlap_y*100:.0f}%")
        
        choice = input("Enter number to change (1-3) or 'ok' to continue: ").strip().lower()
        
        if choice == 'ok':
            print(f"Using {total} segments\n")
            break
        elif choice == '1':
            new_val = input(f"Enter columns (current {cols}): ").strip()
            if new_val.isdigit() and int(new_val) > 0:
                config['segmentation']['overlap_cols'] = int(new_val)
        elif choice == '2':
            new_val = input(f"Enter overlap X as decimal (current {overlap_x}): ").strip()
            try:
                val = float(new_val)
                if 0 <= val < 1:
                    config['segmentation']['overlap_x'] = val
            except ValueError:
                print("Invalid input, keeping current value")
        elif choice == '3':
            new_val = input(f"Enter overlap Y as decimal (current {overlap_y}): ").strip()
            try:
                val = float(new_val)
                if 0 <= val < 1:
                    config['segmentation']['overlap_y'] = val
            except ValueError:
                print("Invalid input, keeping current value")
        else:
            print("Invalid choice")
    
    return config


def setup_directories(config):
    """Create all required output directories"""
    directories = [
        'output_dir',
        'config_dir',
        'segments_dir',
        'segments_grid_dir',
        'segments_overlap_dir',
        'dot_viz_dir',
        'number_viz_dir',
        'number_debug_dir'
    ]
    
    logging.info("Setting up directories:")
    for dir_key in directories:
        path = get_path(config, dir_key, create=True)
        logging.info(f"{path.name}")


def select_image(config):
    """Prompt user to select an image"""
    pictures_dir = get_path(config, 'pictures_dir')
    
    if not pictures_dir.exists():
        raise FileNotFoundError(f"Pictures directory not found: {pictures_dir}")
    
    pictures = sorted([
        f for f in pictures_dir.iterdir() 
        if f.suffix.lower() in ['.jpg', '.png', '.jpeg']
    ])
    
    if not pictures:
        raise FileNotFoundError(f"No images found in {pictures_dir}")
    
    logging.info("Available images:")
    for i, picture in enumerate(pictures, 1):
        logging.info(f"  {i}. {picture.name}")
    
    while True:
        try:
            choice = int(input("Enter number: "))
            if 1 <= choice <= len(pictures):
                selected = pictures[choice - 1]
                logging.info(f"Selected: {selected.name}")
                return selected
        except (ValueError, IndexError):
            logging.warning("Invalid selection. Try again.")


def get_expected_range():
    """Prompt user for expected number range"""
    while True:
        try:
            upper_limit = int(input("What is the upper limit of your expected range: "))
            if upper_limit > 0:
                logging.info(f"✓ Expected range: 1-{upper_limit}\n")
                return (1, upper_limit)
        except ValueError:
            logging.warning("Invalid input. Please enter a positive number.")


def run_pipeline(config, image_path, expected_range):
    """Execute the complete detection pipeline"""
    try:
        logging.info("STEP 1: Image Segmentation")
        segment_and_merge_image(config, image_path)
        
        logging.info("STEP 2: Dot Detection")
        run_dot_detection_for_all_segments(config, image_path.name)
        
        logging.info("STEP 3: Number Detection")
        use_combo = config['number_detection']['use_easyocr']
        run_detection_for_all_segments(config, image_path.name, expected_range, use_combo)
        
        logging.info("STEP 4: Dot-Number Matching")
        matchmaker_main(config, image_path.name, expected_range[1])
                
    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        raise


def main():
    """Main entry point"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
        
    config = load_config()

    if torch.cuda.is_available():
        logging.info(f"CUDA available: {torch.cuda.get_device_name(0)}\n")
    else:
        logging.warning("CUDA not available, using CPU\n")
    
    setup_directories(config)
    
    image_path = select_image(config)
    preview_segmentation(config, image_path) 

    expected_range = get_expected_range()
    
    # Run pipeline
    run_pipeline(config, image_path, expected_range)

if __name__ == "__main__":
    main()