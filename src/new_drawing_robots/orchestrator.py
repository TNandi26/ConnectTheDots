import json
import logging
from pathlib import Path
import torch
from PIL import Image
from dot_detection import run_dot_detection_for_all_segments
from segment_merge import segment_and_merge_image
from number_detection import run_segment_based_detection
from matchmaker import pair_numbers_to_dots


def load_config(config_file="config.json"):
    """Load configuration from JSON file"""
    base_path = Path(__file__).parent
    config_path = base_path / config_file
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    try:
        with open(config_path) as f: 
            config = json.load(f)
        config['_base_path'] = base_path
        return config
    except json.JSONDecodeError as e:
        logging.error(f"Error during reading the config.json ({config_path}) : Invalid JSON format")
        logging.error(e)
        raise
    except Exception as e:
        logging.error(f"Error during loading the config.json: {e}")
        raise

def get_path(config, key, create=False):
    """Get absolute path from config"""
    try:
        base_path = config['_base_path']
        rel_path = config['paths'][key]
        abs_path = base_path / rel_path
        if create: 
            abs_path.mkdir(parents=True, exist_ok=True)
        return abs_path
    except KeyError:
        logging.error(f"Missing key in config 'paths' section: {key}")
        raise
    except Exception as e:
        logging.error(f"Error creating path ('{key}'): {e}")
        raise

def get_file_path(config, filename_key, subdir='config', create_dir=True):
    """Get full path for a file in output directory """
    try:
        base_path = config['_base_path']
        filename = config['filenames'][filename_key]
        if subdir: 
            full_path = base_path / "Output_pictures" / subdir / filename
        else: 
            full_path = base_path / "Output_pictures" / filename
        if create_dir: 
            full_path.parent.mkdir(parents=True, exist_ok=True)
        return full_path
    except KeyError: 
        logging.error(f"Missing key in config 'filenames' section: '{filename_key}'")
        raise
    except Exception as e: 
        logging.error(f"Error creating file path ('{filename_key}'): {e}")
        raise

def preview_segmentation(config, image_path):
    """Show segment count and allow adjustment of all parameters"""
    while True:
        try:
            img = Image.open(image_path)
            w, h = img.size
            cols = config['segmentation']['overlap_cols']
            overlap_x = config['segmentation']['overlap_x']
            overlap_y = config['segmentation']['overlap_y']
            seg_w = int(w / cols)
            seg_h = int(seg_w * (h / w))
            step_x = int(seg_w * (1 - overlap_x))
            step_y = int(seg_h * (1 - overlap_y))
            if step_x <= 0 or step_y <= 0: 
                raise ValueError("Overlap too large resulting in non-positive step size")
            total = len(range(0, w, step_x)) * len(range(0, h, step_y))
        except KeyError as e: 
            logging.error(f"Missing key in config 'segmentation' section: {e}")
            raise
        except FileNotFoundError: 
            logging.error(f"Error: Image not found for preview: {image_path}")
            raise
        except Exception as e: 
            logging.error(f"Error calculating segmentation preview: {e}")
            raise

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
    directories = ['output_dir', 'config_dir', 'segments_dir', 'segments_grid_dir', 'segments_overlap_dir', 'dot_viz_dir', 'number_viz_dir', 'number_debug_dir']
    logging.info("Setting up directories:")
    for dir_key in directories:
        try: 
            path = get_path(config, dir_key, create=True)
            logging.info(f"{path.name}")
        except Exception as e: 
            logging.error(f"Failed to create directory: {dir_key}. Error: {e}")

def select_image(config):
    """Prompt user to select an image"""
    try:
        pictures_dir = get_path(config, 'pictures_dir')
        if not pictures_dir.exists() or not pictures_dir.is_dir(): 
            raise FileNotFoundError(f"Pictures directory not found or not a directory: {pictures_dir}")
        pictures = sorted([ f for f in pictures_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.png', '.jpeg']])
        if not pictures: 
            raise FileNotFoundError(f"No images found in {pictures_dir}")
    except Exception as e: 
        logging.error(f"Error listing images: {e}")
        raise
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
            else: 
                logging.warning("Invalid selection number.")
        except (ValueError, IndexError): 
            logging.warning("Invalid input. Please enter a number.")

def get_expected_range():
    """Prompt user for expected number range"""
    while True:
        try: 
            upper_limit = int(input("What is the upper limit of your expected range: "))
            if upper_limit > 0: 
                logging.info(f"✓ Expected range: 1-{upper_limit}\n")
                return (1, upper_limit)
            else: 
                logging.warning("Please enter a positive number greater than 0.")
        except ValueError: 
            logging.warning("Invalid input. Please enter a number.")


def run_pipeline(config, image_path, expected_range):
    """Execute the complete detection pipeline with checks between steps."""
    
    try:
        dots_json_path = get_file_path(config, 'global_dots', create_dir=False)
        numbers_json_path = get_file_path(config, 'global_numbers', create_dir=False)
        pairing_json_path = get_file_path(config, 'pairing', create_dir=False)
    except Exception as e: 
        logging.error(f"Error reading configuration filenames: {e}")
        return

    try:
        logging.info("STEP 1: Image Segmentation")
        segment_and_merge_image(config, image_path)
        
        logging.info("STEP 2: Dot Detection")
        run_dot_detection_for_all_segments(config, image_path.name, expected_range[1])
        if not dots_json_path.exists():
            logging.error("❌ Error: STEP 2 did not create the 'global_dot_coordinates.json' file.")
            return
        logging.info(f"✓ '{dots_json_path.name}' created successfully.")

        logging.info("STEP 3: Number Detection (Pairing Logic Inside)")
        run_segment_based_detection(config, image_path.name, expected_range)
        if not numbers_json_path.exists():
            logging.error("❌ Error: STEP 3 did not create the 'global_numbers.json' file.")
            return
        logging.info(f"✓ '{numbers_json_path.name}' created successfully.")
        
        # --- STEP 4 RE-ADDED ---
        logging.info("STEP 4: Finalize Pairing and Visualize")
        pair_numbers_to_dots(config, image_path.name)
        # --- STEP 4 END ---

        if not pairing_json_path.exists():
            logging.warning("⚠️ Warning: STEP 4 did not create the 'pairing.json' file.")
        else:
             logging.info(f"✓ '{pairing_json_path.name}' created successfully.")
                
        logging.info("PIPELINE COMPLETED.")
                
    except Exception as e:
        logging.error(f"An error occurred during pipeline execution: {e}", exc_info=True)


def main():
    """Main entry point"""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    try:
        config = load_config()
        try:
            if torch.cuda.is_available(): 
                logging.info(f"CUDA available: {torch.cuda.get_device_name(0)}\n")
            else: 
                logging.warning("CUDA not available, using CPU\n")
        except Exception as e: 
            logging.warning(f"Error checking CUDA: {e}. Assuming CPU is used.")
        setup_directories(config)
        image_path = select_image(config)
        config = preview_segmentation(config, image_path)
        expected_range = get_expected_range()
        run_pipeline(config, image_path, expected_range)
    except FileNotFoundError as e: 
        logging.error(f"Critical file or directory missing: {e}")
    except ValueError as e: 
        logging.error(f"Invalid value: {e}")
    except Exception as e: 
        logging.error(f"Unexpected error in main process: {e}", exc_info=True)

if __name__ == "__main__":
    main()