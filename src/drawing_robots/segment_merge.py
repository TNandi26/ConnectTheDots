"""
Image Segmentation Module
Splits images into overlapping tiles for parallel processing
"""

from PIL import Image
import json
import logging
from pathlib import Path


def segment_and_merge_image(config, picture_path):
    """
    Main entry point for segmentation
    Creates both grid and overlapping segments
    """
    base_path = config['_base_path']
    segments_grid_dir = get_path(config, 'segments_grid_dir')
    segments_overlap_dir = get_path(config, 'segments_overlap_dir')
    output_dir = get_path(config, 'output_dir')
    
    # Get segmentation parameters from config
    grid_cols = config['segmentation']['grid_cols']
    overlap_cols = config['segmentation']['overlap_cols']
    overlap_x = config['segmentation']['overlap_x']
    overlap_y = config['segmentation']['overlap_y']
    
    # Fixed grid segmentation
    logging.info(f"Creating fixed grid: {grid_cols} columns")
    segment_fixed_grid(picture_path, segments_grid_dir, grid_cols, config)
    
    # Overlapping segmentation
    logging.info(f"Creating overlapping segments: {overlap_cols} columns, {overlap_x*100}% overlap")
    segment_with_overlap(picture_path, segments_overlap_dir, overlap_cols, overlap_x, overlap_y, config)
    
    # Merge tiles for verification
    overlap_meta = segments_overlap_dir / config['filenames']['overlap_segments_meta']
    merged_output = output_dir / config['filenames']['merged_result']
    merge_tiles_from_json(segments_overlap_dir, merged_output, overlap_meta)
    
    logging.info("Segmentation completed\n")


def segment_fixed_grid(picture_path, dir_out, cols, config):
    """
    Splits an image into a fixed grid
    Saves tiles as tile_r#_c# and writes JSON metadata
    """
    img = Image.open(picture_path)
    w, h = img.size
    logging.info(f"  Image dimensions: {w}x{h}")

    seg_w = w / cols
    seg_h = seg_w * (h / w)
    rows = int(round(h / seg_h))
    seg_w = int(round(seg_w))
    seg_h = int(round(seg_h))

    dir_out.mkdir(parents=True, exist_ok=True)
    metadata = {}

    for r in range(rows):
        for c in range(cols):
            left = int(round(c * seg_w))
            upper = int(round(r * seg_h))
            right = min(left + seg_w, w)
            lower = min(upper + seg_h, h)

            box = (left, upper, right, lower)
            out_name = f"tile_r{r}_c{c}.jpg"
            out_path = dir_out / out_name
            img.crop(box).save(out_path)

            metadata[out_name] = {
                "start": {"x": left, "y": upper},
                "end": {"x": right, "y": lower}
            }

    meta_path = dir_out / config['filenames']['grid_segments_meta']
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logging.info(f"Saved {len(metadata)} grid segments")


def segment_with_overlap(picture_path, dir_out, cols, overlap_x, overlap_y, config):
    """
    Splits an image into overlapping segments
    Saves tiles as tile_r#_c# and writes JSON metadata
    """
    img = Image.open(picture_path)
    w, h = img.size

    seg_w = int(round(w / cols))
    seg_h = int(round(seg_w * (h / w)))
    step_x = int(seg_w * (1 - overlap_x))
    step_y = int(seg_h * (1 - overlap_y))
    
    if step_x <= 0 or step_y <= 0:
        raise ValueError("Overlap too large - adjust overlap_x/overlap_y in config")

    dir_out.mkdir(parents=True, exist_ok=True)
    metadata = {}
    r = 0
    
    for y in range(0, h, step_y):
        c = 0
        for x in range(0, w, step_x):
            left = x
            upper = y
            right = min(x + seg_w, w)
            lower = min(y + seg_h, h)
            box = (left, upper, right, lower)

            out_name = f"tile_r{r}_c{c}.jpg"
            out_path = dir_out / out_name
            img.crop(box).save(out_path)

            metadata[out_name] = {
                "start": {"x": left, "y": upper},
                "end": {"x": right, "y": lower}
            }
            c += 1
        r += 1

    meta_path = dir_out / config['filenames']['overlap_segments_meta']
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logging.info(f"Saved {len(metadata)} overlapping segments")


def merge_tiles_from_json(dir_in, out_path, json_file):
    """
    Merge tiles using saved coordinates from JSON metadata
    Used for verification that segmentation worked correctly
    """
    with open(json_file, "r") as f:
        metadata = json.load(f)

    if not metadata:
        raise ValueError("No metadata found in JSON")

    # Calculate canvas size
    max_x = max(info["end"]["x"] for info in metadata.values())
    max_y = max(info["end"]["y"] for info in metadata.values())

    merged = Image.new("RGB", (max_x, max_y))

    for tile_name, info in metadata.items():
        tile_path = dir_in / tile_name
        if not tile_path.exists():
            continue
        
        tile = Image.open(tile_path)
        start_x = info["start"]["x"]
        start_y = info["start"]["y"]
        merged.paste(tile, (start_x, start_y))

    merged.save(out_path)
    logging.info(f"Merged verification image: {max_x}×{max_y}")


def get_path(config, key):
    """Get absolute path from config"""
    return config['_base_path'] / config['paths'][key]