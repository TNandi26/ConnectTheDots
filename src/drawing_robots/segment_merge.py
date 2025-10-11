from PIL import Image
import os
import json
import logging


def segment_fixed_grid(picture_path, dir_out, cols):
    """
    Splits an image into a fixed number of columns (cols) and auto-calculated rows.
    Saves tiles as tile_r#_c# and writes a JSON file with bounding rectangles.
    """
    img = Image.open(picture_path)
    w, h = img.size
    logging.info(f"Image width={w}, height={h}")

    seg_w = w / cols
    seg_h = seg_w * (h / w)
    rows = int(round(h / seg_h))
    seg_w = int(round(seg_w))
    seg_h = int(round(seg_h))

    os.makedirs(dir_out, exist_ok=True)
    metadata = {}

    for r in range(rows):
        for c in range(cols):
            left = int(round(c * seg_w))
            upper = int(round(r * seg_h))
            right = min(left + seg_w, w)
            lower = min(upper + seg_h, h)

            box = (left, upper, right, lower)
            out_name = f"tile_r{r}_c{c}.jpg"
            out_path = os.path.join(dir_out, out_name)
            img.crop(box).save(out_path)

            metadata[out_name] = {"start": {"x": left, "y": upper}, "end": {"x": right, "y": lower}}

    meta_path = os.path.join(dir_out, "grid-segments.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logging.info(f"Saved {len(metadata)} segments and metadata to {meta_path}")



def segment_with_overlap(picture_path, dir_out, cols, overlap_x=0.5, overlap_y=0.5):
    """
    Splits an image into overlapping segments.
    Segment size is determined by fixed grid with 'cols' columns.
    Tiles are saved as tile_r#_c# and a JSON file with bounding rectangles is saved.
    """
    img = Image.open(picture_path)
    w, h = img.size

    seg_w = int(round(w / cols))
    seg_h = int(round(seg_w * (h / w)))
    step_x = int(seg_w * (1 - overlap_x))
    step_y = int(seg_h * (1 - overlap_y))
    if step_x <= 0 or step_y <= 0:
        raise ValueError("Overlap too large")

    os.makedirs(dir_out, exist_ok=True)
    metadata = {}
    r = 0
    y_positions = list(range(0, h, step_y))
    for y in y_positions:
        c = 0
        x_positions = list(range(0, w, step_x))
        for x in x_positions:
            left = x
            upper = y
            right = min(x + seg_w, w)
            lower = min(y + seg_h, h)
            box = (left, upper, right, lower)

            out_name = f"tile_r{r}_c{c}.jpg"
            out_path = os.path.join(dir_out, out_name)
            img.crop(box).save(out_path)

            metadata[out_name] = {"start": {"x": left, "y": upper}, "end": {"x": right, "y": lower}}
            c += 1
        r += 1

    meta_path = os.path.join(dir_out, "overlap_segments.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logging.info(f"Saved {len(metadata)} overlapping segments and metadata to {meta_path}")


def merge_tiles_from_json(dir_in, out_path, json_file):
    """
    Merge tiles using their saved start/end pixel coordinates in segments.json.
    Works for overlapping tiles.
    """
    with open(os.path.join(dir_in, json_file), "r") as f:
        metadata = json.load(f)

    if not metadata:
        raise ValueError("No metadata found in JSON")

    max_x, max_y = 0, 0
    for tile_name, info in metadata.items():
        end_x = info["end"]["x"]
        end_y = info["end"]["y"]
        max_x = max(max_x, end_x)
        max_y = max(max_y, end_y)

    merged = Image.new("RGB", (max_x, max_y))

    for tile_name, info in metadata.items():
        tile_path = os.path.join(dir_in, tile_name)
        if not os.path.exists(tile_path):
            continue
        tile = Image.open(tile_path)
        start_x = info["start"]["x"]
        start_y = info["start"]["y"]
        merged.paste(tile, (start_x, start_y))

    merged.save(out_path)
    logging.info(f"Merged image saved as {out_path} ({max_x}×{max_y})")

def main_logic(images, picture_number, picture_path):

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    base_path = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_path, "Output_pictures")
    segments_path = os.path.join(output_path, "Segments")


    # Fixed grid
    segment_fixed_grid(picture_path, os.path.join(segments_path, "SegmentsGrid"), cols=5)
    # Overlap segments
    segment_with_overlap(picture_path, os.path.join(segments_path, "SegmentsOverlap"), cols=3, overlap_x=0.5, overlap_y=0.5)
    # Merge overlapping tiles
    merge_tiles_from_json(os.path.join(segments_path, "SegmentsOverlap"), os.path.join(output_path, "merged_result.jpg"), os.path.join(segments_path, "SegmentsOverlap/overlap_segments.json"))
    
if __name__ == "__main__":
    main_logic()