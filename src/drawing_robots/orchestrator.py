import os
import logging
from  circle_detection import run_dot_detection_for_all_segments
import pathlib
from segment_merge import main_logic
from number_detection import run_detection_for_all_segments

def check_folder_status():

    base_path = os.path.dirname(os.path.abspath(__file__))
    output_picture = os.path.join(base_path, "Output_pictures")
    segment_path = os.path.join(base_path, "Segments")

    if not os.path.exists(output_picture):
        logging.info("Output_pictures does not exists, creating it now...")
        os.makedirs(output_picture, exist_ok=True)
    else:
        logging.info("Output picture already exists")

    if not os.path.exists(segment_path):
        logging.info("Segments does not exists, creating it now...")
        os.makedirs(segment_path, exist_ok=True)
    else:
        logging.info("Segments already exists")

    if not os.path.exists(os.path.join(segment_path, "SegmentsOverlap")):
        logging.info("SegmentsOverlap does not exists, creating it now...")
        os.makedirs(os.path.join(segment_path, "SegmentsOverlap"), exist_ok=True)
    else:
        logging.info("SegmentsOverlap already exists")

    if not os.path.exists(os.path.join(segment_path, "SegmentsGrid")):
        logging.info("SegmentsGrid does not exists, creating it now...")
        os.makedirs(os.path.join(segment_path, "SegmentsGrid"), exist_ok=True)
    else:
        logging.info("SegmentsGrid already exists")

    if not os.path.exists(os.path.join(base_path, "..", "config")):
        logging.info("config does not exists, creating it now...")
        os.makedirs(os.path.join(base_path, "..", "..", "config"), exist_ok=True)
    else:
        logging.info("Config already exists")


def run_detection_for_dots(picture_name):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Segments/SegmentsOverlap")

    count_segments = len(list(pathlib.Path(path).glob("*.jpg")))
    logging.info(f"There are {count_segments} segments for detection")

    folder = pathlib.Path(path)
    jpg_files = list(folder.glob("*.jpg"))

    run_dot_detection_for_all_segments(picture_name)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    check_folder_status() # Setting up the working directory
    picture_name = main_logic() # Merging together the pictures, segment_merge.py
    run_detection_for_dots(picture_name) # Run the dot detection algorithm for each segmented image, circle_detection.py
    exit()
    run_detection_for_all_segments(picture_name, expected_range=(1, 10), use_combo_ocr=True)
    