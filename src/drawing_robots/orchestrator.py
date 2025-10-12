import os
import logging
from  circle_detection import run_dot_detection_for_all_segments
import pathlib
from segment_merge import main_logic
from number_detection import run_detection_for_all_segments
from matchmaker import matchmaker_main
import torch


def check_folder_status():

    base_path = os.path.dirname(os.path.abspath(__file__))
    output_picture = os.path.join(base_path, "Output_pictures")
    segment_path = os.path.join(base_path, "Segments")

    if not os.path.exists(output_picture):
        logging.info("Output_pictures does not exists, creating it now...")
        os.makedirs(output_picture, exist_ok=True)
    else:
        logging.info("Output picture already exists")
    
    if not os.path.exists(os.path.join(output_picture, "config")):
        logging.info("config does not exists, creating it now...")
        os.makedirs(os.path.join(output_picture, "config"), exist_ok=True)
    else:
        logging.info("config already exists")

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

    if not os.path.exists(os.path.join(segment_path, "Segment_dot_visualizations")):
        logging.info("Segment_dot_visualizations does not exists, creating it now...")
        os.makedirs(os.path.join(segment_path, "Segment_dot_visualizations"), exist_ok=True)
    else:
        logging.info("SegmentsGrid already exists") 

    if not os.path.exists(os.path.join(base_path, "..", "..", "config")):
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
    print(torch.cuda.is_available())
    logging.info("Please select an image to use:")

    images = []
    base_path = os.path.dirname(os.path.abspath(__file__))
    picture_folder = os.path.join(base_path, "../../Pictures")
    pictures = sorted(os.listdir(picture_folder))
    i=0

    for i, picture in enumerate(pictures, start=1):
        images.append(picture)
        logging.info(f"{i}. {picture}")

    picture_number = int(input("Enter number: "))
    logging.info(f"Using {images[picture_number - 1]}")
    picture_path = os.path.join(picture_folder, images[picture_number - 1])
    print(picture_path)
    range = int(input("What is the upper limit of your expected range: "))

    check_folder_status() # Setting up the working directory
    main_logic(images, picture_number, picture_path) # Merging together the pictures, segment_merge.py
    run_detection_for_dots(images[picture_number - 1]) # Run the dot detection algorithm for each segmented image, circle_detection.py
    run_detection_for_all_segments(images[picture_number - 1], expected_range=(1, range), use_combo_ocr=True)
    matchmaker_main(images[picture_number - 1], range)
    