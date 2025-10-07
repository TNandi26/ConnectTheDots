import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from pathlib import Path
import sys
from PIL import Image
import math


def load_image(image_path):
    """
        Load image from file
        Args: image_path: Path to the image that will be loaded in
        Returns: The loaded in image
    """
    image = cv2.imread(str(image_path))
    if os.path.exists(image_path) is None:
        logging.error(f"Path is not valid: {image_path}")

    if image is None:
        logging.error(f"Could not load image from {image_path}")
        return None
    
    logging.info(f"Loaded image: {image.shape}")
    return image


def convert_to_grayscale(image):
    """
        Convert BGR image to grayscale
        Args: image: An image that will be converted to grayscale
        Returns: Grayscale image
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    logging.info(f"Converted to grayscale: {gray_image.shape}")
    return gray_image


def apply_blur(image, kernel_size=7):
    """
        Apply Gaussian blur to reduce noise - INCREASED for smoother edges
        Args: image: An image that will be blurred
              kernel_size: Kernel size sets the width and height of the neighborhood used for blurring, where larger values mean stronger smoothing
    """
    if kernel_size > 0:
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        logging.info(f"Applied Gaussian blur with kernel size {kernel_size}")
        return blurred
    else:
        logging.info("Skipping blur step")
        return image
    

def apply_adaptive_threshold(image):
    """
        Apply threshold to locate the A4 paper, 
            where the first number in cv2.threshold is the pixel value where if any pixel value is above that it will be white,
            if it's lower than that it will be black
        Args: image: A grayscale image that will be used for thresholding
        Returns: Image that is black and white
    """
    _, threshold = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
    
    return threshold


def detect_a4_contour(threshold_image):
    """Detect the A4 paper contour from threshold image"""
    # Find contours
    contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        logging.error("No contours found")
        return None
    
    # Find largest contour by area
    image_area = threshold_image.shape[0] * threshold_image.shape[1]
    
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        
        # Must be significant portion of image (A4 paper)
        if area > image_area * 0.15:  # At least 15% of image
            logging.info(f"Found A4 contour with area: {area}")
            return contour
    
    logging.error("No suitable A4 contour found")
    return None

def create_mask_from_contour(contour, image_shape):
    """Create binary mask from contour"""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    if contour is not None:
        # Fill the contour area with white (255)
        cv2.fillPoly(mask, [contour], 255)
        logging.info("Mask created successfully")
    
    return mask


def apply_mask_to_original(original_image, mask):
    """Apply mask to original image to keep only paper area"""
    if len(original_image.shape) == 3:
        # 3-channel image - convert mask to 3 channels
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        masked_result = cv2.bitwise_and(original_image, mask_3ch)
    else:
        # Grayscale image
        masked_result = cv2.bitwise_and(original_image, original_image, mask=mask)
    
    logging.info("Mask applied to original image")
    return masked_result


def visualize_contour(original_image, contour):
    """Draw contour on original image for visualization"""
    vis_image = original_image.copy()
    
    if contour is not None:
        cv2.drawContours(vis_image, [contour], -1, (0, 255, 0), 3)
        logging.info("Contour visualization created")
    
    return vis_image

def crop_paper_only(original_image, contour):
    """Crop image to show only the paper area with minimal background"""
    if contour is None:
        return original_image
    
    # Get bounding rectangle of the contour
    x, y, w, h = cv2.boundingRect(contour)
    
    # Add small padding around the paper
    padding = 20
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(original_image.shape[1] - x, w + 2*padding)
    h = min(original_image.shape[0] - y, h + 2*padding)
    
    # Crop the image
    cropped = original_image[y:y+h, x:x+w]
    
    logging.info(f"Cropped paper to size: {cropped.shape}")
    return cropped


def crop_paper_tight(original_image, contour):
    """Crop image using exact contour bounds (tighter crop)"""
    if contour is None:
        return original_image
    
    # Get exact contour bounds
    x, y, w, h = cv2.boundingRect(contour)
    
    # Crop exactly to contour bounds
    cropped = original_image[y:y+h, x:x+w]
    
    logging.info(f"Tight cropped paper to size: {cropped.shape}")
    return cropped


def save_step(image, step_name, step_number):
    """Save image for each processing step"""

    filename = f"step_{step_number:02d}_{step_name}.jpg"
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Output_pictures")

    os.makedirs(output_path, exist_ok=True) #Create it if it doesnt exists
    picture_folder = os.path.join(output_path, filename)

    cv2.imwrite(picture_folder, image)
    logging.info(f"Saved: {filename}")


def check_resolution(image):
    img = Image.open(image)
    width, height = img.size
    ratio = width/height

    if 0.95*ratio <= 1/math.sqrt(2) <=1.05*ratio:
        logging.info(f"A4 paper is detected with good ratio: {ratio}") 
    else:
        logging.error(f"A4 paper ratio is not ideal, please adjust the camera, ratio is {ratio}")
        

def locate_corners_white_paper(path_to_image):
    """
    Detect corner coordinates of white paper on dark background
    Returns: (result_image, corner_coordinates_list)
    """
    image = cv2.imread(path_to_image)
    if image is None:
        logging.error(f"Could not load image from {path_to_image}")
        return None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Binary threshold - extract white paper
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        logging.error("No contours found for corner detection")
        return image, []
    
    # Get largest contour (the paper)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Approximate contour to 4 points (rectangle)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Try different epsilon values if we don't get exactly 4 points
    if len(approx) != 4:
        for eps_factor in [0.01, 0.03, 0.05, 0.08]:
            epsilon = eps_factor * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            if len(approx) == 4:
                break
    
    # Extract coordinates
    corners = []
    for point in approx:
        x, y = point[0]
        corners.append((x, y))
    
    # Sort corners: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    if len(corners) == 4:
        corners = sort_corners_clockwise(corners)
        logging.info(f"Found 4 corners: TL{corners[0]}, TR{corners[1]}, BR{corners[2]}, BL{corners[3]}")
    else:
        logging.warning(f"Found {len(corners)} corners instead of 4: {corners}")
    
    # Draw corners on image - only circles, no text
    result_image = image.copy()
    for i, corner in enumerate(corners):
        x, y = corner
        cv2.circle(result_image, (x, y), 8, (0, 0, 255), -1)  # Red circles only
    
    return result_image, corners


def sort_corners_clockwise(corners):
    """Sort corners in clockwise order: Top-Left, Top-Right, Bottom-Right, Bottom-Left"""
    if len(corners) != 4:
        return corners
    
    # Calculate center point
    center_x = sum(x for x, y in corners) / 4
    center_y = sum(y for x, y in corners) / 4
    
    # Classify corners by position relative to center
    def classify_corner(corner):
        x, y = corner
        if x < center_x and y < center_y:
            return 0  # Top-left
        elif x >= center_x and y < center_y:
            return 1  # Top-right
        elif x >= center_x and y >= center_y:
            return 2  # Bottom-right
        else:
            return 3  # Bottom-left
    
    # Sort corners
    sorted_corners = [None] * 4
    for corner in corners:
        idx = classify_corner(corner)
        sorted_corners[idx] = corner
    
    # Handle any None values (fallback)
    for i in range(4):
        if sorted_corners[i] is None:
            remaining_corners = [c for c in corners if c not in sorted_corners]
            if remaining_corners:
                sorted_corners[i] = remaining_corners[0]
    
    return [corner for corner in sorted_corners if corner is not None]


def apply_perspective_transform(image, corners, output_width=595, output_height=842):
    """
    Apply perspective transformation to correct document perspective
    Standard A4 size: 595x842 pixels (roughly 210x297mm at 72 DPI)
    """
    if len(corners) != 4:
        logging.error(f"Need exactly 4 corners for perspective transform, got {len(corners)}")
        return image
    
    # Source points (detected corners)
    src_points = np.float32(corners)
    
    # Destination points (perfect rectangle)
    dst_points = np.float32([
        [0, 0],                           # Top-left
        [output_width - 1, 0],            # Top-right
        [output_width - 1, output_height - 1],  # Bottom-right
        [0, output_height - 1]            # Bottom-left
    ])
    
    # Calculate perspective transformation matrix
    transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    logging.info("Perspective transformation matrix calculated")
    
    # Apply transformation
    corrected_image = cv2.warpPerspective(image, transform_matrix, (output_width, output_height))
    
    logging.info(f"Applied perspective transformation: {image.shape} -> ({output_width}, {output_height})")
    return corrected_image


def calculate_optimal_output_size(corners):
    """
    Calculate optimal output size based on corner distances to maintain aspect ratio
    """
    if len(corners) != 4:
        logging.warning("Cannot calculate optimal size without 4 corners, using default A4")
        return 595, 842  # Default A4 size
    
    # Calculate distances
    top_width = np.linalg.norm(np.array(corners[1]) - np.array(corners[0]))
    bottom_width = np.linalg.norm(np.array(corners[2]) - np.array(corners[3]))
    left_height = np.linalg.norm(np.array(corners[3]) - np.array(corners[0]))
    right_height = np.linalg.norm(np.array(corners[2]) - np.array(corners[1]))
    
    # Use average dimensions
    avg_width = int((top_width + bottom_width) / 2)
    avg_height = int((left_height + right_height) / 2)
    
    logging.info(f"Calculated optimal size: {avg_width}x{avg_height}")
    return avg_width, avg_height


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    logging.info("Please select an image to use:")

    base_path = os.path.dirname(os.path.abspath(__file__))
    picture_folder = os.path.join(base_path, "Pictures")
    pictures = os.listdir(picture_folder)
    i=0
    for picture in pictures:
        i+=1
        logging.info(f"{i}. {picture}")
    picture_number = int(input())
    
    base_image = load_image(os.path.join(picture_folder, pictures[picture_number-1]))
    save_step(base_image, "original", 1)

    grayscale_image = convert_to_grayscale(base_image)
    save_step(grayscale_image, "grayscale", 2)

    blurred_image = apply_blur(grayscale_image)
    save_step(blurred_image, "blurred", 3)

    threshold_image = apply_adaptive_threshold(blurred_image)
    save_step(threshold_image, "threshold", 4)

    a4_contour = detect_a4_contour(threshold_image)
    save_step(threshold_image, "contour", 5)
    
    if a4_contour is not None:
        logging.info("A4 paper detected successfully!")
        
        paper_mask = create_mask_from_contour(a4_contour, threshold_image.shape)
        save_step(paper_mask, "paper_mask", 6)
        
        masked_original = apply_mask_to_original(base_image, paper_mask)
        save_step(masked_original, "masked_original", 7)
        
        # Visualize detected contour
        contour_visualization = visualize_contour(base_image, a4_contour)
        save_step(contour_visualization, "contour_detected", 8)

        # Corner detection with coordinates
        output_picture_folder = os.path.join(base_path, "Output_pictures/step_07_masked_original.jpg")
        image_corner, corner_coordinates = locate_corners_white_paper(output_picture_folder)
        save_step(image_corner, "corners", 14)
        # Log corner coordinates for use in other parts of code
        if len(corner_coordinates) == 4:
            logging.info("=== CORNER COORDINATES ===")
            logging.info(f"Top-Left corner: {corner_coordinates[0]}")
            logging.info(f"Top-Right corner: {corner_coordinates[1]}")
            logging.info(f"Bottom-Right corner: {corner_coordinates[2]}")
            logging.info(f"Bottom-Left corner: {corner_coordinates[3]}")
            logging.info("===========================")

        # Apply standard A4 transformation

        logging.info("Applying standard A4 perspective transform...")
        corrected_standard = apply_perspective_transform(masked_original, corner_coordinates)
        save_step(corrected_standard, "perspective_A4", 9)
        
        # Apply adaptive transformation (maintains original proportions)
        logging.info("Applying perspective transform...")

        # Also apply transform to grayscale for better processing
        grayscale_cropped = convert_to_grayscale(masked_original)
        save_step(grayscale_cropped, "before_transform_gray", 10)
        
        corrected_gray_standard = apply_perspective_transform(grayscale_cropped, corner_coordinates)
        save_step(corrected_gray_standard, "perspective_A4_gray", 11)
        
        corrected_image = apply_perspective_transform(masked_original, corner_coordinates, output_width=1240, output_height=1754)
        save_step(corrected_image, "smaller_resolution", 12)

        corrected_image = apply_perspective_transform(masked_original, corner_coordinates, output_width=1654, output_height=2339)
        save_step(corrected_image, "bigger_resolution", 13)

    else:
        logging.error("FAILED: Could not detect A4 paper")
        logging.error("Try adjusting the area threshold in detect_a4_contour function")
        sys.exit()