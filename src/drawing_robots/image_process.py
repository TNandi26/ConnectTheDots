import cv2
import numpy as np
import os
import logging
import sys
from PIL import Image
import math
from itertools import product
from orchestrator import load_config


def load_image(image_path):
    """Load image from file"""
    image = cv2.imread(str(image_path))
    if os.path.exists(image_path) is None:
        logging.error(f"Path is not valid: {image_path}")

    if image is None:
        logging.error(f"Could not load image from {image_path}")
        return None
    
    logging.info(f"Loaded image: {image.shape}")
    return image


def convert_to_grayscale(image):
    """Convert BGR image to grayscale"""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    logging.info(f"Converted to grayscale: {gray_image.shape}")
    return gray_image


def apply_blur(image, kernel_size=7):
    """Apply Gaussian blur to reduce noise"""
    if kernel_size > 0:
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        logging.info(f"Applied Gaussian blur with kernel size {kernel_size}")
        return blurred
    else:
        logging.info("Skipping blur step")
        return image


def apply_preprocessing(image_gray, use_heavy=True):
    """
    Apply preprocessing to enhance image quality
    Set use_heavy=False to skip preprocessing
    """
    if not use_heavy:
        logging.info("Skipping preprocessing (using original)")
        return image_gray
    
    logging.info("Applying HEAVY preprocessing...")
    
    # Step 1: Remove gradient lighting, preserve blacks
    logging.info(f"  Removing gradients (blur size: {GRADIENT_BLUR_SIZE})")
    background = cv2.GaussianBlur(image_gray, (GRADIENT_BLUR_SIZE, GRADIENT_BLUR_SIZE), 0)

    # Compute normalized but protect dark regions
    normalized = cv2.divide(image_gray, background + 1, scale=255)

    # Preserve dark pixels (if pixel is dark in original, keep it)
    mask_dark = image_gray < 60  # threshold can be tuned
    normalized[mask_dark] = image_gray[mask_dark]

    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    # Step 2: Contrast enhancement (mild CLAHE)
    logging.info(f"  Enhancing contrast (strength: {CONTRAST_STRENGTH})")
    clahe = cv2.createCLAHE(clipLimit=CONTRAST_STRENGTH, tileGridSize=(5,5))
    enhanced = clahe.apply(normalized)

    # Step 3: Reduce noise but keep edges sharp
    logging.info(f"  Reducing noise (filter size: {NOISE_REDUCTION})")
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)

    # Step 4: Gentle sharpening
    logging.info("  Sharpening edges")
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)

    result = np.clip(sharpened, 0, 255).astype(np.uint8)
    logging.info("Preprocessing complete!")

    return result


def apply_adaptive_threshold(image, threshold):
    """Apply threshold to locate the A4 paper"""
    _, threshold = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return threshold


def detect_a4_contour(threshold_image):
    """Detect the A4 paper contour from threshold image"""
    contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        logging.error("No contours found")
        return None
    
    image_area = threshold_image.shape[0] * threshold_image.shape[1]
    
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area > image_area * 0.15:
            logging.info(f"Found A4 contour with area: {area}")
            return contour
    
    logging.error("No suitable A4 contour found")
    return None


def create_mask_from_contour(contour, image_shape):
    """Create binary mask from contour"""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    if contour is not None:
        cv2.fillPoly(mask, [contour], 255)
        logging.info("Mask created successfully")
    return mask


def apply_mask_to_original(original_image, mask):
    """Apply mask to original image to keep only paper area"""
    if len(original_image.shape) == 3:
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        masked_result = cv2.bitwise_and(original_image, mask_3ch)
    else:
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
    
    x, y, w, h = cv2.boundingRect(contour)
    padding = 20
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(original_image.shape[1] - x, w + 2*padding)
    h = min(original_image.shape[0] - y, h + 2*padding)
    
    cropped = original_image[y:y+h, x:x+w]
    logging.info(f"Cropped paper to size: {cropped.shape}")
    return cropped


def crop_paper_tight(original_image, contour):
    """Crop image using exact contour bounds (tighter crop)"""
    if contour is None:
        return original_image
    
    x, y, w, h = cv2.boundingRect(contour)
    cropped = original_image[y:y+h, x:x+w]
    logging.info(f"Tight cropped paper to size: {cropped.shape}")
    return cropped


def save_step(image, step_name, step_number, output_dir="Output_pictures"):
    """Save image for each processing step"""
    filename = f"step_{step_number:02d}_{step_name}.jpg"
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_dir)
    os.makedirs(output_path, exist_ok=True)
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
    Detect corners of WHITE PAPER
    """
    image = cv2.imread(path_to_image)
    if image is None:
        logging.error(f"Could not load image from {path_to_image}")
        return None, []
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # CRITICAL FIX: Detect WHITE paper by looking for bright regions
    # Use higher threshold to find white paper edges, not black background

    _, binary = cv2.threshold(gray, PAPER_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    # Clean up the binary image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Find contours on WHITE regions (paper)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        logging.error("No contours found for corner detection")
        return image, []
    
    # Get largest WHITE contour (the paper)
    largest_contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(largest_contour)
    image_area = gray.shape[0] * gray.shape[1]
    
    logging.info(f"Paper contour area: {contour_area} ({contour_area/image_area*100:.1f}% of image)")
    
    # Approximate to 4 corners
    corners = None
    for epsilon_factor in [CORNER_EPSILON, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05]:
        epsilon = epsilon_factor * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        if len(approx) == 4:
            corners = [(int(p[0][0]), int(p[0][1])) for p in approx]
            logging.info(f"Found 4 corners with epsilon: {epsilon_factor}")
            break
    
    # Fallback: use bounding rectangle corners
    if corners is None or len(corners) != 4:
        logging.warning("Could not find 4 corners, using bounding rectangle")
        x, y, w, h = cv2.boundingRect(largest_contour)
        corners = [
            (x, y),           # Top-left
            (x + w, y),       # Top-right
            (x + w, y + h),   # Bottom-right
            (x, y + h)        # Bottom-left
        ]
    
    # Sort corners properly
    corners = sort_corners_clockwise(corners)
    
    logging.info(f"Final corners:")
    logging.info(f"  TL: {corners[0]}")
    logging.info(f"  TR: {corners[1]}")
    logging.info(f"  BR: {corners[2]}")
    logging.info(f"  BL: {corners[3]}")
    
    # Draw corners on image
    result_image = image.copy()
    for i, corner in enumerate(corners):
        x, y = corner
        cv2.circle(result_image, (x, y), 3, (0, 0, 255), -1)
        cv2.putText(result_image, str(i+1), (x+5, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return result_image, corners


def sort_corners_clockwise(corners):
    """
    Sort corners: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    """
    if len(corners) != 4:
        return corners
    
    # Convert to numpy array
    points = np.array(corners, dtype=np.float32)
    
    # Sort by y-coordinate (top vs bottom)
    sorted_by_y = points[np.argsort(points[:, 1])]
    
    # Top 2 points (smaller y)
    top_points = sorted_by_y[:2]
    # Bottom 2 points (larger y)
    bottom_points = sorted_by_y[2:]
    
    # Sort top points by x (left vs right)
    top_left = top_points[np.argmin(top_points[:, 0])]
    top_right = top_points[np.argmax(top_points[:, 0])]
    
    # Sort bottom points by x (left vs right)
    bottom_left = bottom_points[np.argmin(bottom_points[:, 0])]
    bottom_right = bottom_points[np.argmax(bottom_points[:, 0])]
    
    # Return in correct order
    result = [
        tuple(map(int, top_left)),
        tuple(map(int, top_right)),
        tuple(map(int, bottom_right)),
        tuple(map(int, bottom_left))
    ]
    
    return result


def apply_perspective_transform(image, corners, output_width=595, output_height=842):
    """Apply perspective transformation to correct document perspective"""
    if len(corners) != 4:
        logging.error(f"Need exactly 4 corners for perspective transform, got {len(corners)}")
        return image
    
    src_points = np.float32(corners)
    dst_points = np.float32([
        [0, 0],
        [output_width - 1, 0],
        [output_width - 1, output_height - 1],
        [0, output_height - 1]
    ])
    
    transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    logging.info("Perspective transformation matrix calculated")
    corrected_image = cv2.warpPerspective(image, transform_matrix, (output_width, output_height))
    logging.info(f"Applied perspective transformation: {image.shape} -> ({output_width}, {output_height})")
    return corrected_image


def calculate_optimal_output_size(corners):
    """Calculate optimal output size based on corner distances to maintain aspect ratio"""
    if len(corners) != 4:
        logging.warning("Cannot calculate optimal size without 4 corners, using default A4")
        return 595, 842
    
    top_width = np.linalg.norm(np.array(corners[1]) - np.array(corners[0]))
    bottom_width = np.linalg.norm(np.array(corners[2]) - np.array(corners[3]))
    left_height = np.linalg.norm(np.array(corners[3]) - np.array(corners[0]))
    right_height = np.linalg.norm(np.array(corners[2]) - np.array(corners[1]))
    
    avg_width = int((top_width + bottom_width) / 2)
    avg_height = int((left_height + right_height) / 2)
    
    logging.info(f"Calculated optimal size: {avg_width}x{avg_height}")
    return avg_width, avg_height


def fix_lighting_inconsistency(image, blur_size=51, clip_limit=2.0, gamma=0.9):
    """
    Kiegyenlíti a világítási inkonzisztenciákat egy képen.
    Paraméterek:
        image: bemeneti kép (szürkeárnyalatos vagy színes)
        blur_size: a háttér kisimítására használt GaussianBlur kernel méret (páratlan szám)
        clip_limit: CLAHE kontrasztjavítás erőssége (2–4 közötti javasolt)
        gamma: gamma korrekció (0.7–1.2 tartományban javasolt)
        
    Visszatér:
        result: világításban kiegyenlített, kontrasztos kép (uint8)
    """

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    background = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    normalized = cv2.divide(gray, background, scale=255)
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
    contrast = clahe.apply(normalized)

    img_float = contrast.astype(np.float32) / 255.0
    gamma_corrected = np.power(img_float, gamma)
    gamma_corrected = np.clip(gamma_corrected * 255, 0, 255).astype(np.uint8)

    result = cv2.fastNlMeansDenoising(gamma_corrected, h=8)

    return result


def massive_lighting_experiments(image, output_dir="Output_massive/test"):
    """
    Rengeteg fény- és kontrasztkiegyenlítési módszer kipróbálása.
    Cél: látni, melyik kombináció segíti a pontdetekciót.
    """

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    os.makedirs(output_dir, exist_ok=True)
    step = 1

    # ────────────────────────────────────────────────
    # 1️⃣ Paramétertartományok
    blur_sizes = [21, 51, 81, 101]
    clip_limits = [1.0, 2.0, 3.0, 4.0]
    gammas = [0.7, 0.9, 1.0, 1.2, 1.5]
    morph_sizes = [15, 31, 51, 71]
    methods = ["blackhat", "retinex", "clahe", "illum"]

    logging.info("Starting massive preprocessing experiments...")

    # ────────────────────────────────────────────────
    # 2️⃣ Iterálunk minden kombináción
    for method in methods:
        for (blur, clip, gamma, morph) in product(blur_sizes, clip_limits, gammas, morph_sizes):

            try:
                result = gray.copy()

                # ───── Method 1: Black-hat enhancement ─────
                if method == "blackhat":
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph, morph))
                    blackhat = cv2.morphologyEx(result, cv2.MORPH_BLACKHAT, kernel)
                    result = cv2.add(result, blackhat)

                # ───── Method 2: Retinex-like log normalization ─────
                elif method == "retinex":
                    blur_img = cv2.GaussianBlur(result, (blur, blur), 0)
                    retinex = cv2.log(np.float32(result) + 1) - cv2.log(np.float32(blur_img) + 1)
                    result = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                # ───── Method 3: CLAHE + Laplacian ─────
                elif method == "clahe":
                    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8))
                    clahe_img = clahe.apply(result)
                    lap = cv2.Laplacian(clahe_img, cv2.CV_16S, ksize=3)
                    result = cv2.convertScaleAbs(clahe_img - 0.3 * lap)

                # ───── Method 4: Illumination flattening ─────
                elif method == "illum":
                    background = cv2.GaussianBlur(result, (blur, blur), 0)
                    result = cv2.divide(result, background, scale=255)

                # ───── Gamma correction ─────
                gamma_corr = np.power(result / 255.0, gamma)
                result = np.clip(gamma_corr * 255, 0, 255).astype(np.uint8)

                # ───── Optional noise reduction ─────
                result = cv2.bilateralFilter(result, 5, 50, 50)

                # ───── Save result ─────
                filename = f"step_{step:03d}_{method}_b{blur}_c{clip}_g{gamma}_m{morph}.jpg"
                path = os.path.join(output_dir, filename)
                cv2.imwrite(path, result)
                logging.info(f"Saved {filename}")
                step += 1

            except Exception as e:
                logging.warning(f"Failed for combo {method}, b={blur}, c={clip}, g={gamma}, m={morph}: {e}")

    logging.info(f"All {step-1} variations saved in '{output_dir}' directory.")



if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    config = load_config()
    GRADIENT_BLUR_SIZE = config["preprocess"]["GRADIENT_BLUR_SIZE"]
    CONTRAST_STRENGTH = config["preprocess"]["CONTRAST_STRENGTH"]
    NOISE_REDUCTION = config["preprocess"]["NOISE_REDUCTION"]
    PAPER_THRESHOLD = config["preprocess"]["PAPER_THRESHOLD"]
    CORNER_EPSILON = config["preprocess"]["CORNER_EPSILON"]
    logging.info("Current settings:")
    logging.info(f"Gradient Blur Size: {GRADIENT_BLUR_SIZE}")
    logging.info(f"Contrast Strength: {CONTRAST_STRENGTH}")
    logging.info(f"Noise Reduction: {NOISE_REDUCTION}")
    logging.info(f"Paper Threshold: {PAPER_THRESHOLD}")
    logging.info(f"Corner Epsilon: {CORNER_EPSILON}")
    logging.info("Please select an image to use:")

    base_path = os.path.dirname(os.path.abspath(__file__))
    picture_folder = os.path.join(base_path, "../../Pictures")
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

    threshold = 100
    test_image = apply_adaptive_threshold(grayscale_image, threshold=PAPER_THRESHOLD)
    save_step(test_image, "Test_threshold", 10000)

    gpt_test = fix_lighting_inconsistency(grayscale_image)
    save_step(gpt_test, "gpt_threshold", 794675)
    massive_lighting_experiments(grayscale_image)

    logging.info("\nUse heavy preprocessing? (y/n, default=y): ")
    use_preprocessing = input().strip().lower()
    use_heavy = use_preprocessing != 'n'
    
    processed_image = apply_preprocessing(grayscale_image, use_heavy=use_heavy)
    save_step(processed_image, "preprocessed" if use_heavy else "no_preprocessing", 3)

    blurred_image = apply_blur(processed_image)
    save_step(blurred_image, "blurred", 4)

    threshold_image = apply_adaptive_threshold(blurred_image, PAPER_THRESHOLD)
    save_step(threshold_image, "threshold", 5)

    a4_contour = detect_a4_contour(threshold_image)
    save_step(threshold_image, "contour", 6)
    
    if a4_contour is not None:
        logging.info("A4 paper detected successfully!")
        
        paper_mask = create_mask_from_contour(a4_contour, threshold_image.shape)
        save_step(paper_mask, "paper_mask", 7)
        
        masked_original = apply_mask_to_original(base_image, paper_mask)
        save_step(masked_original, "masked_original", 8)
        
        contour_visualization = visualize_contour(base_image, a4_contour)
        save_step(contour_visualization, "contour_detected", 9)

        output_picture_folder = os.path.join(base_path, "Output_pictures/step_08_masked_original.jpg")
        image_corner, corner_coordinates = locate_corners_white_paper(output_picture_folder)
        save_step(image_corner, "corners", 10)
        
        if len(corner_coordinates) == 4:
            logging.info(f"Top-Left corner: {corner_coordinates[0]}")
            logging.info(f"Top-Right corner: {corner_coordinates[1]}")
            logging.info(f"Bottom-Right corner: {corner_coordinates[2]}")
            logging.info(f"Bottom-Left corner: {corner_coordinates[3]}")

        logging.info("Applying standard A4 perspective transform...")
        corrected_standard = apply_perspective_transform(masked_original, corner_coordinates)
        save_step(corrected_standard, "perspective_A4", 11)
        
        grayscale_cropped = convert_to_grayscale(masked_original)
        
        if use_heavy:
            grayscale_cropped = apply_preprocessing(grayscale_cropped, use_heavy=True)
            save_step(grayscale_cropped, "before_transform_preprocessed", 12)
        else:
            save_step(grayscale_cropped, "before_transform_gray", 12)
        
        corrected_gray_standard = apply_perspective_transform(grayscale_cropped, corner_coordinates)
        save_step(corrected_gray_standard, "perspective_A4_gray", 13)
        
        corrected_image = apply_perspective_transform(masked_original, corner_coordinates, output_width=1240, output_height=1754)
        save_step(corrected_image, "smaller_resolution", 14)

        corrected_image = apply_perspective_transform(masked_original, corner_coordinates, output_width=1654, output_height=2339)
        save_step(corrected_image, "bigger_resolution", 15)
        
    else:
        logging.error("FAILED: Could not detect A4 paper")
        logging.error(f"Try adjusting PAPER_THRESHOLD (current: {PAPER_THRESHOLD})")
        sys.exit()