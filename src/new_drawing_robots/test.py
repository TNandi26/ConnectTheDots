import cv2
import os

image = cv2.imread(r'C:\ConnectTheDots\Pictures\lo_Color_2.png')
if image is None:
    print("Hiba: A kép nem található!")
else:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(gray, (51, 51), 0)
    normalized_image = cv2.divide(gray, background, scale=255.0)

    output_dir = "thresh_results"
    os.makedirs(output_dir, exist_ok=True)

    # Csak érvényes blockSize értékek: páratlan és > 1
    for blockSize in range(3, 16, 2):  # 3, 5, 7, 9, 11, 13, 15
        for C in range(1, 16):
            final_binary_image = cv2.adaptiveThreshold(
                normalized_image, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize, C
            )

            filename = f"adaptive_b{blockSize}_C{C}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), final_binary_image)
            print(f"Mentve: {filename}")
