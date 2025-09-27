import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from collections import defaultdict
import os
import pytesseract

class NumberDetectorContour:
    def __init__(self, image_path, circles_json_path=None):
        """
        Kontúr-alapú számdetektort inicializálja
        
        Args:
            image_path (str): A kép elérési útja
            circles_json_path (str): A körök JSON fájljának elérési útja
        """
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Nem található a kép: {image_path}")
        
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Körök betöltése
        if circles_json_path:
            self.circles_data = self.load_circles_from_json(circles_json_path)
        else:
            self.circles_data = []
        
        self.detected_numbers = []
        self.debug_image = None
        
    def load_circles_from_json(self, json_path):
        """Körök betöltése JSON fájlból"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            circles = []
            for circle in data['circles']:
                converted_circle = {
                    'id': circle['id'],
                    'x': circle['center_x'],
                    'y': circle['center_y'],
                    'radius': circle['radius']
                }
                circles.append(converted_circle)
            
            print(f"Betöltve {len(circles)} kör a JSON fájlból")
            return circles
            
        except Exception as e:
            print(f"Hiba a JSON fájl betöltésekor: {e}")
            return []
    
    def preprocess_image(self, use_blur=True, use_morphology=True):
        """
        1️⃣ Kép előkészítése kontúr detektáláshoz
        """
        # Zajcsökkentés
        if use_blur:
            processed = cv2.GaussianBlur(self.gray, (3, 3), 0)
        else:
            processed = self.gray.copy()
        
        # Threshold / binarizálás - fekete számok fehér háttéren
        _, binary = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morfológiai műveletek
        if use_morphology:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        self.binary = binary
        return binary
    
    def detect_digit_contours(self, min_area=20, max_area=500, min_aspect_ratio=0.2, max_aspect_ratio=3.0):
        """
        2️⃣ Számjegy kontúrok detektálása és szűrése
        """
        if not hasattr(self, 'binary'):
            self.preprocess_image()
        
        # Kontúrok keresése
        contours, _ = cv2.findContours(self.binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        digit_regions = []
        
        for contour in contours:
            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Szűrés méret és arány alapján
            if (min_area <= area <= max_area and 
                min_aspect_ratio <= aspect_ratio <= max_aspect_ratio and
                w > 5 and h > 8):  # Minimum méretek
                
                digit_regions.append({
                    'contour': contour,
                    'bbox': (x, y, w, h),
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'center': (x + w//2, y + h//2)
                })
        
        print(f"Talált {len(digit_regions)} potenciális számjegy régiót")
        return digit_regions
    
    def extract_and_recognize_digits(self, digit_regions, scale_factor=3):
        """
        3️⃣ ROI kivágás és OCR minden számjegyre
        """
        try:
            import pytesseract
        except ImportError:
            print("FIGYELEM: pytesseract nincs telepítve. Dummy eredményekkel folytatom...")
            return self._create_dummy_digits(digit_regions)
        
        recognized_digits = []
        
        for i, region in enumerate(digit_regions):
            x, y, w, h = region['bbox']
            
            # ROI kivágása
            roi = self.binary[y:y+h, x:x+w]
            
            # ROI nagyítása OCR-hez
            if scale_factor > 1:
                new_w, new_h = w * scale_factor, h * scale_factor
                roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            # Padding hozzáadása (OCR szereti)
            padded_roi = cv2.copyMakeBorder(roi, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=0)
            
            try:
                # OCR egyetlen karakterre optimalizálva
                config = '--psm 10 -c tessedit_char_whitelist=0123456789'
                text = pytesseract.image_to_string(padded_roi, config=config).strip()
                
                if text.isdigit():
                    recognized_digits.append({
                        'digit': int(text),
                        'bbox': region['bbox'],
                        'center': region['center'],
                        'confidence': 100,  # Pytesseract confidence nem elérhető psm 10-nél
                        'roi_id': i
                    })
                
            except Exception as e:
                print(f"OCR hiba a {i}. régióban: {e}")
        
        print(f"Felismert {len(recognized_digits)} számjegy")
        return recognized_digits
    
    def _create_dummy_digits(self, digit_regions):
        """Dummy számjegyek tesseract nélküli teszteléshez"""
        dummy_digits = []
        for i, region in enumerate(digit_regions):
            dummy_digits.append({
                'digit': (i % 10),  # 0-9 ciklikus
                'bbox': region['bbox'],
                'center': region['center'],
                'confidence': 95,
                'roi_id': i
            })
        return dummy_digits
    
    def group_digits_into_numbers(self, recognized_digits, row_tolerance=15, number_gap_threshold=25):
        """
        4️⃣ Számjegyek csoportosítása többjegyű számokká
        
        Args:
            row_tolerance: Maximális y-koordináta különbség egy sorban
            number_gap_threshold: Minimális távolság különböző számok között
        """
        if not recognized_digits:
            return []
        
        # Sorok azonosítása y-koordináta alapján
        rows = defaultdict(list)
        for digit in recognized_digits:
            y_center = digit['center'][1]
            # Keressük meg a legközelebbi sort
            assigned = False
            for row_y in rows.keys():
                if abs(y_center - row_y) <= row_tolerance:
                    rows[row_y].append(digit)
                    assigned = True
                    break
            
            if not assigned:
                rows[y_center] = [digit]
        
        # Minden sorban számok összeállítása
        all_numbers = []
        number_id = 1
        
        for row_y, digits_in_row in rows.items():
            # Rendezés x-koordináta szerint
            digits_in_row.sort(key=lambda d: d['center'][0])
            
            # Számok szeparálása távolság alapján
            current_number_digits = [digits_in_row[0]]
            
            for i in range(1, len(digits_in_row)):
                current_digit = digits_in_row[i]
                prev_digit = digits_in_row[i-1]
                
                # Távolság az előző számjegytől
                distance = current_digit['center'][0] - prev_digit['center'][0]
                
                if distance > number_gap_threshold:
                    # Új szám kezdődik
                    if current_number_digits:
                        number = self._create_number_from_digits(current_number_digits, number_id)
                        all_numbers.append(number)
                        number_id += 1
                    current_number_digits = [current_digit]
                else:
                    # Ugyanahhoz a számhoz tartozik
                    current_number_digits.append(current_digit)
            
            # Az utolsó szám hozzáadása
            if current_number_digits:
                number = self._create_number_from_digits(current_number_digits, number_id)
                all_numbers.append(number)
                number_id += 1
        
        print(f"Összeállított {len(all_numbers)} szám")
        return all_numbers
    
    def _create_number_from_digits(self, digits, number_id):
        """Számjegyek listájából szám objektum létrehozása"""
        # Számjegyek összeillesztése
        number_str = "".join([str(d['digit']) for d in digits])
        number_value = int(number_str)
        
        # Bounding box kiszámítása
        min_x = min(d['bbox'][0] for d in digits)
        min_y = min(d['bbox'][1] for d in digits)
        max_x = max(d['bbox'][0] + d['bbox'][2] for d in digits)
        max_y = max(d['bbox'][1] + d['bbox'][3] for d in digits)
        
        width = max_x - min_x
        height = max_y - min_y
        center_x = min_x + width // 2
        center_y = min_y + height // 2
        
        return {
            'id': number_id,
            'number': number_value,
            'digits': digits,
            'bounding_box': {
                'x': min_x, 'y': min_y, 
                'width': width, 'height': height
            },
            'center': {'x': center_x, 'y': center_y},
            'confidence': sum(d['confidence'] for d in digits) / len(digits)
        }
    
    def pair_circles_with_numbers(self, numbers, max_distance=50):
        """Körök párosítása a számokkal"""
        if not self.circles_data or not numbers:
            return []
        
        circle_coords = np.array([[c['x'], c['y']] for c in self.circles_data])
        number_coords = np.array([[n['center']['x'], n['center']['y']] for n in numbers])
        
        distances = cdist(circle_coords, number_coords)
        
        paired_data = []
        used_numbers = set()
        
        for i, circle in enumerate(self.circles_data):
            closest_distances = distances[i]
            sorted_indices = np.argsort(closest_distances)
            
            for j in sorted_indices:
                if j not in used_numbers and closest_distances[j] <= max_distance:
                    number_data = numbers[j]
                    
                    paired_item = {
                        'circle': {
                            'id': circle['id'],
                            'center': {'x': circle['x'], 'y': circle['y']},
                            'radius': circle['radius']
                        },
                        'number': {
                            'id': number_data['id'],
                            'value': number_data['number'],
                            'center': number_data['center'],
                            'bounding_box': number_data['bounding_box'],
                            'confidence': number_data['confidence'],
                            'digit_count': len(number_data['digits'])
                        },
                        'distance': float(closest_distances[j])
                    }
                    
                    paired_data.append(paired_item)
                    used_numbers.add(j)
                    break
        
        paired_data.sort(key=lambda x: x['number']['value'])
        print(f"Sikeres párosítások: {len(paired_data)}")
        return paired_data
    
    def create_debug_visualization(self, digit_regions=None, numbers=None):
        """5️⃣ Debug vizualizáció"""
        debug_img = self.image.copy()
        
        if digit_regions:
            # Számjegy régiók
            for region in digit_regions:
                x, y, w, h = region['bbox']
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 1)
        
        if numbers:
            # Felismert számok
            for number in numbers:
                bbox = number['bounding_box']
                x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
                
                # Szám bounding box
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Szám szöveg
                cv2.putText(debug_img, str(number['number']), 
                           (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Középpont
                center = (number['center']['x'], number['center']['y'])
                cv2.circle(debug_img, center, 3, (255, 0, 255), -1)
        
        self.debug_image = debug_img
        return debug_img
    
    def save_results(self, output_path='number_detection_results.json', paired_data=None):
        """Eredmények mentése"""
        output_data = {
            'image_path': self.image_path,
            'detection_method': 'Contour-based segmentation + OCR',
            'total_pairs': len(paired_data) if paired_data else 0,
            'detection_summary': {
                'circles_detected': len(self.circles_data),
                'numbers_detected': len(self.detected_numbers),
                'successful_pairs': len(paired_data) if paired_data else 0
            },
            'paired_data': paired_data or []
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Eredmények mentve: {output_path}")
        return output_data
    
    def visualize_results(self, paired_data=None, save_path=None):
        """Eredmények vizualizálása"""
        if paired_data is None:
            return
        
        plt.figure(figsize=(15, 10))
        plt.imshow(self.image_rgb)
        
        for item in paired_data:
            circle = item['circle']
            number = item['number']
            
            # Kör
            circle_plot = plt.Circle(
                (circle['center']['x'], circle['center']['y']), 
                circle['radius'], 
                fill=False, color='red', linewidth=2
            )
            plt.gca().add_patch(circle_plot)
            
            # Szám bounding box
            bbox = number['bounding_box']
            rect = plt.Rectangle(
                (bbox['x'], bbox['y']), 
                bbox['width'], bbox['height'], 
                fill=False, color='blue', linewidth=2
            )
            plt.gca().add_patch(rect)
            
            # Összekötő vonal
            plt.plot(
                [circle['center']['x'], number['center']['x']], 
                [circle['center']['y'], number['center']['y']], 
                'g--', linewidth=1, alpha=0.7
            )
            
            # Szám címke
            plt.text(
                number['center']['x'], number['center']['y'] - 10, 
                str(number['value']), 
                ha='center', va='bottom', 
                fontsize=10, fontweight='bold', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8)
            )
        
        plt.title('Körök és számok párosítása (Kontúr-alapú detektálás)')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

# ================== FŐPROGRAM ==================

def main():
    """
    Kontúr-alapú számdetektálás főprogramja
    """
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    base_path = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_path, "Output_pictures")

    # ================== ELÉRÉSI UTAK ==================
    image_path = os.path.join(output_path, "step_01_original.jpg")
    circles_json_path = os.path.join(output_path, "black_filtered_coordinates.json")
    output_json = "contour_detection_results.json"
    visualization_output = "contour_visualization.png"
    debug_output = "debug_visualization.png"
    # ==================================================
    
    try:
        # Detektor inicializálása
        detector = NumberDetectorContour(image_path, circles_json_path)
        print(f"✓ Kép betöltve: {image_path}")
        print(f"✓ Körök betöltve: {len(detector.circles_data)} darab")
        
        # 1️⃣ Kép előfeldolgozása
        binary = detector.preprocess_image(use_blur=True, use_morphology=True)
        print("✓ Kép előfeldolgozása kész")
        
        # 2️⃣ Számjegy kontúrok detektálása
        digit_regions = detector.detect_digit_contours(
            min_area=20, max_area=500, 
            min_aspect_ratio=0.2, max_aspect_ratio=3.0
        )
        
        if not digit_regions:
            print("❌ Nem találtunk számjegy régiókat!")
            return
        
        # 3️⃣ OCR számjegyekre
        recognized_digits = detector.extract_and_recognize_digits(digit_regions, scale_factor=3)
        
        if not recognized_digits:
            print("❌ Nem sikerült számjegyeket felismerni!")
            return
        
        # 4️⃣ Számjegyek csoportosítása számokká
        numbers = detector.group_digits_into_numbers(
            recognized_digits, 
            row_tolerance=15, 
            number_gap_threshold=25
        )
        detector.detected_numbers = numbers
        
        if numbers:
            print("✓ Felismert számok:")
            for num in numbers[:10]:  # Első 10 szám
                print(f"  Szám: {num['number']} @ ({num['center']['x']}, {num['center']['y']})")
        
        # Párosítás
        paired_data = detector.pair_circles_with_numbers(numbers, max_distance=50)
        
        # 5️⃣ Debug vizualizáció
        debug_img = detector.create_debug_visualization(digit_regions, numbers)
        cv2.imwrite(debug_output, debug_img)
        print(f"✓ Debug kép mentve: {debug_output}")
        
        # Eredmények mentése
        detector.save_results(output_json, paired_data)
        detector.visualize_results(paired_data, visualization_output)
        
        # Összefoglaló
        print("\n" + "="*50)
        print("ÖSSZEFOGLALÓ:")
        print(f"Körök száma: {len(detector.circles_data)}")
        print(f"Számjegy régiók: {len(digit_regions)}")
        print(f"Felismert számjegyek: {len(recognized_digits)}")
        print(f"Összeállított számok: {len(numbers)}")
        print(f"Sikeres párosítások: {len(paired_data)}")
        if detector.circles_data:
            print(f"Párosítási arány: {len(paired_data)/len(detector.circles_data)*100:.1f}%")
        
    except Exception as e:
        print(f"❌ Hiba történt: {e}")
        import traceback
        traceback.print_exc()

# ================== TESZT FÜGGVÉNYEK ==================

def test_without_tesseract():
    """Tesztelés Tesseract nélkül (dummy adatokkal)"""
    print("=== TESSERACT NÉLKÜLI TESZT ===")
    main()

def show_preprocessing_steps(image_path):
    """Előfeldolgozási lépések megjelenítése"""
    detector = NumberDetectorContour(image_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Eredeti
    axes[0,0].imshow(detector.image_rgb)
    axes[0,0].set_title('Eredeti kép')
    axes[0,0].axis('off')
    
    # Szürkeárnyalatos
    axes[0,1].imshow(detector.gray, cmap='gray')
    axes[0,1].set_title('Szürkeárnyalatos')
    axes[0,1].axis('off')
    
    # Blur nélkül
    _, binary_no_blur = cv2.threshold(detector.gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    axes[1,0].imshow(binary_no_blur, cmap='gray')
    axes[1,0].set_title('Threshold (blur nélkül)')
    axes[1,0].axis('off')
    
    # Teljes előfeldolgozás
    binary_full = detector.preprocess_image()
    axes[1,1].imshow(binary_full, cmap='gray')
    axes[1,1].set_title('Teljes előfeldolgozás')
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.savefig('preprocessing_steps.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Alap futtatás
    main()
    
    # Vagy tesztelés Tesseract nélkül:
    # test_without_tesseract()
    
    # Vagy előfeldolgozási lépések megjelenítése:
    # show_preprocessing_steps("your_image.jpg")