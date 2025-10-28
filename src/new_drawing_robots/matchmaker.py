import json
import math
import pandas as pd
import os  # Szükséges a mappák és fájlnevek kezeléséhez


def load_config(config_file):
    """Betölti a megadott JSON konfigurációs fájlt."""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"Konfigurációs fájl ('{config_file}') sikeresen betöltve.")
        return config
    except FileNotFoundError:
        print(f"HIBA: A konfigurációs fájl nem található: {config_file}")
        return None
    except json.JSONDecodeError:
        print(f"HIBA: A konfigurációs fájl ('{config_file}') hibás formátumú.")
        return None


def calculate_distance(p1, p2):
    """Két pont (mindkettő {'x': int, 'y': int} formátumú) között euklideszi távolságot számol."""
    dx = p1['x'] - p2['x']
    dy = p1['y'] - p2['y']
    return math.sqrt(dx * dx + dy * dy)


def pair_numbers_to_dots(config, expected_range):
    """
    Összepárosítja a SZÁMOKAT a legközelebbi PONTOKKAL a config fájl alapján.
    Minden fájlt a megadott 'config_dir' mappából olvas és oda ment.
    """

    # 1. Fájlnevek kiolvasása a configból
    try:
        filenames = config['filenames']
        config_dir = config["paths"]["config_dir"]

        # Bemeneti fájlok teljes elérési útjának összeállítása
        dots_file_path = os.path.join(config_dir, filenames['global_dots'])
        numbers_file_path = os.path.join(config_dir, filenames['global_numbers'])

        # Kimeneti fájlok teljes elérési útjának összeállítása
        output_json_name = filenames['pairing']
        base_output_name = os.path.splitext(output_json_name)[0]

        output_json_path = os.path.join(config_dir, output_json_name)
        output_csv_path = os.path.join(config_dir, f"{base_output_name}.csv")
        collision_json_path = os.path.join(config_dir, f"{base_output_name}_collisions.json")

    except KeyError as e:
        print(f"HIBA: Hiányzó kulcs a config['filenames'] részben: {e}")
        return
    except TypeError:
        print("HIBA: A 'config' paraméter érvénytelennek tűnik (talán None?).")
        return

    # 2. Adatok betöltése a megadott elérési utakról
    try:
        with open(dots_file_path, 'r', encoding='utf-8') as f:
            dots_data = json.load(f)
        with open(numbers_file_path, 'r', encoding='utf-8') as f:
            numbers_data = json.load(f)
    except FileNotFoundError as e:
        print(f"HIBA: Az adatfájl nem található: {e.filename}")
        return
    except json.JSONDecodeError as e:
        print(f"HIBA: JSON formátum hiba a(z) '{e.doc}' fájlban.")
        return

    dots_list = dots_data.get('circles', [])
    all_numbers_list = numbers_data.get('numbers', [])

    if not dots_list:
        print(f"Hiba: Nem találhatók 'circles' (pontok) adatok a(z) '{dots_file_path}' fájlban.")
        return
    if not all_numbers_list:
        print(f"Hiba: Nem találhatók 'numbers' (számok) adatok a(z) '{numbers_file_path}' fájlban.")
        return

    # 3. Számok szűrése az expected_range alapján
    valid_numbers = [
        num for num in all_numbers_list
        if 1 <= num.get('number', -1) <= expected_range
    ]

    print(f"Összesen betöltött pont (dot): {len(dots_list)}")
    print(f"Összesen betöltött szám-detekció: {len(all_numbers_list)}")
    print(f"Ebből érvényes (1-{expected_range} tartományban): {len(valid_numbers)}")

    if not valid_numbers:
        print("Hiba: Nincsenek érvényes számok a megadott tartományban. A párosítás leáll.")
        return

    # 4. Párosítás: Minden érvényes SZÁMHOZ a legközelebbi PONTOT
    final_pairings = {}

    for number_info in valid_numbers:
        number_detection_id = number_info.get('dot_id')
        number_coord = number_info.get('global_coordinates')

        if number_detection_id is None or number_coord is None:
            print(f"Figyelmeztetés: Hiányos adatú szám-detekciót találtam, kihagyom: {number_info}")
            continue

        best_match_dot = None
        min_distance = float('inf')

        for dot in dots_list:
            dot_coord = dot.get('global_coordinates')
            if dot_coord is None:
                continue

            distance = calculate_distance(number_coord, dot_coord)

            if distance < min_distance:
                min_distance = distance
                best_match_dot = dot

        if best_match_dot:
            final_pairings[number_detection_id] = {
                'number_detection_id': number_detection_id,
                'number_value': number_info.get('number'),
                'number_coords': number_coord,
                'number_confidence': number_info.get('confidence'),
                'matched_dot_id': best_match_dot.get('id'),
                'matched_dot_coords': best_match_dot.get('global_coordinates'),
                'distance': min_distance
            }
        else:
            print(f"Figyelmeztetés: A {number_detection_id} azonosítójú számhoz nem található párosítható pont.")

    # 5. Duplikátumok/Ütközések elemzése
    dot_id_to_number_map = {}
    for pairing in final_pairings.values():
        matched_dot_id = pairing['matched_dot_id']
        if matched_dot_id not in dot_id_to_number_map:
            dot_id_to_number_map[matched_dot_id] = []
        dot_id_to_number_map[matched_dot_id].append(pairing)

    dot_collisions = {
        str(dot_id): pairings_list
        for dot_id, pairings_list in dot_id_to_number_map.items()
        if len(pairings_list) > 1
    }

    # 6. Eredmények mentése a 'config_dir' mappába
    final_pairings_list = list(final_pairings.values())

    try:
        # Fő párosítási fájl (JSON)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_pairings_list, f, indent=2, ensure_ascii=False)

        # Ütközés riport (JSON)
        with open(collision_json_path, 'w', encoding='utf-8') as f:
            json.dump(dot_collisions, f, indent=2, ensure_ascii=False)

        # CSV kimenet
        df_pairings = pd.DataFrame(final_pairings_list)
        df_pairings = df_pairings.rename(columns={
            'number_detection_id': 'Szam_Detekcio_ID',
            'number_value': 'Szam_Ertek',
            'matched_dot_id': 'Hozza_Rendelt_Pont_ID',
            'distance': 'Tavolsag',
            'number_confidence': 'Szam_Konfidencia',
            'number_coords': 'Szam_Koordinata',
            'matched_dot_coords': 'Pont_Koordinata'
        })
        csv_columns = [
            'Szam_Detekcio_ID', 'Szam_Ertek', 'Hozza_Rendelt_Pont_ID',
            'Tavolsag', 'Szam_Konfidencia', 'Szam_Koordinata', 'Pont_Koordinata'
        ]
        csv_columns_exist = [col for col in csv_columns if col in df_pairings.columns]
        df_pairings[csv_columns_exist].to_csv(output_csv_path, index=False, encoding='utf-8-sig')

    except IOError as e:
        print(f"HIBA: Írási hiba történt a(z) '{e.filename}' fájl mentésekor: {e}")
    except Exception as e:
        print(f"HIBA: Váratlan hiba a fájlmentés során: {e}")

    # --- Összegzés ---
    print("\n--- ÖSSZEGZÉS (Szám -> Pont logika alapján) ---")
    print(f"Sikeresen párosítva: {len(final_pairings_list)} érvényes szám.")
    print(f"Azonosított pont-ütközés: {len(dot_collisions)} db")
    print(f"\nA teljes párosítási lista elmentve: '{output_json_path}'")
    print(f"A CSV változat elmentve: '{output_csv_path}'")
    print(f"Az ütközések riportja elmentve: '{collision_json_path}'")