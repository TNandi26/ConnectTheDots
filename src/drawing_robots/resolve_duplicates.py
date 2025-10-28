import json
import math
import pandas as pd
import os
from collections import defaultdict

# --- Konfiguráció ---
# A 3 legközelebbi szomszéd vizsgálata a tipphez
NUM_NEIGHBORS_FOR_GUESS = 3
EXPECTED_RANGE = 100
CONFIG_DIR = "Output_pictures/config"
CONFIG_FILE = "config.json"


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


def resolve_duplicate_pairings(config, config_dir, expected_range, num_neighbors):
    """
    Beolvassa a 'pairing.json'-t, és a szomszédok alapján
    megpróbálja feloldani a duplikált számértékeket.
    """

    # 1. Fájlnevek és elérési utak
    try:
        filenames = config['filenames']
        # Bemenet: az előző szkript kimenete
        input_json_name = filenames['pairing']
        input_json_path = os.path.join(config_dir, input_json_name)

        # Kimenet: új, javított fájlok
        base_output_name = os.path.splitext(input_json_name)[0] + "_corrected"
        output_json_path = os.path.join(config_dir, f"{base_output_name}.json")
        output_csv_path = os.path.join(config_dir, f"{base_output_name}.csv")
        report_json_path = os.path.join(config_dir, f"{base_output_name}_report.json")

    except KeyError as e:
        print(f"HIBA: Hiányzó kulcs a config['filenames'] részben: {e}")
        return
    except TypeError:
        print("HIBA: A 'config' paraméter érvénytelennek tűnik.")
        return

    # 2. Bemeneti párosítások betöltése
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            all_pairs = json.load(f)
    except FileNotFoundError:
        print(f"HIBA: A bemeneti fájl nem található: {input_json_path}")
        return
    except json.JSONDecodeError:
        print(f"HIBA: A bemeneti JSON fájl hibás: {input_json_path}")
        return

    print(f"Összesen {len(all_pairs)} párosítás betöltve innen: {input_json_path}")

    # 3. Párok csoportosítása és hiányzó számok keresése
    number_groups = defaultdict(list)
    present_numbers = set()

    for pair in all_pairs:
        num_val = pair['number_value']
        number_groups[num_val].append(pair)
        present_numbers.add(num_val)

    # Hiányzó számok listája
    missing_numbers = sorted(list(set(range(1, expected_range + 1)) - present_numbers))
    print(f"Hiányzó számok (későbbi tipphez): {missing_numbers}")

    # Duplikátumok azonosítása
    duplicate_values = {val for val, group in number_groups.items() if len(group) > 1}
    print(f"Duplikált számértékek: {duplicate_values}")

    # 4. Feldolgozás: Egyedi és Duplikált párok szétválasztása

    corrected_pairs = []  # Ez lesz a végső, javított lista
    original_pairs = []  # Ebből számoljuk a szomszédokat
    reassignment_report = []  # Napló a változtatásokról

    # 4a. Egyedi párok (nem duplikáltak) azonnali hozzáadása
    for num_val, group in number_groups.items():
        if num_val not in duplicate_values:
            pair = group[0]
            pair['match_type'] = 'original'
            corrected_pairs.append(pair)
            original_pairs.append(pair)  # Hozzáadás a referencia listához

    print(f"Azonosítva {len(original_pairs)} egyedi (original) pár.")

    # 4b. Duplikált párok feldolgozása
    for num_val in sorted(list(duplicate_values)):  # Sorba rendezés a konzisztencia miatt
        conflicting_pairs = number_groups[num_val]
        print(f"\n--- Feldolgozás: '{num_val}' (találat: {len(conflicting_pairs)} db) ---")

        scores = []  # (költség, tipp_átlag, pár_objektum)

        # Szomszédok keresése az *eredeti* (egyedi) párok között
        if not original_pairs:
            print(
                "FIGYELMEZTETÉS: Nincsenek 'original' párok a szomszéd-elemzéshez. Duplikátumok feloldása sikertelen.")
            # Hozzáadjuk őket feloldatlanul
            for pair in conflicting_pairs:
                pair['match_type'] = 'duplicate_unresolved'
                corrected_pairs.append(pair)
            continue  # Ugrás a következő duplikált számra

        # Költség számítása minden duplikált párra
        for pair_to_score in conflicting_pairs:
            coord_to_score = pair_to_score['matched_dot_coords']
            distances = []  # (távolság, szomszéd_értéke)

            for other_pair in original_pairs:
                # Önmagát ne vegye figyelembe (bár itt nem is fogja, mert 'original_pairs'-ban nincs benne)
                if other_pair['matched_dot_id'] == pair_to_score['matched_dot_id']:
                    continue

                other_coord = other_pair['matched_dot_coords']
                dist = calculate_distance(coord_to_score, other_coord)
                distances.append((dist, other_pair['number_value']))

            # Legközelebbi N szomszéd kiválasztása
            sorted_neighbors = sorted(distances, key=lambda x: x[0])
            closest_neighbors = sorted_neighbors[:num_neighbors]
            neighbor_values = [val for dist, val in closest_neighbors]

            if not neighbor_values:
                print(f"FIGYELMEZTETÉS: A(z) {pair_to_score['matched_dot_id']} ponthoz nem található szomszéd.")
                scores.append((float('inf'), 0, pair_to_score))  # Magas költség, hogy a végére kerüljön
                continue

            # Tipp és Költség
            avg_guess = sum(neighbor_values) / len(neighbor_values)
            cost = abs(num_val - avg_guess)  # Mennyire tér el a tipp a jelenlegi értéktől
            scores.append((cost, avg_guess, pair_to_score))
            print(
                f"  > Pár (Detekció ID: {pair_to_score['number_detection_id']}): Szomszédok: {neighbor_values}, Tipp: {avg_guess:.2f}, Költség: {cost:.2f}")

        # 5. Döntés: Legjobb megtartása, többi átnevezése

        # Sorbarendezés költség szerint (legalacsonyabb elöl)
        sorted_scores = sorted(scores, key=lambda x: x[0])

        # 5a. A legjobb (legalacsonyabb költségű) pár megtartása
        best_pair_data = sorted_scores[0]
        best_pair = best_pair_data[2]
        best_pair['match_type'] = 'duplicate_kept'
        corrected_pairs.append(best_pair)
        print(
            f"  > DÖNTÉS: Megtartva (legalacsonyabb költség): Detekció ID {best_pair['number_detection_id']} (Költség: {best_pair_data[0]:.2f})")

        # 5b. A többi ("rossz") pár átnevezése
        for cost, avg_guess, bad_pair in sorted_scores[1:]:

            # Keressük a tipphez (avg_guess) legközelebbi hiányzó számot
            if not missing_numbers:
                print(f"  > HIBA: A(z) {bad_pair['number_detection_id']} átnevezéséhez nincs több hiányzó szám!")
                bad_pair['match_type'] = 'duplicate_unresolved'
                corrected_pairs.append(bad_pair)
                continue

            # Hiányzó szám keresése
            best_missing_num = -1
            min_diff = float('inf')
            for missing_num in missing_numbers:
                diff = abs(avg_guess - missing_num)
                if diff < min_diff:
                    min_diff = diff
                    best_missing_num = missing_num

            # Átnevezés
            original_val = bad_pair['number_value']
            bad_pair['number_value'] = best_missing_num  # Érték átírása!
            bad_pair['match_type'] = 'duplicate_reassigned'
            bad_pair['original_number_value'] = original_val  # Eltároljuk, mi volt az eredeti
            bad_pair['guess_based_on_neighbors'] = avg_guess
            corrected_pairs.append(bad_pair)

            # Eltávolítjuk a számot a listából, hogy ne használjuk fel újra
            missing_numbers.remove(best_missing_num)

            # Naplózás
            report_entry = {
                'reassigned_detection_id': bad_pair['number_detection_id'],
                'original_value': original_val,
                'reassigned_value': best_missing_num,
                'neighbor_guess': avg_guess,
                'cost': cost
            }
            reassignment_report.append(report_entry)
            print(
                f"  > DÖNTÉS: Átnevezve (Detekció ID: {bad_pair['number_detection_id']}) -> {best_missing_num} (Tipp: {avg_guess:.2f})")

    # 6. Mentés
    try:
        # Fő javított JSON
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(corrected_pairs, f, indent=2, ensure_ascii=False)

        # Riport JSON
        with open(report_json_path, 'w', encoding='utf-8') as f:
            json.dump(reassignment_report, f, indent=2, ensure_ascii=False)

        # CSV
        df_corrected = pd.DataFrame(corrected_pairs)
        df_corrected.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

        print("\n--- ÖSSZEGZÉS ---")
        print(f"Sikeresen feldolgozva {len(corrected_pairs)} pár.")
        print(f"Duplikátum feloldások naplózva: {len(reassignment_report)} db")
        print(f"Javított JSON elmentve: {output_json_path}")
        print(f"Javított CSV elmentve: {output_csv_path}")
        print(f"Feloldási riport elmentve: {report_json_path}")

    except IOError as e:
        print(f"HIBA: Írási hiba történt a(z) '{e.filename}' fájl mentésekor: {e}")
    except Exception as e:
        print(f"HIBA: Váratlan hiba a fájlmentés során: {e}")


# --- A SZKRIPT FUTTATÁSA ---

# 1. Konfiguráció betöltése (a szkript melletti config.json)
config_data = load_config(CONFIG_FILE)

# 2. Duplikátumkezelő futtatása
if config_data:
    resolve_duplicate_pairings(
        config=config_data,
        config_dir=CONFIG_DIR,
        expected_range=EXPECTED_RANGE,
        num_neighbors=NUM_NEIGHBORS_FOR_GUESS
    )
else:
    print("A program leáll, mert a konfigurációs fájl nem tölthető be.")