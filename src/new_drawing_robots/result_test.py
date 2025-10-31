import json
import matplotlib.pyplot as plt
import os
from collections import Counter
import logging
import sys
from pathlib import Path

# Naplózás beállítása
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# -- 1. Adatok betöltése --
try:
    base_path = Path(__file__).parent
    json_path = base_path / "Output_pictures" / "config" / "pairing.json"

    with open(json_path, 'r') as f:
        data = json.load(f)
        pairs = data.get('pairings') # Helyes beolvasás

except FileNotFoundError:
    logging.error(f"HIBA: A fájl nem található: {json_path}")
    logging.error("Ellenőrizd, hogy a szkript a megfelelő mappából fut-e,")
    logging.error("és hogy a pairing.json létezik-e az Output_pictures/config mappában.")
    sys.exit(1)
except json.JSONDecodeError:
    logging.error(f"HIBA: A JSON fájl hibás formátumú: {json_path}")
    sys.exit(1)
except AttributeError:
     # Ez a hiba akkor jön, ha 'data' = None
     logging.error(f"HIBA: A JSON fájl ({json_path}) betöltése nem sikerült, vagy nem tartalmaz 'pairings' kulcsot.")
     sys.exit(1)
except Exception as e:
     logging.error(f"Váratlan hiba a JSON betöltésekor: {e}")
     sys.exit(1)


if not pairs:
    logging.error("HIBA: A 'pairings' lista üres vagy hiányzik a pairing.json fájlból.")
    sys.exit(1)

# -- 2. Koordináták kinyerése és rendezés szám szerint --
try:
    # Helyes kulcs: number_info['number']
    sorted_pairs = sorted(pairs, key=lambda p: p['number_info']['number'])
except KeyError as e:
    logging.error(f"HIBA: A pairing.json formátuma nem megfelelő. Hiányzó kulcs: {e}")
    sys.exit(1)
except TypeError as e:
     logging.error(f"HIBA: Típushiba a rendezés során: {e}")
     sys.exit(1)

x_coords = []
y_coords = []
numbers = []
match_types = []

for item in sorted_pairs:
    try:
        # Helyes kulcs: dot_coordinates
        dot_coord = item['dot_coordinates']
        x_coords.append(dot_coord['x'])
        y_coords.append(dot_coord['y'])
        numbers.append(item['number_info']['number'])
        match_types.append('matched')
    except KeyError as e:
        logging.warning(f"FIGYELEM: Hiányzó kulcs egy párban: {e}. A hibás elem kihagyva: {item}")
        continue

if not x_coords:
    logging.error("HIBA: Nem sikerült érvényes koordinátákat kinyerni.")
    sys.exit(1)

# --- MÓDOSÍTÁS: A hurok bezárása KIVÉVE ---
# x_coords.append(x_coords[0])
# y_coords.append(y_coords[0])
logging.info("A rajzot nem zárjuk be (első-utolsó pont nincs összekötve).")
# --- MÓDOSÍTÁS VÉGE ---

# -- 3. Színkódolás beállítása --
color_map = { 'matched': 'green', 'unknown': 'gray' }
point_colors = [color_map.get(mt, 'gray') for mt in match_types]

# -- 4. Ábra létrehozása --
fig, ax = plt.subplots(figsize=(14, 12))
ax.plot(x_coords, y_coords, 'b-', linewidth=1.5, alpha=0.6, label='Összekötő vonalak')

for i, (x, y, num, color) in enumerate(zip(x_coords, y_coords, numbers, point_colors)):
    ax.plot(x, y, 'o', color=color, markersize=6, markeredgecolor='black', markeredgewidth=0.5)
    ax.annotate(str(num), (x, y), xytext=(5, 5), textcoords='offset points',
                fontsize=7, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))

ax.invert_yaxis()
ax.set_xlabel('X koordináta', fontsize=12)
ax.set_ylabel('Y koordináta', fontsize=12)
ax.set_title('Connect the Dots - Párosítás vizualizáció', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.axis('equal')

# -- 5. Legenda és Statisztika --
from matplotlib.patches import Patch
legend_elements = [ Patch(facecolor='green', edgecolor='black', label='Matched (Párosítva)') ]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
total_pairs_from_json = data.get('total_pairs', len(pairs))
stats_text = f"Összesen: {total_pairs_from_json} pár"
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.show()

# -- 6. Konzol kimenet --
print(f"\n✓ Összesen {len(sorted_pairs)} pont lett összekötve a grafikonon.")
if numbers:
    print(f"✓ Számok tartománya: {min(numbers)} - {max(numbers)}")
else:
    print("✓ Nem találhatóak számok a megjelenítéshez.")
type_counts = Counter(match_types)
print(f"\n✓ Match type bontás:")
for match_type, count in type_counts.items():
    print(f"  - {match_type}: {count}")