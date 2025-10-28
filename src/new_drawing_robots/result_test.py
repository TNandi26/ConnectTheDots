import json
import matplotlib.pyplot as plt
import os
from collections import Counter

# -- 1. Adatok betöltése --
# Feltételezzük, hogy a script a gyökérben van, és mellette van az Output_pictures mappa
try:
    base_path = os.path.dirname(os.path.realpath(__file__))
    json_path = os.path.join(base_path, "Output_pictures/config/pairing.json")

    with open(json_path, 'r') as f:
        # A mi pairing.json fájlunk egy lista
        pairs = json.load(f)

except FileNotFoundError:
    print(f"HIBA: A fájl nem található: {json_path}")
    print("Ellenőrizd, hogy a szkript a megfelelő mappából fut-e.")
    exit()
except json.JSONDecodeError:
    print(f"HIBA: A JSON fájl hibás formátumú: {json_path}")
    exit()

if not pairs:
    print("HIBA: A pairing.json fájl üres.")
    exit()

# -- 2. Koordináták kinyerése és rendezés szám szerint --
try:
    sorted_pairs = sorted(pairs, key=lambda p: p['number_value'])
except KeyError:
    print("HIBA: Úgy tűnik, a pairing.json fájl formátuma nem megfelelő.")
    print("Hiányzó kulcsok (pl. 'number_value').")
    exit()

x_coords = []
y_coords = []
numbers = []
match_types = []

for item in sorted_pairs:
    # A mi kulcsneveink használata
    dot_coord = item['matched_dot_coords']
    x_coords.append(dot_coord['x'])
    y_coords.append(dot_coord['y'])
    numbers.append(item['number_value'])

    # Mivel a mi szkriptünk csak sima párosítást végez,
    # minden párt 'matched'-nek tekintünk
    match_types.append('matched')

# Első pont hozzáadása a végéhez (zárt görbe)
x_coords.append(x_coords[0])
y_coords.append(y_coords[0])

# -- 3. Színkódolás beállítása --
# Egyszerűsített színkészlet
color_map = {
    'matched': 'green',
    'unknown': 'gray'
}

point_colors = [color_map.get(mt, 'gray') for mt in match_types]

# -- 4. Ábra létrehozása --
fig, ax = plt.subplots(figsize=(14, 12))

# Vonalak rajzolása
ax.plot(x_coords, y_coords, 'b-', linewidth=1.5, alpha=0.6, label='Összekötő vonalak')

# Pontok rajzolása színkódolással
for i, (x, y, num, color) in enumerate(zip(x_coords[:-1], y_coords[:-1], numbers, point_colors)):
    ax.plot(x, y, 'o', color=color, markersize=6, markeredgecolor='black', markeredgewidth=0.5)

    # Számok kiírása a pontok mellé
    ax.annotate(str(num), (x, y), xytext=(5, 5), textcoords='offset points',
                fontsize=7, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))

# Y tengely megfordítása (pixel koordináták)
ax.invert_yaxis()

# Címkék és cím
ax.set_xlabel('X koordináta', fontsize=12)
ax.set_ylabel('Y koordináta', fontsize=12)
ax.set_title('Connect the Dots - Párosítás vizualizáció', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.axis('equal')

# -- 5. Legenda és Statisztika --
from matplotlib.patches import Patch

# Egyszerűsített legenda
legend_elements = [
    Patch(facecolor='green', edgecolor='black', label='Matched (Párosítva)')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

# Egyszerűsített statisztika
stats_text = f"Összesen: {len(pairs)} pont párosítva"
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.show()

# -- 6. Konzol kimenet --
print(f"\n✓ Összesen {len(pairs)} pont lett összekötve.")
print(f"✓ Számok tartománya: {min(numbers)} - {max(numbers)}")

# Match type statisztika
type_counts = Counter(match_types)
print(f"\n✓ Match type bontás:")
for match_type, count in type_counts.items():
    print(f"  - {match_type}: {count}")