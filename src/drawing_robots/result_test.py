import json
import matplotlib.pyplot as plt
import os

# JSON fájl beolvasása
base_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(base_path, "Output_pictures/config/pairing.json"), 'r') as f:
    data = json.load(f)

# Ellenőrizzük az új formátumot
if isinstance(data, dict) and "pairs" in data:
    # Új formátum: metadata + pairs
    pairs = data["pairs"]
    metadata = data.get("metadata", {})
    print(f"Algoritmus: {metadata.get('algorithm', 'N/A')}")
    print(f"Statisztikák: {metadata.get('statistics', {})}\n")
else:
    # Régi formátum: csak lista
    pairs = data

# Koordináták kinyerése és rendezés szám szerint
sorted_pairs = sorted(pairs, key=lambda p: p['num_value'])

x_coords = []
y_coords = []
numbers = []
match_types = []

for item in sorted_pairs:
    dot_coord = item['dot_coord']
    x_coords.append(dot_coord['x'])
    y_coords.append(dot_coord['y'])
    numbers.append(item['num_value'])
    match_types.append(item.get('match_type', 'unknown'))

# Első pont hozzáadása a végéhez (zárt görbe)
x_coords.append(x_coords[0])
y_coords.append(y_coords[0])

# Színkódolás match_type szerint
color_map = {
    'matched': 'green',
    'predicted': 'yellow',
    'duplicate_resolved': 'orange',
    'outlier_recovered': 'magenta',
    'unknown': 'gray'
}

point_colors = [color_map.get(mt, 'gray') for mt in match_types]

# Ábra létrehozása
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
ax.set_title('Connect the Dots - Összekötött pontok', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.axis('equal')

# Legenda
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='green', edgecolor='black', label='Matched'),
    Patch(facecolor='yellow', edgecolor='black', label='Predicted'),
    Patch(facecolor='orange', edgecolor='black', label='Duplicate Resolved'),
    Patch(facecolor='magenta', edgecolor='black', label='Outlier Recovered')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

# Statisztikák kiírása
stats_text = f"Összesen: {len(pairs)} pont"
if isinstance(data, dict) and "metadata" in data:
    stats = data["metadata"].get("statistics", {})
    if stats:
        stats_text += f"\nPontosság: {stats.get('accuracy', 0)}%"
        breakdown = stats.get('breakdown', {})
        stats_text += f"\nMatched: {breakdown.get('matched', 0)}"
        stats_text += f"\nPredicted: {breakdown.get('predicted', 0)}"

ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.show()

print(f"\n✓ Összesen {len(pairs)} pont lett összekötve.")
print(f"✓ Számok tartománya: {min(numbers)} - {max(numbers)}")

# Match type statisztika
from collections import Counter
type_counts = Counter(match_types)
print(f"\n✓ Match type bontás:")
for match_type, count in type_counts.items():
    print(f"  - {match_type}: {count}")