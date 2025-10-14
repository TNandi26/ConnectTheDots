import json
import matplotlib.pyplot as plt
import os

# JSON fájl beolvasása
base_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(base_path, "Examples/200/config/pairing.json"), 'r') as f:
    data = json.load(f)

# Koordináták kinyerése
x_coords = []
y_coords = []

for item in data:
    dot_coord = item['dot_coord']
    x_coords.append(dot_coord['x'])
    y_coords.append(dot_coord['y'])

# Első pont hozzáadása a végéhez (zárt görbe)
x_coords.append(x_coords[0])
y_coords.append(y_coords[0])

# Ábra létrehozása
plt.figure(figsize=(12, 10))
plt.plot(x_coords, y_coords, 'b-', linewidth=1)
plt.plot(x_coords[:-1], y_coords[:-1], 'ro', markersize=3)  # pontok jelölése
plt.gca().invert_yaxis()  # Y tengely megfordítása (pixel koordináták)
plt.xlabel('X koordináta')
plt.ylabel('Y koordináta')
plt.title('Összekötött pontok')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.show()

print(f"Összesen {len(data)} pont lett összekötve.")