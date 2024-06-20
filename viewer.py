import json
import matplotlib.pyplot as plt
from collections import defaultdict

with open('output.json', 'r') as file:
    data = json.load(file)

# フレームごとに認識されたオブジェクトを格納する辞書
objects_by_frame = defaultdict(list)

for entry in data:
    index = entry["index"]
    if "objects" in entry["output"]:
        for obj in entry["output"]["objects"]:
            if "name" in obj:
                objects_by_frame[index].append(obj["name"])
            else:
                print(f"No name found for object in frame {index}")

# 時系列データの準備
frames = sorted(objects_by_frame.keys())
object_names = sorted({obj for objs in objects_by_frame.values() for obj in objs})

object_index = {obj: i for i, obj in enumerate(object_names)}

fig, ax = plt.subplots(figsize=(15, 20)) 

# 各オブジェクトの認識を時系列でプロット
for obj, i in object_index.items():
    times = [frame for frame in frames if obj in objects_by_frame[frame]]
    ax.plot(times, [i] * len(times), 'o-', label=obj)

ax.set_xlabel('Frame Index', fontsize=14)
ax.set_ylabel('Recognized Objects', fontsize=14)
ax.set_title('Objects Recognized Over Time', fontsize=16)

ax.set_ylim(-1, len(object_names))

ax.set_yticks(range(len(object_names)))
ax.set_yticklabels(object_names, fontsize=8)

ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

plt.tight_layout(pad=3.0)
plt.subplots_adjust(left=0.1, right=0.8)  # 余白を追加

plt.show()
