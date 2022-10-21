import os
from glob import glob
import json
from tqdm import tqdm

data_path = "./"
img_list = glob(os.path.join(data_path, "*.png"))

file_list = []

for img_path in tqdm(img_list):
    json_path = img_path.replace(".png", ".json")

    with open(json_path, 'r') as f:
        data = json.load(f)

    width = int(data["camera"]["resolution_width"])
    height = int(data["camera"]["resolution_height"])

    txt = ""

    try:
        for ann in data["annotations"]:
            points = ann["points"]

            top_left = points[0]
            top_right = points[1]
            bottom_right = points[2]
            bottom_left = points[3]

            center_x = (top_left[0] + top_right[0]) / 2. / width
            center_y = (top_left[1] + bottom_left[1]) / 2. / height
            bounding_width = (top_right[0] - top_left[0]) / width
            bounding_height = (bottom_left[1] - top_left[1]) / height

            label = 0 if ann["label"] == "보행자" else 1

            txt += '%d %.6f %.6f %.6f %.6f\n' % (label, center_x, center_y, bounding_width, bounding_height)

        with open(os.path.basename(json_path).replace(".json", ".txt"), "w") as f:
            f.write(txt)
        
        file_list.append(os.path.basename(img_path))
    except Exception as e:
        print(e, img_path)

with open(os.path.join("./", "train.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(file_list) + "\n")

print(len(file_list))
