import json
import torch

path = "/Users/sha168/Downloads/AUDD/annotations/instances_train.json"
path_out = "/Users/sha168/Downloads/AUDD/annotations/instances_train2.json"

# Opening JSON file
f = open(path)

# returns JSON object as a dictionary
data = json.load(f)

new_dict = {"images": data["images"], "categories": data["categories"], "annotations": []}

annotations = {}
for item in data["annotations"]:
    if item["image_id"] not in annotations:
        annotations[item["image_id"]] = {"image_id": item["image_id"], "id": item["id"], "boxes": [], "labels": []}
    annotations[item["image_id"]]["boxes"].append(item["bbox"])
    annotations[item["image_id"]]["labels"].append(item["category_id"])

for key in annotations.keys():
    #boxes = torch.tensor(annotations[key]["boxes"]).double()
    #labels = torch.tensor(annotations[key]["labels"])
    boxes = annotations[key]["boxes"]
    labels = annotations[key]["labels"]
    new_dict["annotations"].append({"image_id": key, "id": key, "boxes": boxes, "labels": labels})

with open(path_out, "w") as write_file:
    json.dump(new_dict, write_file)