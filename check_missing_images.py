
import json
import os

# Ajuste conforme o seu ambiente
SOURCE_DIR = r"C:\Users\Luciano\Desktop\Coco"
ANNOTATIONS_DIR = os.path.join(SOURCE_DIR, "annotations_trainval2017", "annotations")
DATASETS = ["train2017", "val2017"]

for dataset in DATASETS:
    print(f"📂 Verificando {dataset}...")
    json_path = os.path.join(ANNOTATIONS_DIR, f"instances_{dataset}.json")
    img_dir = os.path.join(SOURCE_DIR, dataset)

    if not os.path.exists(json_path):
        print(f"❌ JSON não encontrado: {json_path}")
        continue

    with open(json_path, "r") as f:
        coco = json.load(f)

    image_files = [img["file_name"] for img in coco["images"]]
    total = len(image_files)
    missing = []

    for file in image_files:
        path = os.path.join(img_dir, file)
        if not os.path.exists(path):
            missing.append(file)

    print(f"🔍 Total de imagens listadas no JSON: {total}")
    print(f"❌ Imagens ausentes: {len(missing)}")
    
    if missing:
        print("📉 Exemplos de imagens não encontradas:")
        print("\n".join(missing[:10]))

    print("-" * 40)
