
import json
import os
import shutil
from tqdm import tqdm

# Caminhos no seu computador (corrigidos)
ROOT_DIR = r"C:\Users\Luciano\Desktop\Coco\annotations_trainval2017"
ANNOTATIONS_DIR = os.path.join(ROOT_DIR, "annotations")
IMAGE_DIR = ROOT_DIR  # imagens estão dentro deste mesmo diretório

DEST_DIR = r"C:\Users\Luciano\Desktop\FilteredCoco"
TARGET_CATEGORY_IDS = {1, 2, 3, 4, 6, 8, 18}
DATASETS = ["train2017", "val2017"]

# Criação das pastas de destino
os.makedirs(os.path.join(DEST_DIR, "annotations"), exist_ok=True)
for ds in DATASETS:
    os.makedirs(os.path.join(DEST_DIR, ds), exist_ok=True)

def filter_coco_dataset(dataset):
    print(f"🔍 Processando {dataset}...")

    ann_path = os.path.join(ANNOTATIONS_DIR, f"instances_{dataset}.json")
    img_dir = os.path.join(IMAGE_DIR, dataset)
    dest_img_dir = os.path.join(DEST_DIR, dataset)
    dest_ann_path = os.path.join(DEST_DIR, "annotations", f"filtered_instances_{dataset}.json")

    with open(ann_path, "r") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    filtered_annotations = [ann for ann in annotations if ann["category_id"] in TARGET_CATEGORY_IDS]
    image_ids = {ann["image_id"] for ann in filtered_annotations}
    filtered_images = [img for img in images if img["id"] in image_ids]

    print(f"📁 Copiando {len(filtered_images)} imagens para {dest_img_dir}...")
    for img in tqdm(filtered_images):
        src = os.path.join(img_dir, img["file_name"])
        dst = os.path.join(dest_img_dir, img["file_name"])
        if os.path.exists(src):
            shutil.copyfile(src, dst)
        else:
            print(f"⚠️ Imagem não encontrada: {src}")

    filtered_coco = {
        "images": filtered_images,
        "annotations": filtered_annotations,
        "categories": [cat for cat in categories if cat["id"] in TARGET_CATEGORY_IDS]
    }

    with open(dest_ann_path, "w") as f:
        json.dump(filtered_coco, f)

    print(f"✅ {dataset} processado e salvo em: {dest_ann_path}\n")

if __name__ == "__main__":
    for ds in DATASETS:
        filter_coco_dataset(ds)
    print("🎉 Filtragem concluída!")
