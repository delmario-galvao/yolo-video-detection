
import os
import json
from tqdm import tqdm

# Caminhos do dataset COCO filtrado
COCO_ANNOTATIONS_DIR = r"C:\Users\Luciano\Desktop\FilteredCoco\annotations"
COCO_IMAGES_DIR = r"C:\Users\Luciano\Desktop\FilteredCoco"

# Caminho de saída no formato YOLO
YOLO_OUTPUT_DIR = r"C:\Users\Luciano\Desktop\YOLODataset"
os.makedirs(YOLO_OUTPUT_DIR, exist_ok=True)

# Função para converter
def convert_coco_to_yolo(dataset_type):
    json_path = os.path.join(COCO_ANNOTATIONS_DIR, f"filtered_instances_{dataset_type}.json")
    image_dir = os.path.join(COCO_IMAGES_DIR, dataset_type)
    label_dir = os.path.join(YOLO_OUTPUT_DIR, "labels", dataset_type)
    yolo_image_dir = os.path.join(YOLO_OUTPUT_DIR, "images", dataset_type)

    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(yolo_image_dir, exist_ok=True)

    with open(json_path, "r") as f:
        coco = json.load(f)

    # Mapear categorias (id para índice 0-N)
    categories = coco["categories"]
    cat_id_to_index = {cat["id"]: i for i, cat in enumerate(categories)}

    images = {img["id"]: img for img in coco["images"]}

    annotations_by_image = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    for img_id, anns in tqdm(annotations_by_image.items(), desc=f"Convertendo {dataset_type}"):
        img_info = images[img_id]
        file_name = img_info["file_name"]
        width = img_info["width"]
        height = img_info["height"]

        # Copiar imagem para nova pasta
        src_img_path = os.path.join(image_dir, file_name)
        dst_img_path = os.path.join(yolo_image_dir, file_name)
        if os.path.exists(src_img_path):
            with open(src_img_path, "rb") as src_file:
                with open(dst_img_path, "wb") as dst_file:
                    dst_file.write(src_file.read())

        # Criar .txt com anotações
        label_path = os.path.join(label_dir, file_name.replace(".jpg", ".txt"))
        with open(label_path, "w") as f:
            for ann in anns:
                cat_id = ann["category_id"]
                if cat_id not in cat_id_to_index:
                    continue
                category_idx = cat_id_to_index[cat_id]
                x, y, w, h = ann["bbox"]
                # converter para YOLO format (center x/y, width/height) normalizado
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                w_norm = w / width
                h_norm = h / height
                f.write(f"{category_idx} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

# Criar arquivo data.yaml
def create_data_yaml(class_names):
    yaml_path = os.path.join(YOLO_OUTPUT_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write("names: [" + ", ".join([f'"{name}"' for name in class_names]) + "]\n")
    print(f"✅ Arquivo data.yaml criado em: {yaml_path}")

if __name__ == "__main__":
    convert_coco_to_yolo("train2017")
    convert_coco_to_yolo("val2017")

    # Lê classes direto do COCO filtrado
    json_train = os.path.join(COCO_ANNOTATIONS_DIR, "filtered_instances_train2017.json")
    with open(json_train, "r") as f:
        categories = json.load(f)["categories"]
        class_names = [cat["name"] for cat in categories]

    create_data_yaml(class_names)
    print("🎉 Conversão concluída!")
