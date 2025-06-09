
from ultralytics import YOLO

def main():
    # Caminho para o modelo base
    model = YOLO('yolov8n.pt')

    # Caminho para o dataset
    data_yaml = r"C:/Users/Luciano/Desktop/YOLODataset/data.yaml"

    # Iniciar o treino
    model.train(
        data=data_yaml,
        epochs=50,
        imgsz=640,
        batch=16,
        workers=4,
        device=0  # 0 para GPU, 'cpu' para CPU
    )

if __name__ == "__main__":
    main()
