
from ultralytics import YOLO
import os

def main():
    model_path = r"C:/Users/Luciano/runs/detect/train7/weights/best.pt"
    video_path = r"C:/Users/Luciano/Downloads/GX010006.MP4"

    if not os.path.exists(model_path):
        print(f"❌ Modelo não encontrado: {model_path}")
        return

    if not os.path.exists(video_path):
        print(f"❌ Vídeo não encontrado: {video_path}")
        return

    print("✅ Modelo e vídeo encontrados. Iniciando predição padrão YOLO...")

    model = YOLO(model_path)

    results = model.predict(
        source=video_path,
        save=True,
        stream=True,
        show=False,
        imgsz=480,
        verbose=True
    )

    print("\n📝 Resumo de detecções:")
    for i, r in enumerate(results):
        print(f"Frame {i+1}: {r.boxes.shape[0]} objetos detectados")

    print("\n✅ Predição finalizada.")
    print("📂 Verifique a pasta: C:/Users/Luciano/runs/detect/predict/ (ou predict2, predict3...)")

if __name__ == "__main__":
    main()
