
import os
import subprocess

# Mudar para a pasta do YOLOv5
os.chdir("C:/Users/Luciano/yolov5")

# Rodar inferência no vídeo com o best.pt do YOLOv5
subprocess.run([
    "python", "detect.py",
    "--weights", "runs/train/exp3/weights/best.pt",
    "--source", "C:/Users/Luciano/Downloads/GX010006.MP4",
    "--device", "0",
    "--save-txt",
    "--save-conf"
])
