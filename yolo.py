from ultralytics import YOLO
import torch

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO('yolov8n.yaml')  # or yolov8s.yaml

    model.train(
        data='data.yaml',
        epochs=10,
        imgsz=640,
        batch=16,
        device=device,
        name='sar_aircraft_detector',
        patience=10,
        lr0=0.0005,
        optimizer='SGD',
        augment=True,
        cos_lr=True,
    )

if __name__ == '__main__':
    main()
