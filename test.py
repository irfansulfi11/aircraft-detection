from ultralytics import YOLO
import cv2
import os

def main():
    model = YOLO('runs/detect/sar_aircraft_detector/weights/best.pt')

    # List of custom .png test images
    test_images = [
        r"D:\PROJECTS\SAR-aircraft\SAR-aircraft\yolo_dataset\images\test\00072.bmp",
        r"D:\PROJECTS\SAR-aircraft\SAR-aircraft\yolo_dataset\images\test\02771.bmp"
    ]

    out_dir = 'predictions_custom'
    os.makedirs(out_dir, exist_ok=True)

    for img_path in test_images:
        img_name = os.path.basename(img_path)
        results = model(img_path, conf=0.1, iou=0.7, save=False)
        result_img = results[0].plot()
        save_path = os.path.join(out_dir, img_name)
        cv2.imwrite(save_path, result_img)

    print("✅ Inference complete on custom images. Check 'predictions_custom' folder.")

if __name__ == '__main__':
    main()
