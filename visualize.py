from ultralytics import YOLO
import cv2
import os

def main():
    model = YOLO('runs/detect/sar_aircraft_detector/weights/best.pt')

    img_path = r'D:\PROJECTS\SAR-aircraft\SAR-aircraft\yolo_dataset\images\test\02771.bmp'
    out_dir = 'predictions'
    os.makedirs(out_dir, exist_ok=True)

    results = model(img_path, conf=0.1, iou=0.7, save=False)
    result_img = results[0].plot()

    # Save result
    img_name = os.path.basename(img_path)
    cv2.imwrite(os.path.join(out_dir, img_name), result_img)

    print("✅ Inference complete. Check 'predictions' folder.")

if __name__ == '__main__':
    main()
