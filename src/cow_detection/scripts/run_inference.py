import argparse
from pathlib import Path
from cow_detection.configs import DetectionConfig
from cow_detection.inference import YOLOInference
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Run cow detection inference")
    parser.add_argument('--image', type=str, required=True, 
                        help='Path to input image')
    parser.add_argument('--output', type=str, default='output.jpg', 
                        help='Path to output image')
    parser.add_argument('--weights', type=str, default='runs/detect/train14/weights/best.pt', 
                        help='Path to model weights')
    
    args = parser.parse_args()

    detection_config = DetectionConfig(
        conf_threshold=0.4,
        iou_threshold=0.70,
        min_area=300,
        max_overlap_ratio=0.65,
        enable_temporal=False,
    )

    detector = YOLOInference(
        "yolov9c.pt",
        args.weights,
        detection_config=detection_config,
    )

    image, results = detector.predict_image(
        args.image,
        save_path=args.output,
    )

    print(f"Found {len(results.boxes) if results.boxes is not None else 0} detections")
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.title('Cow Detection Results')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()