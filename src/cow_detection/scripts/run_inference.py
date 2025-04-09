import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
import tqdm

from cow_detection.configs import DetectionConfig
from cow_detection.inference import YOLOInference


def process_single_image(
    detector: YOLOInference,
    image_path: str,
    output_path: Optional[str] = None,
    render_boxes: bool = True,
):
    """
    Process a single image with the detector.

    Args:
        detector: The YOLOInference object
        image_path: Path to the input image
        output_path: Path to save the output image (if render_boxes is True)
        render_boxes: Whether to render bounding boxes on the image

    Returns:
        If render_boxes is True: Tuple of (image, results)
        If render_boxes is False: Detection summary dictionary
    """
    result = detector.predict_image(
        image_path=image_path,
        save_path=output_path if render_boxes else None,
        render_boxes=render_boxes,
    )

    if render_boxes:
        image, results = result
        detection_count = len(results.boxes) if results.boxes is not None else 0
        print(f"Found {detection_count} detections in {Path(image_path).name}")
        return image, results
    else:
        print(f"Image: {Path(image_path).name}")
        print(f"  Detections: {result['detection_count']}")
        if result["detection_count"] > 0:
            for i, prob in enumerate(result["probabilities"], 1):
                print(f"  Detection {i}: probability {prob:.2f}")
        return result


def process_directory(
    detector: YOLOInference,
    input_dir: str,
    output_dir: Optional[str] = None,
    render_boxes: bool = True,
    image_extensions: List[str] = [".jpg", ".jpeg", ".png"],
) -> List[Dict[str, Any]]:
    """
    Process all images in a directory.

    Args:
        detector: The YOLOInference object
        input_dir: Path to the input directory containing images
        output_dir: Path to save output images (if render_boxes is True)
        render_boxes: Whether to render bounding boxes on the images
        image_extensions: List of valid image file extensions to process

    Returns:
        List of detection results for all processed images
    """
    input_path = Path(input_dir)
    image_paths = []

    # Find all valid image files
    for ext in image_extensions:
        image_paths.extend(list(input_path.glob(f"*{ext}")))
        image_paths.extend(list(input_path.glob(f"*{ext.upper()}")))

    if not image_paths:
        print(f"No images found in {input_dir} with extensions {image_extensions}")
        return []

    # Create output directory if it doesn't exist and rendering is enabled
    if render_boxes and output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = []

    # Process each image
    for img_path in tqdm.tqdm(image_paths):
        if render_boxes and output_dir:
            output_path = (
                Path(output_dir) / f"{img_path.stem}_detected{img_path.suffix}"
            )
        else:
            output_path = None

        result = process_single_image(
            detector=detector,
            image_path=str(img_path),
            output_path=str(output_path) if output_path else None,
            render_boxes=render_boxes,
        )

        # For summary statistics
        if not render_boxes:
            results.append({"image_name": img_path.name, **result})

    # Print summary if not rendering boxes
    if not render_boxes and results:
        total_detections = sum(r["detection_count"] for r in results)
        avg_detections = total_detections / len(results)
        print(f"\nSummary:")
        print(f"  Processed {len(results)} images")
        print(f"  Total detections: {total_detections}")
        print(f"  Average detections per image: {avg_detections:.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run cow detection inference")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=str, help="Path to input image")
    input_group.add_argument("--dir", type=str, help="Path to directory of images")

    parser.add_argument(
        "--output",
        type=str,
        default="output.jpg",
        help="Path to output image (for single image) or output directory (for directory input)",
    )

    parser.add_argument(
        "--weights",
        type=str,
        default="runs/detect/train14/weights/best.pt",
        help="Path to model weights",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="yolov9c.pt",
        help="Base model architecture",
    )

    # Inference options
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold")
    parser.add_argument(
        "--min-area", type=float, default=300, help="Minimum detection area"
    )

    # Rendering options
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Don't render bounding boxes, only output detection information",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show results using matplotlib (only for single image and when rendering)",
    )

    args = parser.parse_args()

    # Configure detector
    detection_config = DetectionConfig(
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        min_area=args.min_area,
        max_overlap_ratio=0.65,
        enable_temporal=False,
    )

    detector = YOLOInference(
        args.model,
        args.weights,
        detection_config=detection_config,
    )

    render_boxes = not args.no_render
    if args.image:
        result = process_single_image(
            detector=detector,
            image_path=args.image,
            output_path=args.output if render_boxes else None,
            render_boxes=render_boxes,
        )

        if render_boxes and args.show:
            image, _ = result
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            plt.axis("off")
            plt.title("Cow Detection Results")
            plt.tight_layout()
            plt.show()
    else:
        output_dir = args.output
        if output_dir.endswith((".jpg", ".jpeg", ".png")):
            output_dir = os.path.dirname(output_dir)
            if not output_dir:
                output_dir = "output"

        process_directory(
            detector=detector,
            input_dir=args.dir,
            output_dir=output_dir if render_boxes else None,
            render_boxes=render_boxes,
        )


if __name__ == "__main__":
    main()
