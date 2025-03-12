import argparse
import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
from pathlib import Path
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

"""
Hyperparameters
"""
parser = argparse.ArgumentParser()
parser.add_argument("--input-dir", default="input_images")  # Folder with input images
parser.add_argument("--output-dir", default="segmented_output")  # Folder to store outputs
parser.add_argument("--grounding-model", default="IDEA-Research/grounding-dino-tiny")
parser.add_argument("--text-prompt", default="bicycle. tree. building. bench. flowers.")  # Define your class prompts
parser.add_argument("--sam2-checkpoint", default="./checkpoints/sam2.1_hiera_large.pt")
parser.add_argument("--sam2-model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
parser.add_argument("--force-cpu", action="store_true")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"

# Ensure output directory exists
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Build SAM2 image predictor
sam2_model = build_sam2(args.sam2_model_config, args.sam2_checkpoint, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# Load Grounding DINO
processor = AutoProcessor.from_pretrained(args.grounding_model)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(args.grounding_model).to(DEVICE)

# CHANGE THIS

class_score_dict = {
    "bicycle": 10,
    "bench": 10
}



# Process all images in the input directory
for img_file in Path(args.input_dir).glob("*.jpg"):  # Adjust for different formats if needed
    print(f"Processing: {img_file}")

    # Open image
    image = Image.open(img_file)
    img_array = np.array(image.convert("RGB"))

    # Set image for SAM
    sam2_predictor.set_image(img_array)

    # Process image through Grounding DINO
    inputs = processor(images=image, text=args.text_prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    # Extract detections
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )

    # Create an empty segmentation map (default all zeros)
    seg_map = np.zeros((image.height, image.width), dtype=np.uint8)

    if results and results[0]["boxes"].shape[0]:  # If objects detected
        input_boxes = results[0]["boxes"].cpu().numpy()
        class_names = results[0]["labels"]  # Detected class names

        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        # Ensure correct shape before iterating
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)  # Remove singleton dimension if present
        elif masks.ndim != 3:
            print(f"Unexpected mask shape {masks.shape} for {img_file.name}, skipping...")
            continue  # Skip this image to prevent errors

        # Assign scores to detected masks
        for mask, class_name in zip(masks, class_names):
            score = class_score_dict.get(class_name, 1)  # Default score = 1 if class not in dict
            seg_map[mask.astype(bool)] = score

    else:
        print(f"No objects detected in {img_file.name} -> Saving zero mask.")

    # Save segmentation map as grayscale image
    seg_map_path = OUTPUT_DIR / f"{img_file.stem}.png"
    cv2.imwrite(str(seg_map_path), seg_map)

    print(f"Saved segmentation map: {seg_map_path}")