"""
OpenVINO FP16 Converter & INT8 Quantization Pipeline for YOLOv26 / YOLOv8.
Optimized for Smart Security Applications running on CPU / Intel Integrated GPUs.
"""

import os
from pathlib import Path
from ultralytics import YOLO

# Define model paths
PRIMARY_PYTORCH_MODEL = "best.pt"          # E.g., custom YOLOv26n trained on fire/smoke
SECONDARY_PYTORCH_MODEL = "yolo26n.pt"     # General COCO baseline for tracking people

# Calibration dataset configuration (Required for INT8 calibration to avoid accuracy drop)
# This should point to a small subset of your training data (~100-300 images)
CALIBRATION_DATA_YAML = "coco8.yaml"       # Fallback to default, replace with your custom data.yaml


def convert_to_fp16(model_path: str, output_name: str):
    """
    Converts a PyTorch model to OpenVINO FP16 half-precision.
    Highly recommended for all edge targets. FP16 offers near-identical accuracy 
    to FP32 with 2x speed and half the memory footprint.
    """
    print(f"\n--- [FP16 Export] Converting {model_path} ---")
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Skipping.")
        return None

    model = YOLO(model_path)
    
    # YOLOv26 is end-to-end NMS-free by design. 
    # For YOLOv26, 'end2end=True' is mandatory.
    # 'dynamic=False' generates fixed 640x640 inputs, offering better performance on NPU/iGPU pipelines.
    export_dir = model.export(
        format="openvino",
        half=True,
        dynamic=False,
        end2end=True
    )
    
    print(f"Success! FP16 OpenVINO model saved to: {export_dir}")
    return export_dir


def convert_to_int8_native(model_path: str, data_yaml: str):
    """
    Quantizes a PyTorch model to INT8 using the native Ultralytics API.
    Uses a training/validation dataset (yaml) to calibrate the weights,
    minimizing accuracy degradation.
    """
    print(f"\n--- [INT8 Quantization] Converting {model_path} ---")
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Skipping.")
        return None
        
    if not os.path.exists(data_yaml):
        print(f"Warning: Calibration dataset config '{data_yaml}' not found.")
        print("Standard random weight calibration will be used, which may hurt accuracy.")
        # We can still attempt export, or fallback. It's safer to provide a small validation dataset.

    model = YOLO(model_path)
    
    # Export with integer quantization. The 'int8=True' parameter tells the exporter
    # to quantize operations while using 'data' configuration for calibration.
    export_dir = model.export(
        format="openvino",
        int8=True,
        data=data_yaml,
        dynamic=False,
        end2end=True
    )
    
    print(f"Success! INT8 Quantized model saved to: {export_dir}")
    return export_dir


def main():
    print("=========================================================")
    print("      YOLO to OpenVINO Optimization Pipeline (2026)      ")
    print("=========================================================")

    # 1. PRIMARY MODEL: FP16 (Highly recommended for Custom/Critical alerts like Fire/Smoke)
    # Keeping this in FP16 guarantees no accuracy drop on complex, fine textures.
    primary_ov_fp16 = convert_to_fp16(PRIMARY_PYTORCH_MODEL, "primary_fp16")

    # 2. SECONDARY MODEL: INT8 (Highly recommended for General People Tracking)
    # Since tracking people uses standard high-contrast boundaries, INT8 is ideal here.
    secondary_ov_int8 = convert_to_int8_native(SECONDARY_PYTORCH_MODEL, CALIBRATION_DATA_YAML)

    # Alternate/Fallback options:
    # If your primary model's inference is still taking too long on your CPU, 
    # uncomment the line below to quantize the primary model as well.
    # primary_ov_int8 = convert_to_int8_native(PRIMARY_PYTORCH_MODEL, "your_custom_data.yaml")

    print("\n=========================================================")
    print("Processing complete. Keep your folder names aligned with main.py:")
    print(" - Primary Model Folder:   'last_openvino_model/'")
    print(" - Secondary Model Folder: 'yolov8n_openvino_model/' or 'yolo26n_int8_openvino_model/'")
    print("=========================================================")


if __name__ == "__main__":
    main()