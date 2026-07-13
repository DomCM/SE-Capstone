from ultralytics import YOLO

# 1. Convert your primary custom model (highly recommended to use FP16 half-precision for older CPUs)
model_primary = YOLO("last.pt")
model_primary.export(format="openvino", half=True)  # Generates folder: 'last_openvino_model/'

# 2. Convert the secondary object model
model_secondary = YOLO("yolov8n.pt")
model_secondary.export(format="openvino", half=True)  # Generates folder: 'yolov8n_openvino_model/'