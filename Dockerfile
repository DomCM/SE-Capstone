FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    BEHAVIOR_POSE_MODEL=models/pose/yolo26n-pose.pt \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && python -m pip install \
        --index-url https://pypi.org/simple \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1+cpu \
        torchvision==0.20.1+cpu \
        torchaudio==2.5.1+cpu \
    && python -m pip install -r requirements.txt \
    && python -m pip install --force-reinstall --no-deps \
        opencv-contrib-python-headless \
    && python -m pip check

COPY . .

RUN mkdir -p models/pose storage/instance storage/users_data \
    && python -c "from huggingface_hub import hf_hub_download; import shutil; \
shutil.copyfile(hf_hub_download('SalahALHaismawi/yolov26-fire-detection', 'best.pt', revision='5397b43994c39c2ed4ef81a6996c6334e2b7a5fa'), 'models/best.pt'); \
shutil.copyfile(hf_hub_download('Ultralytics/YOLO26', 'yolo26m.pt', revision='070ac3c51435984d992ba92eb3f234daf4e4500d'), 'models/yolo26m.pt'); \
shutil.copyfile(hf_hub_download('Ultralytics/YOLO26', 'yolo26m-pose.pt', revision='070ac3c51435984d992ba92eb3f234daf4e4500d'), 'models/pose/yolo26m-pose.pt'); \
shutil.copyfile(hf_hub_download('openvision/yoloe26-n-seg', 'model.pt', revision='86f1684b97c4f14ae1434be945be7aa172fd6987'), 'yoloe-26n-seg.pt')"

EXPOSE 5000

CMD ["python", "main.py"]
