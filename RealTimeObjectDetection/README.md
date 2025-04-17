# Sign Language Translator

A real-time web application that translates sign language gestures into readable text using a deep learning model. Built using TensorFlow's Object Detection API, this project aims to bridge the communication gap for deaf and hard-of-hearing individuals by interpreting both static and dynamic sign gestures through a webcam or video upload.

# 📦 Tech Stack

Framework: TensorFlow 2.10.0 (under Rosetta for macOS compatibility)

Object Detection: TensorFlow Object Detection API

Language: Python 3.9

Model: SSD MobileNet V2 FPNLite 320x320

Tools: LabelImg, OpenCV, Jupyter Notebook

Environment: Miniforge + Rosetta (x86_64 on Apple Silicon)

Annotation Format: Pascal VOC XML

# 🚀 Features

- Detects and classifies ASL gestures in real-time from webcam or video
- Supports static signs (e.g., alphabet) and extendable to dynamic gestures
- Converts recognized signs to readable text on the interface
- Easily retrainable for custom gestures
- Clean modular pipeline for dataset generation, training, and inference

# 🛠️ Getting Started

# 🔧 Prerequisites

- macOS (Apple Silicon) running under Rosetta
- Miniforge installed with Python 3.9
- protoc installed via Homebrew
- Webcam access (for real-time testing)

# ✅ Setup Instructions

- Clone the Repository
    git clone https://github.com/your-username/sign-language-translator.git
    cd sign-language-translator

- Create the Environment (via Rosetta Terminal)
    conda create -n tf-od-2.10 python=3.9
    conda activate tf-od-2.10

- Install TensorFlow and Dependencies
    pip install tensorflow==2.10.0 tensorflow-estimator==2.10.0
    pip install protobuf==3.20.3 opencv-python pillow pandas matplotlib

- Install Object Detection API
    cd Tensorflow/models/research
    protoc object_detection/protos/*.proto --python_out=.
    cp object_detection/packages/tf2/setup.py .
    python -m pip install .

- Verify Installation
    python -c "from object_detection.utils import label_map_util; print('✅ Object Detection API installed!')"

# 🎯 Model Training

- Prepare Dataset
- Annotate images with LabelImg
- Export annotations as Pascal VOC XML
- Create label_map.pbtxt and image folders under Tensorflow/workspace/images/train and test
- Generate TFRecords
    - !python Tensorflow/scripts/generate_tfrecord.py -x Tensorflow/workspace/images/train -l Tensorflow/workspace/annotations/label_map.pbtxt -o Tensorflow/workspace/annotations/train.record
    !python Tensorflow/scripts/generate_tfrecord.py -x Tensorflow/workspace/images/test -l Tensorflow/workspace/annotations/label_map.pbtxt -o Tensorflow/workspace/annotations/test.record
- Start Training
    - python Tensorflow/models/research/object_detection/model_main_tf2.py \
    --model_dir=Tensorflow/workspace/models/my_ssd_mobnet \
    --pipeline_config_path=Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config \
    --num_train_steps=20000

# 📂 Project Structure

RealTimeObjectDetection/
├── Tensorflow/
│   ├── models/
│   ├── scripts/
│   ├── workspace/
│       ├── annotations/
│       ├── images/
│       ├── models/
│       └── pre-trained-models/
├── labelImg/
├── generate_tfrecord.py

# ✨ Contributions

This project is open to community improvements! Feel free to fork the repository, improve the model, or add new gesture classes. Raise an issue or pull request and let’s build an accessible future together!