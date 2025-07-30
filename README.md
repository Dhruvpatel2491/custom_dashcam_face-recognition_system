# Dashcam - Driver Identity Recognition

A Python-based facial recognition pipeline designed for dashcam image processing, including dataset creation, encoding, and recognition.  
Supports local workflows and AWS Lambda deployment.

---

## Table of Contents

- [Installation (Local System)](#installation-local-system)
- [Local Workflow](#local-workflow)
- [Technical Details](#technical-details)
- [AWS Lambda: EncodeGenerator](#aws-lambda-encodegenerator)
- [Other Resources](#other-resources)

---

## Installation (Local System)

1. **Install Python 3.7**  
   [Download Python 3.7.0](https://www.python.org/downloads/release/python-370/)

2. **Install Dependencies**  
   Run the setup script to automatically install required packages:  
python installation/setup.py

yaml
Copy
Edit
*Ensure that `dlib` is present in the `installation` folder before running this step.*

---

## Local Workflow

There are three main scripts to run in order:

### 1. `dataset_generation_local.py`
- **Purpose:** Capture face images for the dataset.
- **Usage:**
-
- python dataset_generation_local.py
- **Instructions:**  
- Enter the ID/name when prompted.
- Press **SPACE** to capture each photo (take multiple shots from different angles).

### 2. `encodeGenerator_local.py`
- **Purpose:** Generate facial encodings from the captured images.
- **Usage:**  

python encodeGenerator_local.py
- **Instructions:**  
- This will fetch all collected images and generate an encoding `.p` (pickle) file.

### 3. `faceRecognition_local.py`
- **Purpose:** Perform facial recognition using the generated encodings.
- **Usage:**  
python faceRecognition_local.py

- **Instructions:**  
- This script will process images and recognize faces using the encoding file.

---

## Technical Details

- **Python Version:** 3.7.0

- **Key Packages:**
- `os`
- `numpy`
- `opencv-python`
- `face_recognition` ([GitHub link](https://github.com/ageitgey/face_recognition))
  - `face_recognition_models`
  - `Click>=6.0`
  - `dlib>=19.3.0` ([dlib build guide](https://github.com/sachadee/Dlib/tree/main))
  - `Pillow`
  - `scipy>=0.17.0`

- **Alternative for AWS Lambda Layers (ARNs):**  
[KLayers Repository](https://github.com/keithrozario/Klayers/tree/master)

---

## AWS Lambda: EncodeGenerator

For cloud deployment, add these key-value pairs as environment variables in your Lambda function configuration:

| **Key**                    | **Value**        |
|----------------------------|------------------|
| `enhanced_img_bucket`      | temp-dashcam     |
| `enhanced_img_folder`      | enhanced-img     |
| `source_img_bucket`        | temp-dashcam     |
| `source_img_folder`        | dataset-img      |
| `encoding_storing_bucket`  | temp-dashcam     |
| `encoding_storing_folder`  | encoding         |

---

## Other Resources

- **Image Enhancement Reference:**  
[Image enhancement techniques using OpenCV](https://www.geeksforgeeks.org/image-enhancement-techniques-using-opencv-python/)

---
