# Dashcam_Facial_Recognition

## Step to install 
### Local System

1. Download and Install Python 3.7 [https://www.python.org/downloads/release/python-370/]
2. Run `installation/setup.py` for downloading Dependences/Library (make sure dlib file is present in installation folder)

3. Find `dataset_generation_local.py` , `encodeGenerator_local.py` & `faceRecognition_local.py` to run the flow and flow below given order
4. Run using `python file_name.py` (e.g. : dataset_generation_local.py)
#### Local File Workflow

1. `dataset_generation_local.py` : Enter ID name and Press SPACE to take photos (take multiple shots with different angle).
2. `encodeGenerator_local.py` : It will fetch images taken and will generate an encoding ".p"(pickle) file
3. `faceRecognition_local.py` : It will process the encoding file and recognize the images. 

### Technical Details
- Python Version: 3.7.0

- Package 
  - os
  -  numpy
  -  opencv-python
  -  face_recognition (https://github.com/ageitgey/face_recognition)
     -  face_recognition_models
     -  Click>=6.0
     -  dlib>=19.3.0 (https://github.com/sachadee/Dlib/tree/main)
     -  Pillow
     -  scipy>=0.17.0

Alternative for Layers ARNS : https://github.com/keithrozario/Klayers/tree/master 

## EncodeGenrator Lambda Function
Add below (Key, Value) in `Configurations/Environment variables`.

**This is Example for Building and Testing**
| **Key**                     | **Value**        |
| ----------------------  | -----------  |
|enhanced_img_bucket      | temp-dashcam |
|enhanced_img_folder      | enhanced-img |
|source_img_bucket        | temp-dashcam |
|source_img_folder        | dataset-img  |
|encoding_storing_bucket  | temp-dashcam |
|encoding_storing_folder  | encoding     |

### Other Resources
Image Correction used in `encodeGenerator_local.py` : https://www.geeksforgeeks.org/image-enhancement-techniques-using-opencv-python/