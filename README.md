# Deepfake-Detection
CSC 4810 Artificial Intelligence Final Project

## Dataset
Data pulled from [Kaggle's Deepfake Detection Challenge](https://www.kaggle.com/c/deepfake-detection-challenge/data). The data in question is very large, but it can be downloaded as smaller chunks for ease of use. If space is a concern, downloading only from the data explorer, which only holds about 800 files, may be best. 

## User Guide 
### Directory Setup 
Place all downloaded videos, as well as the associated metadata.json file, in the dataset directory in a folder labeled train_sample_videos. 

### Order to Run Program Files
Begin by running video_to_image.py. This will create subdirectories within train_sample_videos containing frames from videos within the dataset of the same name. 

Next, run crop_faces.py to create a faces subdirectory within each video's associated folder of images containing cropped face images. 

Next, run create_dataset.py to create the final split_dataset directory, on the same level as the original dataset directory, which is what the model will use to train and test against. 

Finally, run deepfake_detection.py to run the program which builds, trains, and evaluates the model. 

### Package Installation 
OpenCV: pip install opencv-python

Splitfolders: pip install split-folders

PyTorch,Torchvision: pip install torch torchvision (for cuda support visit [here](https://pytorch.org/get-started/locally/#windows-pip) to find proper installation URL) 

Keras: pip install keras

TensorFlow: pip install tensorflow

Shutil: pip install pytest-shutil

MTCNN: pip install mtcnn
