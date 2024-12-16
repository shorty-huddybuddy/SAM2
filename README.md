
# SAM2 Integration with Stretch3 Robot and RealSense Camera

This repository contains code for integrating the **Segment Anything Model (SAM2)** with the **Stretch3 robot** using an **Intel RealSense camera** for image capture and segmentation purposes. The project involves camera access, segmentation tasks, and a parent process to synchronize multiple environments for smooth operation.

---

## Project Overview  

The objective of this project is to:  
- **Access RealSense Camera** for capturing live images or video.  
- Perform **image segmentation** using the SAM2 model.  
- **Synchronize processes** between two Python environments:  
   - **Environment 1**: For RealSense SDK (requires Python <= 3.9).  
   - **Environment 2**: For SAM2 (requires Python >= 3.10).  

---

## Folder Structure  

```plaintext
codes/
│
├── Fine_tune_img_seg.ipynb           # Notebook to fine-tune SAM2 for image segmentation tasks.
├── auto_segmentation.py              # Automates the segmentation process using SAM2.
├── automatic_mask_generator_exa...   # Example script for generating masks using SAM2.
├── camera_access.py                  # Code to access the RealSense camera for image capture.
├── depth_sensing.py                  # Depth-sensing functionality using the RealSense camera.
├── image_predictor_example.ipynb     # Example notebook for predicting segmented outputs.
├── parent.py                         # Parent process for synchronizing two processes.
├── segmentation.py                   # Code for performing segmentation on captured data.
├── train_test_yolo.ipynb             # YOLO model training and testing integration (for comparison).
├── use_trained_model.ipynb           # Notebook to use a pre-trained SAM2 model for segmentation.
└── video_predictor_example.ipynb     # Example for performing segmentation on video inputs.
```

---

## External Image Dataset  

Due to the large size of the dataset, it is hosted externally.  

You can download the image dataset from the following link:  
[**Image Dataset Repository**](https://drive.google.com/drive/folders/1YyUG3ZhPNVQTlesTLxuc3DOpR6LcFvXv?usp=drive_link)

Once downloaded, place the dataset in a folder named `dataset/` at the root of the project.

---

## Setup and Installation  

### Prerequisites  
Ensure the following tools and libraries are installed:  
- **Python 3.9** (for RealSense SDK and PyRealSense2)  
- **Python 3.10+** (for SAM2)  
- **Intel RealSense SDK**  
- **PyRealSense2** for camera interfacing  
- **Segment Anything Model (SAM2)**  
- **OpenCV** for image and video processing  

---

### Step 1: Create Two Python Environments  

We need two separate environments:  
1. **Environment 1 (Python 3.9)**: For RealSense SDK.  
2. **Environment 2 (Python 3.10)**: For SAM2.  

#### Create Environment 1 (For RealSense SDK)  
```bash
conda create -n realsense_env python=3.9
conda activate realsense_env
pip install pyrealsense2 opencv-python numpy
```

#### Create Environment 2 (For SAM2 Integration)  
```bash
conda create -n sam2_env python=3.10
conda activate sam2_env
pip install torch torchvision segment-anything-model matplotlib opencv-python numpy
```

---

### Step 2: Verify Installation  

1. **Test RealSense Camera**:  
   Run the `camera_access.py` script in the **`realsense_env`** environment:  
   ```bash
   conda activate realsense_env
   python camera_access.py
   ```

2. **Test SAM2 Integration**:  
   Run the `segmentation.py` script in the **`sam2_env`** environment:  
   ```bash
   conda activate sam2_env
   python segmentation.py
   ```

---

### Step 3: Synchronize Processes with `parent.py`  

The `parent.py` script ensures that both environments work together by synchronizing the processes:  
- **RealSense camera** captures images in Environment 1.  
- **SAM2 model** performs segmentation in Environment 2.  

#### Run Synchronization Script  
```bash
conda activate realsense_env
python parent.py
```

The parent script will manage both processes. Ensure that both environments (`realsense_env` and `sam2_env`) are correctly configured.

---

## Usage  

### 1. **Accessing the Camera**  
Use `camera_access.py` to capture real-time images or videos from the RealSense D435i camera.  

### 2. **Running Image Segmentation**  
To perform SAM2-based segmentation, activate the SAM2 environment and run:  
```bash
conda activate sam2_env
python segmentation.py
```

### 3. **Synchronizing Both Processes**  
Run `parent.py` in the **`realsense_env`** environment to synchronize RealSense and SAM2:  
```bash
conda activate realsense_env
python parent.py
```

---

## Key Features  

- **Dual Environment Management**: Separate environments for RealSense (<=3.9) and SAM2 (>=3.10).  
- **Real-Time Camera Integration**: Access the Intel RealSense camera for live image and depth sensing.  
- **SAM2 Integration**: Perform robust segmentation on captured images and videos.  
- **Process Synchronization**: Ensure seamless coordination between image capture and segmentation using a parent process.  
- **Extensibility**: Includes additional scripts for YOLO-based segmentation for comparison.  

---

## Dependencies  

- Python Libraries:  
  - PyRealSense2 (for Python 3.9)  
  - OpenCV  
  - NumPy  
  - PyTorch (for Python 3.10+)  
  - SAM2 Libraries  
- Intel RealSense SDK  

Install dependencies using:  
```bash
pip install -r requirements.txt
```

---

## Example Workflow  

1. **Capture Images** (Environment 1):  
   Run `camera_access.py` to capture input images using RealSense.  

2. **Segment Images** (Environment 2):  
   Use `segmentation.py` to apply SAM2 for segmentation.  

3. **Synchronize Operations**:  
   Run `parent.py` to coordinate camera input and SAM2 processing.  

4. **Visualize Results**:  
   Display segmented outputs using `video_predictor_example.ipynb`.  

---

## Acknowledgments  

- **Intel RealSense** for powerful depth-sensing capabilities.  
- **Meta's SAM2** model for robust segmentation.  
- The **Stretch3 Robot** for enabling real-world robotic integration.  

---

## Contributions  

Contributions are welcome! Please create a pull request or open an issue to suggest improvements.  

---

## Contact  

For queries or collaborations, please contact:  
**Group Delta**  
**Email**: iit2022219@iiita.ac.in / iit20222218@iiita.ac.in  /iit20222192@iiita.ac.in  /iit20222194@iiita.ac.in  /iit20222170@iiita.ac.in  
**GitHub**: [Your GitHub Profile](https://github.com/shorty-huddybuddy)
```
