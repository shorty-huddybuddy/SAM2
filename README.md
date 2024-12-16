
# SAM2 Integration with Stretch3 Robot and RealSense Camera

This repository contains code for integrating the **Segment Anything Model (SAM2)** with the **Stretch3 robot** using an **Intel RealSense camera** for image capture and segmentation purposes. The project involves camera access, segmentation tasks, and a parent process to synchronize multiple processes for smooth operation.

---

## Project Overview

The objective of this project is to:  
- **Access RealSense Camera** for capturing live images or video.  
- Perform **image segmentation** using the SAM2 model.  
- **Synchronize processes** between camera operations and segmentation tasks using a parent process.

---

## Folder Structure  

```
codes/
│
├── Fine_tune_img_seg.ipynb           # Notebook to fine-tune SAM2 for image segmentation tasks.
├── auto_segmentation.py             # Automates the segmentation process using SAM2.
├── automatic_mask_generator_exa...  # Example script for generating masks using SAM2.
├── camera_access.py                 # Code to access the RealSense camera for image capture.
├── depth_sensing.py                 # Depth-sensing functionality using the RealSense camera.
├── image_predictor_example.ipynb    # Example notebook for predicting segmented outputs.
├── parent.py                        # Parent process for synchronizing two processes.
├── segmentation.py                  # Code for performing segmentation on captured data.
├── train_test_yolo.ipynb            # YOLO model training and testing integration (for comparison).
├── use_trained_model.ipynb          # Notebook to use a pre-trained SAM2 model for segmentation.
└── video_predictor_example.ipynb    # Example for performing segmentation on video inputs.
```

---

## Setup and Installation  

### Prerequisites  
Ensure the following tools and libraries are installed:  
- **Python 3.8+**  
- **Intel RealSense SDK**  
- **PyRealSense2** for camera interfacing  
- **Segment Anything Model (SAM2)**  
- **OpenCV** for image and video processing  

### Installation Steps  
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/sam2-stretch3-integration.git
   cd sam2-stretch3-integration/codes
   ```

2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

3. Verify RealSense camera connection:  
   Run `camera_access.py` to confirm camera functionality.  
   ```bash
   python camera_access.py
   ```

---

## Usage  

### 1. **Accessing the Camera**  
Use `camera_access.py` to capture real-time images or videos from the RealSense D435i camera.  

### 2. **Running Image Segmentation**  
To perform SAM2-based segmentation:  
   ```bash
   python segmentation.py
   ```

### 3. **Synchronizing Processes**  
The `parent.py` script synchronizes the camera access and segmentation processes for smooth operation. Run it as follows:  
   ```bash
   python parent.py
   ```

### 4. **Fine-Tuning or Using Pre-Trained Models**  
- Use **Fine_tune_img_seg.ipynb** to fine-tune SAM2 on custom datasets.  
- Use **use_trained_model.ipynb** to apply pre-trained models for segmentation.  

---

## Key Features  

- **Real-Time Camera Integration**: Access the Intel RealSense camera for live image and depth sensing.  
- **SAM2 Integration**: Segment captured images and videos using the SAM2 model.  
- **Process Synchronization**: Ensure seamless coordination between image capture and segmentation using a parent process.  
- **Extensibility**: Includes additional scripts for YOLO-based segmentation for comparison.  

---

## Dependencies  

- Python Libraries:  
  - PyRealSense2  
  - OpenCV  
  - NumPy  
  - PyTorch  
  - Matplotlib  
- Intel RealSense SDK  
- SAM2 Model  

Install dependencies using:  
```bash
pip install -r requirements.txt
```

---

## Example Workflow  

1. **Capture Images**:  
   Run `camera_access.py` to capture input images.  

2. **Segment Images**:  
   Use `segmentation.py` to apply SAM2 for segmentation.  

3. **Synchronize Operations**:  
   Run `parent.py` to coordinate camera input and SAM2 processing.  

4. **Visualize Results**:  
   Display segmented masks and outputs using `video_predictor_example.ipynb`.  

---

## License  

This project is licensed under the MIT License.  

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

**Email**: iit2022219@iiita.ac.in / iit2022218@iiita.ac.in 
**GitHub**: [Your GitHub Profile](https://github.com/shorty-huddybuddy)  
```

This `README.md` file provides all the necessary details for users and contributors to understand and use the project effectively. Let me know if you'd like to add more sections!
