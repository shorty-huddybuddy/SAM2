print('Loading SAM 2 Model...')
auto_seg = False

import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"using device: {device}")

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
    )

# Initialize SAM 2 model
from sam2.build_sam import build_sam2
if(auto_seg) : 

    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(sam2)
else :

    from sam2.sam2_image_predictor import SAM2ImagePredictor
    sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

print('Model loaded...')

# Helper function for segmentation
def run_auto_segmentation(image_path, output_dir, mask_generator):
    """Run segmentation on the given image."""
    image = Image.open(image_path)
    image = np.array(image.convert("RGB"))
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "original_image.png"), bbox_inches='tight', pad_inches=0)
    plt.close()

    # Generate masks
    masks = mask_generator.generate(image)

    print(f"Generated {len(masks)} masks.")

    # Visualize masks
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_auto_anns(masks)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "auto_mask_gen.png"), bbox_inches='tight', pad_inches=0)
    plt.close()

def show_auto_anns(anns, borders=True):
    """Visualize annotations."""
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    ax.imshow(img)

def show_lab_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_lab_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_lab_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_lab_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_lab_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_lab_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_lab_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        output_path = os.path.join(output_dir, f"mask_{i+1}.png")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()  # Close the figure to free up memory

# Directory where images will be saved
if(auto_seg) :
    output_dir = r"C:\Users\sinha\segment-anything-2\notebooks\auto_seg_output_images"
else : 
    output_dir = r"C:\Users\sinha\segment-anything-2\notebooks\output_images"
os.makedirs(output_dir, exist_ok=True)

import subprocess

# Paths to Python executables
camera_script = r"C:\Users\sinha\Desktop\python\camera_access.py"
python_39 = r'C:/Users/sinha/.conda/envs/realsense_env/python.exe'

while True:
    try:
        # Step 1: Capture image using camera_access.py
        print("Capturing image...")
        subprocess.run([python_39, camera_script], check=True)

        # Step 2: Ensure the image exists before segmentation
        time.sleep(1)  # Allow some time for the image to be saved (adjust if necessary)
        captured_images = [f for f in os.listdir(output_dir) if f.endswith(".jpg") or f.endswith(".png")]

        if not captured_images:
            print("No new image found. Retrying...")
            continue

        latest_image = os.path.join(output_dir, max(captured_images, key=lambda f: os.path.getctime(os.path.join(output_dir, f))))
        print(f"Processing image: {latest_image}")

        # Step 3: Perform segmentation
        if(auto_seg) : 
            run_auto_segmentation(latest_image, "auto_seg_output_images", mask_generator)
        else :
            latest_image = Image.open(r"C:\Users\sinha\segment-anything-2\notebooks\output_images\captured_photo.jpg")
            latest_image = np.array(latest_image.convert("RGB"))
            predictor.set_image(latest_image)
            input_point = np.array([[375, 200]])
            input_label = np.array([1])
            plt.figure(figsize=(10, 10))
            plt.imshow(latest_image)
            show_lab_points(input_point, input_label, plt.gca())
            plt.axis('on')
            plt.savefig(os.path.join(output_dir, "labelled_image.png"), bbox_inches='tight', pad_inches=0)
            plt.close()
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            show_lab_masks(latest_image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)

        print("Segmentation completed successfully.")

    except subprocess.CalledProcessError as e:
        print(f"Error during execution: {e}")
    except KeyboardInterrupt:
        print("Process interrupted by user.")
        break
    except Exception as e:
        print(f"Unexpected error: {e}")
