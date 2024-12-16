import pyrealsense2 as rs
import numpy as np
import cv2
from datetime import datetime

def take_realsense_photo(output_dir1, output_dir2):
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable the color stream
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    try:
        # Start the pipeline
        pipeline.start(config)
        print("Camera started, waiting for a frame...")

        # Allow the camera some time to auto-adjust
        for _ in range(30):  # Skip first 30 frames for auto-exposure to settle
            pipeline.wait_for_frames()

        # Capture the next frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            print("Error: Could not capture color frame.")
            return

        # Convert the image to a NumPy array
        color_image = np.asanyarray(color_frame.get_data())

        # Apply auto white balance (if needed, manually tweak if this doesn't help)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # Increase brightness and contrast
        alpha = 1.8  # Contrast control (1.0 - 3.0)
        beta = 40    # Brightness control (0 - 100)
        enhanced_image = cv2.convertScaleAbs(color_image, alpha=alpha, beta=beta)

        # Convert back to BGR for saving as an image
        enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR)

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Create full file paths with timestamps
        output_filename1 = f"{output_dir1}/captured_photo_{timestamp}.jpg"
        output_filename2 = f"{output_dir2}/captured_photo_{timestamp}.jpg"

        # Save the enhanced image to the specified file paths
        cv2.imwrite(output_filename1, enhanced_image)
        print(f"Photo saved as {output_filename1}")

        cv2.imwrite(output_filename2, enhanced_image)
        print(f"Photo saved as {output_filename2}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Stop the pipeline
        pipeline.stop()
        print("Camera stopped.")

# Call the function to take a photo and save it to two locations
take_realsense_photo(
    r"C:\Users\sinha\segment-anything-2\notebooks\images\LAB",
    r"C:\Users\sinha\OneDrive\Pictures\Saved Pictures"
)
