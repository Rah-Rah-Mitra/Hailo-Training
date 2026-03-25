# main.py

import os
import onnx
from onnxsim import simplify
import numpy as np
from hailo_sdk_client import ClientRunner
from PIL import Image

def parse_onnx(onnx_path: str, calib_folder: str, net_name: str, hw_arch: str, target_size: tuple = (640, 640)):
    # -------------------------------
    # Step 1. Simplify the ONNX model
    # -------------------------------

    # Load the ONNX model
    model = onnx.load(onnx_path)

    # Simplify the model
    model_simp, check = simplify(model)
    if not check:
        raise RuntimeError("Simplified ONNX model validation failed.")
        
    # Save the simplified model to a new file
    simplified_onnx_path = os.path.splitext(onnx_path)[0] + "_simplified.onnx"
    onnx.save(model_simp, simplified_onnx_path)
    print(f"Simplified ONNX model saved to: {simplified_onnx_path}")

    # -----------------------------------------------------
    # Step 2. Translate the simplified ONNX model to Hailo format
    # -----------------------------------------------------

    # Create a ClientRunner instance.
    # The hw_arch parameter should match your target Hailo device (e.g., "hailo8")
    runner = ClientRunner(hw_arch=hw_arch, har=None)

    # Translate the ONNX model. Optionally, you can supply start and end node names if needed.
    hn, params = runner.translate_onnx_model(simplified_onnx_path, net_name)
    print("Model translation to Hailo format completed.")

    # -----------------------------------------------------
    # Step 3. Quantize the model using a calibration dataset
    # -----------------------------------------------------
    # For quantization, you need a calibration dataset.
    # Adjust the shape (batch, height, width, channels) as required by your model.
    # For many YOLO models the expected input is 640x640 with 3 channels.
    calib_dataset = load_calibration_dataset(calib_folder, target_size)
    print("Calibration dataset created.")

    # Run optimization (quantization). This process uses the calibration dataset to
    # convert floating-point parameters into their quantized (integer) counterparts.
    runner.optimize(calib_dataset)
    print("Model quantization complete.")

    # -----------------------------------------------------
    # Step 4. Save the quantized model as an HAR file
    # -----------------------------------------------------
    har_file = f"{net_name}_quantized_model.har"
    runner.save_har(har_file)
    print(f"HAR file saved to: {har_file}")

def load_calibration_dataset(calib_folder, target_size=(640, 640)):
    """
    Loads and preprocesses images from the specified folder.
    Args:
        calib_folder (str): Path to the folder with calibration images.
        target_size (tuple): Desired image size as (width, height).
    Returns:
        np.ndarray: A numpy array of shape (num_images, height, width, 3) in float32.
    """
    image_files = [
        os.path.join(calib_folder, f)
        for f in os.listdir(calib_folder)
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ]
    if not image_files:
        raise ValueError("No calibration images found in the folder.")
    
    images = []
    for img_file in sorted(image_files):
        img = Image.open(img_file).convert("RGB")
        # Resize using bilinear interpolation
        img = img.resize(target_size, Image.BILINEAR)
        img_np = np.array(img).astype(np.float32)
        images.append(img_np)
    
    calib_dataset = np.stack(images, axis=0)
    print(f"Loaded {calib_dataset.shape[0]} calibration images of size {target_size}.")
    return calib_dataset

def main():
    onnx_path = "./models/best.onnx"
    calib_folder = "./calibration_images"
    net_name = "yolov8n"
    hw_arch = "hailo8l"

    parse_onnx(onnx_path, calib_folder, net_name, hw_arch)

if __name__ == "__main__":
    main()