import base64
import io
from typing import Tuple

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms


def _to_square_and_resize(pil_gray: Image.Image, out_size: int = 28, threshold: int = 128) -> Image.Image:
    """
    Center the drawn letter using center of mass (centroid) instead of just bounding box.
    This makes recognition more robust regardless of where the user draws on canvas.
    
    Process:
    - Compute centroid of white pixels
    - Center the letter at (out_size/2, out_size/2) based on centroid
    - Crop to fit content while maintaining centering
    - Scale to fit in out_size x out_size
    """
    # Convert to numpy
    arr = np.array(pil_gray, dtype=np.uint8)

    # Apply binary threshold for detection
    arr_binary = np.where(arr > threshold, 255, 0).astype(np.uint8)
    
    # Find all white pixels
    ys, xs = np.where(arr_binary > 0)
    if ys.size == 0 or xs.size == 0:
        # Nothing drawn: return empty 28x28 black image
        return Image.new("L", (out_size, out_size), color=0)

    # Compute centroid (center of mass)
    centroid_y = int(ys.mean())
    centroid_x = int(xs.mean())
    
    # Compute bounding box
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    
    # Determine content size
    content_h = y_max - y_min + 1
    content_w = x_max - x_min + 1
    
    # Make it square by taking max dimension, with some padding (20% margin)
    side = int(max(content_h, content_w) * 1.2)
    
    # Calculate where to place content so centroid is at center of output
    target_center = side // 2
    
    # Calculate offset to center the content by its centroid
    offset_y = target_center - (centroid_y - y_min)
    offset_x = target_center - (centroid_x - x_min)
    
    # Create square canvas
    square_arr = np.zeros((side, side), dtype=np.uint8)
    
    # Paste content centered by centroid
    # Make sure we don't go out of bounds
    src_y_start = max(0, y_min)
    src_y_end = min(arr.shape[0], y_max + 1)
    src_x_start = max(0, x_min)
    src_x_end = min(arr.shape[1], x_max + 1)
    
    dst_y_start = max(0, offset_y)
    dst_y_end = min(side, offset_y + content_h)
    dst_x_start = max(0, offset_x)
    dst_x_end = min(side, offset_x + content_w)
    
    # Adjust source region if destination is clipped - must maintain matching dimensions
    if offset_y < 0:
        src_y_start -= offset_y
    if offset_x < 0:
        src_x_start -= offset_x
    if dst_y_end > side:
        src_y_end -= (dst_y_end - side)
    if dst_x_end > side:
        src_x_end -= (dst_x_end - side)
    
    # Ensure source and destination have exactly matching dimensions
    actual_src_h = src_y_end - src_y_start
    actual_src_w = src_x_end - src_x_start
    actual_dst_h = dst_y_end - dst_y_start
    actual_dst_w = dst_x_end - dst_x_start
    
    # Adjust to minimum of both to ensure they match
    final_h = min(actual_src_h, actual_dst_h)
    final_w = min(actual_src_w, actual_dst_w)
    
    src_y_end = src_y_start + final_h
    src_x_end = src_x_start + final_w
    dst_y_end = dst_y_start + final_h
    dst_x_end = dst_x_start + final_w
    
    # Copy the region
    square_arr[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
        arr[src_y_start:src_y_end, src_x_start:src_x_end]

    # Convert to PIL and resize to output size
    sq = Image.fromarray(square_arr)
    resample = getattr(getattr(Image, "Resampling", Image), "BILINEAR", Image.BILINEAR)
    return sq.resize((out_size, out_size), resample)


def _preprocess_canvas_png(b64_png: str, debug: bool = False) -> Tuple[torch.Tensor, Image.Image, dict]:
    """
    Convert base64 PNG from HTML canvas to a tensor [1,1,28,28] using
    the same assumptions as the working Streamlit app:
    - Canvas is black strokes on white background
    - Invert to white-on-black for EMNIST/MNIST-style models
    - Center, pad to square, resize; normalize with MNIST stats
    Returns (tensor, pil_image_28x28, debug_dict)
    """
    debug_info = {}
    
    # Remove data URL prefix if exists
    if "," in b64_png:
        b64_png = b64_png.split(",", 1)[1]

    # Decode base64 → bytes → PIL Image (ensure RGB first)
    img_bytes = base64.b64decode(b64_png)
    img = Image.open(io.BytesIO(img_bytes))
    
    debug_info['original_mode'] = img.mode
    debug_info['original_size'] = img.size
    
    # Convert RGBA → RGB if needed
    if img.mode == 'RGBA':
        # Create white background
        bg = Image.new('RGB', img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])  # Use alpha channel as mask
        img = bg
    elif img.mode != 'RGB':
        img = img.convert('RGB')

    # To grayscale
    gray = img.convert("L")
    
    # Check if we need to invert
    gray_arr = np.array(gray)
    mean_val = gray_arr.mean()
    debug_info['gray_mean'] = float(mean_val)
    
    # If mean > 127, background is bright (white), strokes are dark (black) → invert
    # If mean < 127, background is dark (black), strokes are bright (white) → no invert
    if mean_val > 127:
        gray_inv = ImageOps.invert(gray)
        debug_info['inverted'] = True
    else:
        gray_inv = gray
        debug_info['inverted'] = False

    # Center/crop/pad and resize to 28x28 (lower threshold to catch lighter strokes)
    img28 = _to_square_and_resize(gray_inv, out_size=28, threshold=100)

    # To tensor with MNIST normalization
    tfm = transforms.Compose([
        transforms.ToTensor(),                # [0,1]
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    x = tfm(img28).unsqueeze(0)  # [1,1,28,28]
    return x, img28, debug_info


def tensor_from_base64(b64_png: str) -> torch.Tensor:
    """Public entry: return only the model tensor [1,1,28,28]."""
    x, _, _ = _preprocess_canvas_png(b64_png)
    return x