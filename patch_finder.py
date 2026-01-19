import os
import json
import argparse
import openslide
import numpy as np
import onnxruntime as ort
import cv2
from openslide import OpenSlideError
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

###################################################################
# 1. SVD & Bounding Box Logic (Pure Numpy/CPU)                    #
###################################################################

def get_tissue_bbox_svd(thumbnail_rgb, n_vectors=5, threshold=215, min_threshold=25):
    """
    1. Converts thumbnail to grayscale.
    2. Performs SVD and reconstructs using only the top `n_vectors`.
    3. Thresholds to find approximate tissue location.
       - Uses a range [min_threshold, threshold] to exclude black frames.
    4. Computes a bounding box (x_min, y_min, x_max, y_max).
    Returns: bbox, cropped_thumb, debug_images(tuple)
    """
    # Ensure input is numpy array
    thumbnail = np.array(thumbnail_rgb)
    
    # 1. Convert to grayscale (Luminosity method)
    # Weights: 0.299 R + 0.587 G + 0.114 B
    gray = np.dot(thumbnail[..., :3], [0.299, 0.587, 0.114])

    # 2. Perform SVD (CPU based)
    # U: (H, K), s: (K,), Vt: (K, W)
    # We use full_matrices=False for efficiency
    U, s, Vt = np.linalg.svd(gray, full_matrices=False)

    # 3. Reconstruct with top N vectors
    k = n_vectors
    # Reshape s to diagonal matrix for multiplication or broadcast
    reconstructed = np.dot(U[:, :k] * s[:k], Vt[:k, :])

    # 4. Threshold to create a mask 
    # Background is usually bright (> threshold)
    # Black Frame/Artifacts are very dark (< min_threshold)
    # Tissue is in between
    svd_mask = (reconstructed < threshold) & (reconstructed > min_threshold)

    # 5. Find Bounding Box
    rows = np.any(svd_mask, axis=1)
    cols = np.any(svd_mask, axis=0)

    # Handle case where no tissue is detected (return full image)
    if not np.any(rows) or not np.any(cols):
        print("  Warning: SVD found no tissue within valid intensity range. Using full thumbnail.")
        return (0, 0, thumbnail.shape[1], thumbnail.shape[0]), thumbnail, (reconstructed, svd_mask)

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # Define BBox (x_min, y_min, x_max, y_max)
    bbox = (int(x_min), int(y_min), int(x_max), int(y_max))

    # Crop the original thumbnail
    # y_max+1 to include the last pixel
    cropped_thumb = thumbnail[y_min : y_max+1, x_min : x_max+1, :]

    return bbox, cropped_thumb, (reconstructed, svd_mask)

def visualize_svd_steps(original, reconstructed, mask, bbox, output_path):
    """
    Saves a figure showing SVD reconstruction steps.
    """
    x_min, y_min, x_max, y_max = bbox
    box_w = x_max - x_min
    box_h = y_max - y_min

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: SVD Reconstruction
    axes[0].imshow(reconstructed, cmap='gray')
    axes[0].set_title("1. SVD Reconstruction")
    axes[0].axis('off')

    # Plot 2: Thresholded Mask
    axes[1].imshow(mask, cmap='gray')
    rect = mpatches.Rectangle((x_min, y_min), box_w, box_h, linewidth=2, edgecolor='red', facecolor='none')
    axes[1].add_patch(rect)
    axes[1].set_title("2. SVD Mask (Range Filtered) + BBox")
    axes[1].axis('off')

    # Plot 3: Original Thumbnail
    axes[2].imshow(original)
    rect2 = mpatches.Rectangle((x_min, y_min), box_w, box_h, linewidth=2, edgecolor='red', facecolor='none')
    axes[2].add_patch(rect2)
    axes[2].set_title("3. Original with BBox")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close(fig)
    print(f"  Saved SVD visualization to: {output_path}")

###################################################################
# 2. ONNX Inference                                               #
###################################################################

def predict_mask_on_thumbnail(thumbnail, onnx_sess, model_input_size=500):
    """
    Args:
        thumbnail: Numpy array (H, W, 3) in RGB, values 0-255.
        onnx_sess: The loaded ONNX Runtime InferenceSession.
    """
    # 1. Resize to (500, 500)
    img_resized = cv2.resize(thumbnail, (model_input_size, model_input_size), interpolation=cv2.INTER_LINEAR)
    
    # 2. Cast to Float32
    img_float = img_resized.astype(np.float32)

    # 3. Manual ResNet Preprocessing
    # Convert RGB -> BGR
    img_bgr = img_float[..., ::-1] 

    # Subtract ImageNet Mean (BGR order)
    # CRITICAL FIX: Ensure mean is float32 so the result stays float32
    mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)
    
    img_preprocessed = img_bgr - mean

    # 4. Expand Dimensions (Batch Size) -> (1, 500, 500, 3)
    inp = np.expand_dims(img_preprocessed, axis=0)

    # EXTRA SAFETY: Ensure final input is strictly float32
    inp = inp.astype(np.float32)

    # 5. Run ONNX Inference
    input_name = onnx_sess.get_inputs()[0].name
    
    # Run inference
    pred = onnx_sess.run(None, {input_name: inp})[0] 
    
    # Squeeze batch dimension
    pred = np.squeeze(pred)

    # 6. Resize Mask back to original thumbnail dimensions
    orig_h, orig_w = thumbnail.shape[:2]
    pred_mask = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # 7. Threshold
    # Using 0.5 as per the integrated snippet
    return (pred_mask > 0.5).astype(np.uint8)


###################################################################
# 3. Main Patch Finder Framework                                  #
###################################################################

def run_patch_finder_framework(slide_path, onnx_model_path, target_level, patch_size_l, tissue_thresh=0.3, output_dir='./output'):
    print(f"--- Running Patch Finder on: {os.path.basename(slide_path)} ---")
    
    try:
        slide = openslide.OpenSlide(slide_path)
        
        # Dimensions
        th_level = slide.level_count - 1
        w_th, h_th = slide.level_dimensions[th_level]
        w_l, h_l = slide.level_dimensions[target_level]
        w_0, h_0 = slide.level_dimensions[0]

        # 1. Read Full Thumbnail
        thumb_image_pil = slide.read_region((0, 0), th_level, (w_th, h_th)).convert("RGB")
        thumb_np = np.array(thumb_image_pil)
        
        slide_output_dir = os.path.join(output_dir, os.path.splitext(os.path.basename(slide_path))[0])
        os.makedirs(slide_output_dir, exist_ok=True)

        # -------------------------------------------------------------
        # 1. Calculate Patch Size
        # -------------------------------------------------------------
        ps_th_w = int(patch_size_l * (w_th / w_l))
        ps_th_h = int(patch_size_l * (h_th / h_l))

        if ps_th_w == 0 or ps_th_h == 0:
            print(f"  Error: Calculated thumbnail patch size is zero. Skipping.")
            slide.close()
            return

        # -------------------------------------------------------------
        # 2. SVD Stage: Get "Tight" BBox
        # -------------------------------------------------------------
        svd_bbox_tight, _, debug_imgs = get_tissue_bbox_svd(
            thumb_np, n_vectors=15, threshold=215, min_threshold=25
        )
        t_x1, t_y1, t_x2, t_y2 = svd_bbox_tight
        
        # -------------------------------------------------------------
        # 3. Create "Inference Crop" with Padding
        # -------------------------------------------------------------
        # We pad by a FULL patch size to be safe, ensuring we have pixels 
        # for patches that stick out significantly.
        pad_x = ps_th_w
        pad_y = ps_th_h

        # Clamp to image boundaries
        p_x1 = max(0, t_x1 - pad_x)
        p_y1 = max(0, t_y1 - pad_y)
        p_x2 = min(w_th, t_x2 + pad_x)
        p_y2 = min(h_th, t_y2 + pad_y)

        cropped_thumb_padded = thumb_np[p_y1 : p_y2, p_x1 : p_x2, :]
        print(f"  SVD Box: {svd_bbox_tight} -> Inference Crop: {(p_x1, p_y1, p_x2, p_y2)}")

        # -------------------------------------------------------------
        # 4. ONNX Inference
        # -------------------------------------------------------------
        print(f"  Loading ONNX model from: {onnx_model_path}")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        sess = ort.InferenceSession(onnx_model_path, providers=providers)

        # Mask corresponds to the PADDED area
        crop_mask = predict_mask_on_thumbnail(cropped_thumb_padded, sess) 
        
        if np.sum(crop_mask) == 0:
            print("  Warning: Model produced an empty mask.")
            slide.close()
            return

        # -------------------------------------------------------------
        # 5. Generate Patches (Center-Logic Corrected)
        # -------------------------------------------------------------
        patch_positions = []
        valid_patch_coords_th = [] 

        # We want our grid to align with the ORIGINAL SVD Box (t_x1, t_y1)
        # to prevent "phase shifting" the patches.
        
        # Start the loop relative to the Padded Crop (0,0), 
        # but align the first step to match t_x1.
        
        # Offset of the Tight Box relative to the Padded Crop
        offset_x = t_x1 - p_x1
        offset_y = t_y1 - p_y1

        # Determine start/end for the grid relative to the Tight Box
        # We start slightly before 0 (negative relative to tight box) if the padding allows
        # But per your logic: Center must be inside.
        # If x = t_x1 - ps, Center = t_x1 - ps/2 (Outside). 
        # So generally, we can start exactly at offset_x.
        
        crop_h, crop_w = crop_mask.shape
        
        # Grid loop: We iterate coordinate 'x' representing the top-left of a patch
        # relative to the PADDED crop.
        # We start at 'offset_x' (which aligns with t_x1) and go up to the crop limit.
        for x_crop in range(offset_x, crop_w - ps_th_w + 1, ps_th_w):
            for y_crop in range(offset_y, crop_h - ps_th_h + 1, ps_th_h):
                
                # --- CHECK 1: CENTER LOGIC ---
                # Calculate the center of this patch in Global Coords
                x_global = p_x1 + x_crop
                y_global = p_y1 + y_crop
                
                center_x = x_global + ps_th_w / 2
                center_y = y_global + ps_th_h / 2
                
                # Is the center inside the TIGHT SVD box?
                center_in_x = (t_x1 <= center_x <= t_x2)
                center_in_y = (t_y1 <= center_y <= t_y2)
                
                if center_in_x and center_in_y:
                    
                    # --- CHECK 2: TISSUE THRESHOLD ---
                    mask_patch = crop_mask[y_crop : y_crop + ps_th_h, x_crop : x_crop + ps_th_w]
                    tissue_ratio = np.sum(mask_patch) / (ps_th_w * ps_th_h)
                    
                    # NOTE: Edge patches have lots of empty space. 
                    # If tissue_ratio is low but center is inside, you might want to keep it.
                    # Current logic: Strict threshold.
                    if tissue_ratio > tissue_thresh:
                        
                        valid_patch_coords_th.append((x_global, y_global))
                        
                        x_0 = int(x_global * (w_0 / w_th))
                        y_0 = int(y_global * (h_0 / h_th))
                        patch_positions.append({'x': int(x_0), 'y': int(y_0)})

        print(f"  Found {len(patch_positions)} valid patches.")

        # -------------------------------------------------------------
        # 6. Visualization
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(thumb_image_pil)
        
        # Draw SVD Tight Box (Blue)
        svd_rect = mpatches.Rectangle((t_x1, t_y1), t_x2 - t_x1, t_y2 - t_y1,
            fill=False, edgecolor='blue', linewidth=2, label="SVD Limit (Center must be in here)")
        ax.add_patch(svd_rect)

        # Draw Valid Patches (Red)
        for (x_th, y_th) in valid_patch_coords_th:
            rect = mpatches.Rectangle((x_th, y_th), ps_th_w, ps_th_h, fill=False, edgecolor='red', linewidth=1)
            ax.add_patch(rect)

        ax.set_title(f"Patches (Center-in-Box Logic) | Thresh: {tissue_thresh}")
        ax.legend()
        ax.axis('off')

        plot_filename = os.path.join(slide_output_dir, "final_patches_viz.png")
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Save JSON
        json_filename = os.path.join(slide_output_dir, "patch_positions.json")
        with open(json_filename, 'w') as f:
            json.dump(patch_positions, f, indent=4)

        slide.close()
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Patch Finder Framework (SVD + ONNX)")
    parser.add_argument('--slides_dir', type=str, required=True, help='Directory containing .svs slide files')
    parser.add_argument('--onnx_model', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--target_level', type=int, required=True, help='Level to extract patches from')
    parser.add_argument('--patch_size', type=int, required=True, help='Desired patch size at the target level')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save outputs')

    args = parser.parse_args()

    if not os.path.isdir(args.slides_dir):
        print(f"Error: Directory not found at path: {args.slides_dir}")
        return
        
    if not os.path.exists(args.onnx_model):
        print(f"Error: ONNX Model not found at path: {args.onnx_model}")
        return

    found_svs_files = False
    for root, _, files in os.walk(args.slides_dir):
        for filename in files:
            if filename.endswith('.mrxs'):
                found_svs_files = True
                file_path = os.path.join(root, filename)
                run_patch_finder_framework(
                    file_path, 
                    args.onnx_model, 
                    args.target_level, 
                    args.patch_size, 
                    output_dir=args.output_dir
                )
        
    if not found_svs_files:
        print("No .mrxs files were found in the specified directory.")

if __name__ == "__main__":
    main()
