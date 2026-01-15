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

def get_tissue_bbox_svd(thumbnail_rgb, n_vectors=5, threshold=215):
    """
    1. Converts thumbnail to grayscale.
    2. Performs SVD and reconstructs using only the top `n_vectors`.
    3. Thresholds to find approximate tissue location.
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
    # (Background is usually bright > threshold, Tissue is dark)
    svd_mask = (reconstructed < threshold)

    # 5. Find Bounding Box
    rows = np.any(svd_mask, axis=1)
    cols = np.any(svd_mask, axis=0)

    # Handle case where no tissue is detected (return full image)
    if not np.any(rows) or not np.any(cols):
        print("  Warning: SVD found no tissue. Using full thumbnail.")
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
    axes[0].set_title("1. SVD Reconstruction (Top 5 Vecs)")
    axes[0].axis('off')

    # Plot 2: Thresholded Mask
    axes[1].imshow(mask, cmap='gray')
    rect = mpatches.Rectangle((x_min, y_min), box_w, box_h, linewidth=2, edgecolor='red', facecolor='none')
    axes[1].add_patch(rect)
    axes[1].set_title("2. SVD Mask + Bounding Box")
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

def predict_mask_on_thumbnail(thumbnail, onnx_model, model_input_size=500, threshold=0.6):
    """
    Runs ONNX inference on the provided thumbnail (or crop).
    """
    if not hasattr(predict_mask_on_thumbnail, "sess"):
        print(f"  Initializing ONNX session for: {onnx_model}")
        # Prioritize CUDA if available, else CPU
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        predict_mask_on_thumbnail.sess = ort.InferenceSession(onnx_model, providers=providers)
    
    # Resize to model input
    inp = cv2.resize(thumbnail, (model_input_size, model_input_size)).astype(np.float32) / 255.0
    inp = np.expand_dims(inp, 0)
    
    # Run inference
    # Note: Adjust output indexing [0][0, ..., 0] based on your specific ONNX model output shape
    pred = predict_mask_on_thumbnail.sess.run(None, {"input": inp})[0][0, ..., 0]
    
    # Resize output mask back to input image size
    pred = cv2.resize(pred, (thumbnail.shape[1], thumbnail.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    return (pred > threshold).astype(np.uint8)


###################################################################
# 3. Main Patch Finder Framework                                  #
###################################################################

def run_patch_finder_framework(slide_path, onnx_model_path, target_level, patch_size_l, tissue_thresh=0.5, output_dir='./output'):
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
        
        # Create output directory for this slide
        slide_output_dir = os.path.join(output_dir, os.path.splitext(os.path.basename(slide_path))[0])
        os.makedirs(slide_output_dir, exist_ok=True)

        # -------------------------------------------------------------
        # 2. SVD Stage: Get BBox and Crop
        # -------------------------------------------------------------
        svd_bbox, cropped_thumb, debug_imgs = get_tissue_bbox_svd(thumb_np, n_vectors=15)
        
        # Visualize SVD
        svd_viz_path = os.path.join(slide_output_dir, "svd_steps.png")
        visualize_svd_steps(thumb_np, debug_imgs[0], debug_imgs[1], svd_bbox, svd_viz_path)

        # -------------------------------------------------------------
        # 3. ONNX Inference (Run ONLY on the Crop)
        # -------------------------------------------------------------
        # crop_mask corresponds to the cropped_thumb area
        crop_mask = predict_mask_on_thumbnail(cropped_thumb, onnx_model_path) 
        
        if np.sum(crop_mask) == 0:
            print("  Warning: Model produced an empty mask on the crop. No patches will be found.")
            slide.close()
            return

        # -------------------------------------------------------------
        # 4. Generate Patches (Mapping back to Global coords)
        # -------------------------------------------------------------
        
        # Calculate patch size in thumbnail scale
        # Ratio: how many target_level pixels equal 1 thumbnail pixel?
        # Actually we need: how big is 'patch_size_l' in thumbnail pixels?
        # w_th / w_l is the scaling factor < 1
        ps_th_w = int(patch_size_l * (w_th / w_l))
        ps_th_h = int(patch_size_l * (h_th / h_l))
        
        if ps_th_w == 0 or ps_th_h == 0:
            print(f"  Error: Calculated thumbnail patch size is zero. ({ps_th_w}, {ps_th_h}). Skipping.")
            slide.close()
            return

        patch_positions = []
        valid_patch_coords_th = [] # Global thumbnail coords for plotting

        # Unpack SVD bbox offset
        x_min_offset, y_min_offset = svd_bbox[0], svd_bbox[1]
        
        # Iterate over the CROP dimensions
        crop_h, crop_w = crop_mask.shape
        
        for x_crop in range(0, crop_w - ps_th_w, ps_th_w):
            for y_crop in range(0, crop_h - ps_th_h, ps_th_h):
                
                # Check tissue ratio on the crop mask
                mask_patch = crop_mask[y_crop : y_crop + ps_th_h, x_crop : x_crop + ps_th_w]
                tissue_ratio = np.sum(mask_patch) / (ps_th_w * ps_th_h)
                
                if tissue_ratio > tissue_thresh:
                    # 1. Calculate Global Thumbnail Coords
                    x_global_th = x_crop + x_min_offset
                    y_global_th = y_crop + y_min_offset
                    
                    valid_patch_coords_th.append((x_global_th, y_global_th))
                    
                    # 2. Calculate Level 0 Coords (for JSON output)
                    # We map from global thumbnail to level 0
                    x_0 = int(x_global_th * (w_0 / w_th))
                    y_0 = int(y_global_th * (h_0 / h_th))
                    
                    # Store as standard python types
                    patch_positions.append({'x': int(x_0), 'y': int(y_0)})

        print(f"  Found {len(patch_positions)} valid patches.")
        if not patch_positions:
            slide.close()
            return

        # -------------------------------------------------------------
        # 5. Final Visualization (Full Thumbnail + SVD Box + Patches)
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(thumb_image_pil)
        
        # Draw SVD Box
        svd_rect = mpatches.Rectangle(
            (svd_bbox[0], svd_bbox[1]), 
            svd_bbox[2] - svd_bbox[0], 
            svd_bbox[3] - svd_bbox[1],
            fill=False, edgecolor='blue', linewidth=2, label="SVD Crop"
        )
        ax.add_patch(svd_rect)

        # Draw Patch Boxes
        for (x_th, y_th) in valid_patch_coords_th:
            rect = mpatches.Rectangle((x_th, y_th), ps_th_w, ps_th_h, fill=False, edgecolor='red', linewidth=1)
            ax.add_patch(rect)

        ax.set_title("Thumbnail: SVD Crop (Blue) & Valid Patches (Red)")
        ax.legend()
        ax.axis('off')

        plot_filename = os.path.join(slide_output_dir, "final_patches_viz.png")
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"  Saved grid plot to: {plot_filename}")
        plt.close(fig)

        # Save JSON
        json_filename = os.path.join(slide_output_dir, "patch_positions.json")
        with open(json_filename, 'w') as f:
            json.dump(patch_positions, f, indent=4)
        print(f"  Saved patch positions to: {json_filename}")

        slide.close()
        
    except OpenSlideError as e:
        print(f"  Error opening or reading slide: {e}")
    except Exception as e:
        print(f"  An unexpected error occurred: {e}")
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
            if filename.endswith('.svs'):
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
        print("No .svs files were found in the specified directory.")

if __name__ == "__main__":
    main()
