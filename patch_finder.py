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

def predict_mask_on_thumbnail(thumbnail, onnx_model, model_input_size=500, threshold=0.6):
    if not hasattr(predict_mask_on_thumbnail, "sess"):
        print(f"  Initializing ONNX session for: {onnx_model}")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        predict_mask_on_thumbnail.sess = ort.InferenceSession(onnx_model, providers=providers)
    
    inp = cv2.resize(thumbnail, (model_input_size, model_input_size)).astype(np.float32) / 255.0
    inp = np.expand_dims(inp, 0)
    pred = predict_mask_on_thumbnail.sess.run(None, {"input": inp})[0][0, ..., 0]
    pred = cv2.resize(pred, (thumbnail.shape[1], thumbnail.shape[0]), interpolation=cv2.INTER_LINEAR)
    return (pred > threshold).astype(np.uint8)

def run_patch_finder_framework(slide_path, onnx_model_path, target_level, patch_size_l, tissue_thresh=0.5, output_dir='./output'):
    print(f"--- Running Patch Finder on: {os.path.basename(slide_path)} ---")
    
    try:
        slide = openslide.OpenSlide(slide_path)
        
        th_level = slide.level_count - 1
        w_th, h_th = slide.level_dimensions[th_level]
        w_l, h_l = slide.level_dimensions[target_level]
        w_0, h_0 = slide.level_dimensions[0]

        thumb_image_pil = slide.read_region((0, 0), th_level, (w_th, h_th)).convert("RGB")
        thumb_np = np.array(thumb_image_pil)
        
        tissue_mask = predict_mask_on_thumbnail(thumb_np, onnx_model_path) 
        
        if np.sum(tissue_mask) == 0:
            print("  Warning: Model produced an empty mask. No patches will be found.")
            slide.close()
            return

        ps_th_w = int(patch_size_l * (w_th / w_l))
        ps_th_h = int(patch_size_l * (h_th / h_l))
        
        if ps_th_w == 0 or ps_th_h == 0:
            print(f"  Error: Calculated thumbnail patch size is zero. ({ps_th_w}, {ps_th_h}). Skipping.")
            slide.close()
            return

        valid_patch_coords_l0 = [] 
        valid_patch_coords_th = [] 

        patch_positions = []

        for x_th in range(0, w_th - ps_th_w, ps_th_w):
            for y_th in range(0, h_th - ps_th_h, ps_th_h):
                mask_patch = tissue_mask[y_th : y_th + ps_th_h, x_th : x_th + ps_th_w]
                tissue_ratio = np.sum(mask_patch) / (ps_th_w * ps_th_h)
                
                if tissue_ratio > tissue_thresh:
                    valid_patch_coords_th.append((x_th, y_th))
                    x_0 = int(x_th * (w_0 / w_th))
                    y_0 = int(y_th * (h_0 / h_th))
                    valid_patch_coords_l0.append((x_0, y_0))
                    patch_positions.append({'x': x_0, 'y': y_0})

        print(f"  Found {len(valid_patch_coords_l0)} valid patches.")
        if not valid_patch_coords_l0:
            slide.close()
            return

        os.makedirs(output_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(thumb_image_pil)
        ax.set_title("Thumbnail with Valid Patch Grid")
        ax.axis('off')

        for (x_th, y_th) in valid_patch_coords_th:
            rect = mpatches.Rectangle((x_th, y_th), ps_th_w, ps_th_h, fill=False, edgecolor='red', linewidth=1)
            ax.add_patch(rect)

        plot_filename = os.path.join(output_dir, f"patches_on_thumbnail_{os.path.basename(slide_path)}.png")
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"  Saved grid plot to: {plot_filename}")
        plt.close(fig)

        json_filename = os.path.join(output_dir, f"patch_positions_{os.path.basename(slide_path)}.json")
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
    parser = argparse.ArgumentParser(description="Patch Finder Framework for Digital Pathology")
    parser.add_argument('--slides_dir', type=str, required=True, help='Directory containing .svs slide files')
    parser.add_argument('--onnx_model', type=str, required=True, help='Path to ONNX model for tissue segmentation')
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
                run_patch_finder_framework(file_path, args.onnx_model, args.target_level, args.patch_size, output_dir=args.output_dir)
        
    if not found_svs_files:
        print("No .svs files were found in the specified directory.")

if __name__ == "__main__":
    main()
