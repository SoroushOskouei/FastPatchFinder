import os
import argparse
import openslide
import numpy as np
import onnxruntime as ort
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def get_tissue_bbox_svd(thumbnail_rgb, n_vectors=5, threshold=215, min_threshold=25):
    thumbnail = np.array(thumbnail_rgb)
    gray = np.dot(thumbnail[..., :3], [0.299, 0.587, 0.114])
    U, s, Vt = np.linalg.svd(gray, full_matrices=False)
    k = n_vectors
    reconstructed = np.dot(U[:, :k] * s[:k], Vt[:k, :])
    svd_mask = (reconstructed < threshold) & (reconstructed > min_threshold)

    rows = np.any(svd_mask, axis=1)
    cols = np.any(svd_mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        return (0, 0, thumbnail.shape[1], thumbnail.shape[0]), thumbnail, (reconstructed, svd_mask)

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    bbox = (int(x_min), int(y_min), int(x_max), int(y_max))
    cropped_thumb = thumbnail[y_min : y_max+1, x_min : x_max+1, :]

    return bbox, cropped_thumb, (reconstructed, svd_mask)

def predict_mask_on_thumbnail(thumbnail, onnx_sess, model_input_size=500):
    img_resized = cv2.resize(thumbnail, (model_input_size, model_input_size), interpolation=cv2.INTER_LINEAR)
    img_float = img_resized.astype(np.float32)
    img_bgr = img_float[..., ::-1]
    mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)
    img_preprocessed = img_bgr - mean
    inp = np.expand_dims(img_preprocessed, axis=0)
    inp = inp.astype(np.float32)
    
    input_name = onnx_sess.get_inputs()[0].name
    pred = onnx_sess.run(None, {input_name: inp})[0]
    pred = np.squeeze(pred)
    
    orig_h, orig_w = thumbnail.shape[:2]
    pred_mask = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    
    return (pred_mask > 0.5).astype(np.uint8)

def run_mask_generation(slide_path, onnx_model_path, output_dir='./output'):
    print(f"Processing: {os.path.basename(slide_path)}")
    
    try:
        slide = openslide.OpenSlide(slide_path)
        th_level = slide.level_count - 1
        w_th, h_th = slide.level_dimensions[th_level]
        
        thumb_image_pil = slide.read_region((0, 0), th_level, (w_th, h_th)).convert("RGB")
        thumb_np = np.array(thumb_image_pil)
        
        slide_output_dir = os.path.join(output_dir, os.path.splitext(os.path.basename(slide_path))[0])
        os.makedirs(slide_output_dir, exist_ok=True)

        svd_bbox_tight, _, _ = get_tissue_bbox_svd(
            thumb_np, n_vectors=15, threshold=215, min_threshold=25
        )
        t_x1, t_y1, t_x2, t_y2 = svd_bbox_tight
        
        pad_amount = 64
        p_x1 = max(0, t_x1 - pad_amount)
        p_y1 = max(0, t_y1 - pad_amount)
        p_x2 = min(w_th, t_x2 + pad_amount)
        p_y2 = min(h_th, t_y2 + pad_amount)

        cropped_thumb_padded = thumb_np[p_y1 : p_y2, p_x1 : p_x2, :]

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        sess = ort.InferenceSession(onnx_model_path, providers=providers)

        crop_mask = predict_mask_on_thumbnail(cropped_thumb_padded, sess)
        
        full_mask = np.zeros((h_th, w_th), dtype=np.uint8)
        crop_h, crop_w = crop_mask.shape
        full_mask[p_y1 : p_y1 + crop_h, p_x1 : p_x1 + crop_w] = crop_mask

        mask_filename = os.path.join(slide_output_dir, "tissue_mask.png")
        Image.fromarray(full_mask * 255).save(mask_filename)
        
        viz_filename = os.path.join(slide_output_dir, "mask_overlay.png")
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(thumb_np)
        ax.imshow(full_mask, cmap='jet', alpha=0.5)
        ax.axis('off')
        plt.savefig(viz_filename, bbox_inches='tight')
        plt.close(fig)

        slide.close()
        print(f"Saved mask to {mask_filename}")
        
    except Exception as e:
        print(f"Error processing {slide_path}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slides_dir', type=str, required=True)
    parser.add_argument('--onnx_model', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./output')

    args = parser.parse_args()

    if not os.path.isdir(args.slides_dir):
        print(f"Error: Directory not found: {args.slides_dir}")
        return
        
    if not os.path.exists(args.onnx_model):
        print(f"Error: Model not found: {args.onnx_model}")
        return

    valid_extensions = ('.mrxs', '.svs', '.tif')
    found_files = False

    for root, _, files in os.walk(args.slides_dir):
        for filename in files:
            if filename.lower().endswith(valid_extensions):
                found_files = True
                file_path = os.path.join(root, filename)
                run_mask_generation(
                    file_path, 
                    args.onnx_model, 
                    output_dir=args.output_dir
                )
        
    if not found_files:
        print("No .mrxs, .svs, or .tif files found.")

if __name__ == "__main__":
    main()
