import openslide
from PIL import Image, ImageDraw
import json
import os

def draw_patches_with_filled_indices(slide_path, json_path, patch_size, level_L, level_T, fill_indices=None):
    """
    Draws all patch rectangles from JSON (stored at level 0) on a thumbnail (level_T),
    with proper scaling of patch size (from level_L → level_T).
    Fills selected patches (given by indices) with a semi-transparent color.
    """

    if fill_indices is None:
        fill_indices = []

    try:
        slide = openslide.OpenSlide(slide_path)
    except Exception as e:
        print(f"Error opening slide: {e}")
        return

    # --- Load patch coordinates ---
    with open(json_path, 'r') as f:
        coords = json.load(f)

    # --- Compute scaling factors ---
    downsample_L = slide.level_downsamples[level_L]
    downsample_T = slide.level_downsamples[level_T]
    print(f"downsample_L={downsample_L:.2f}, downsample_T={downsample_T:.2f}")

    # --- Compute patch size at thumbnail level ---
    patch_size_T = int(patch_size * (downsample_L / downsample_T))
    print(f"Patch size at level {level_T}: {patch_size_T}")

    # --- Read thumbnail ---
    thumb_dims = slide.level_dimensions[level_T]
    thumbnail_img = slide.read_region((0, 0), level_T, thumb_dims).convert("RGBA")

    # --- Prepare transparent overlay ---
    overlay = Image.new("RGBA", thumbnail_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    # --- Define colors ---
    outline_color = (255, 0, 0, 255)       # solid red outline
    fill_color = (0, 255, 0, 100)          # semi-transparent green fill

    # --- Draw patches ---
    for i, coord in enumerate(coords):
        x_0 = coord['x']
        y_0 = coord['y']

        # Convert from level 0 → level_T
        x_T = int(x_0 / downsample_T)
        y_T = int(y_0 / downsample_T)

        bbox = [x_T, y_T, x_T + patch_size_T, y_T + patch_size_T]

        if i in fill_indices:
            # Draw filled semi-transparent patch
            draw.rectangle(bbox, outline=outline_color, fill=fill_color, width=2)
        else:
            # Draw outline only
            draw.rectangle(bbox, outline=outline_color, width=2)

    # --- Merge overlay with thumbnail ---
    final_img = Image.alpha_composite(thumbnail_img, overlay)

    # --- Save output ---
    os.makedirs("stitches", exist_ok=True)
    output_filename = f"./stitches/patches_filled_L{level_L}_T{level_T}.png"
    final_img.convert("RGB").save(output_filename)
    print(f"✅ Saved thumbnail with filled patches: {output_filename}")

    slide.close()





slide_path = '../Slides_test/TCGA-C8-A12P-01Z-00-DX1.670B5DE8-07B0-4E4C-93FA-FA3DFFCCE50D.svs' 
json_path = '../output_224_lvl_2/patch_positions_TCGA-C8-A12P-01Z-00-DX1.670B5DE8-07B0-4E4C-93FA-FA3DFFCCE50D.svs.json'
patch_size = 224 

level_L = 2

level_T = 3

fill_indices = [0, 3, 5, 10]  # highlight some specific patches

draw_patches_with_filled_indices(slide_path, json_path, patch_size, level_L, level_T, fill_indices)
