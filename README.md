# FastPatchFinder
Fast Patch Finder Framework for digital pathology Supporting Openslide and level input



This framework is designed to find patches in whole slide images (WSIs) using tissue segmentation via a pre-trained ONNX model. It provides functionality for extracting patches based on tissue coverage and saves both the overlayed patch image and the positions of valid patches in a JSON file.

## Requirements

- Python 3.6+
- Required Python packages:
  - `onnxruntime`
  - `openslide-python`
  - `numpy`
  - `opencv-python`
  - `matplotlib`
  - `Pillow`
  - `argparse`

You can install the dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Usage

Clone this repository:

```bash
git clone https://github.com/yourusername/patch-finder-framework.git
cd patch-finder-framework
```

## Run the framework using the following command:
```bash
python patch_finder.py --slides_dir <path_to_slide_directory> --onnx_model <path_to_onnx_model> --target_level <target_level> --patch_size <patch_size> --output_dir <output_directory>
```

## Example:
```bash
python patch_finder.py --slides_dir ./slides --onnx_model ./best_unet.onnx --target_level 3 --patch_size 225 --output_dir ./output
```
