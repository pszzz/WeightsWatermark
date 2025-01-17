# Watermarking for Transformer models


This project provides two different watermark embedding methods, represented by `WatermarkV1` for single-user mode, and `WatermarkV2` for multi-user mode. Below are the usage methods for these two modes.

## Quick Start

This project is developed using CUDA 12.1, PyTorch 2.2, Python 3.10.  

### Install Dependencies

```bash
pip install torch transformers
```
### Examples

```bash
# Single-user watermark embedding
python watermark_v1.py --model_dir ./models/Llama-3-8B --model_type Llama-3-8B --output_dir ./output --command insert
# Single-user watermark extraction
python watermark_v1.py --model_dir ./output/model/Llama-3-8B_WM_10_50_V1 --model_type Llama-3-8B --ori_model_dir ./models/Llama-3-8B --command extract

# Multi-user watermark embedding
python watermark_v2.py --model_dir ./models/Llama-3-8B --model_type Llama-3-8B --output_dir ./output --userid 1 --command insert
# Multi-user watermark extraction
python watermark_v2.py --model_dir ./output/model/Llama-3-8B_WM_10_50_user1_V2 --model_type Llama-3-8B --ori_model_dir ./models/Llama-3-8B --userid 1 --command extract
```

## Single-user Mode

### Watermark Embedding

To embed a watermark, use the following command:

```bash
python watermark_v1.py --model_dir <model_directory> --model_type <model_type> --t <t> --wm_num <wm_num> --scale_wm <scale_wm> --output_dir <output_directory>
```
- **Parameter Explanation**:
  - `--model_dir`: The directory path of the model.
  - `--model_type`: The type of the model (e.g., "Llama-3-8B").
  - `--t`: The length of the watermark (default value is 10).
  - `--wm_num`: The number of watermarks (default value is 50).
  - `--scale_wm`: The watermark scaling parameter (default value is 1000).
  - `--output_dir`: The output directory (default value is `./output`).
  - `--command`: The command to execute, set to `insert`.

### Watermark Extraction

To extract a watermark, use the following command:

```bash
python watermark_v1.py --command extract --model_dir <model_directory> --model_type <model_type> --ori_model_dir <ori_model_directory> --t <t> --wm_num <wm_num> --numit <numit> --beta <beta> --rho <rho> --scale_wm <scale_wm> --key_dir <key_directory>
```
- **Parameter Explanation**:
  - `--model_dir`: The directory path of the watermark model.
  - `--model_type`: The type of the model (e.g., "Llama-3-8B").
  - `--ori_model_dir`: The directory path of the original model.
  - `--t`: The length of the watermark.
  - `--wm_num`: The number of watermarks (default value is 50).
  - `--numit`: The number of iterations (default value is 10000).
  - `--beta`: The threshold for watermark detection (default value is 35).
  - `--rho`: The threshold for watermark detection (default value is 0.3).
  - `--scale_wm`: The watermark scaling parameter (default value is 1000).
  - `--key_dir`: The directory of the key parameters (default value is `./output`).
  - `--command`: The command to execute, set to `extract`.

## Multi-user Mode

### Watermark Embedding

To embed a watermark, use the following command:

```bash
python watermark_v2.py --command insert --userid <userid> --model_dir <model_directory> --model_type <model_type> --t <t> --wm_num <wm_num> --scale_wm <scale_wm> --output_dir <output_directory> 
```
- **Parameter Explanation**:
  - `--userid`: The user ID.
  - `--model_dir`: The directory path of the model.
  - `--model_type`: The type of the model (e.g., "Llama-3-8B").
  - `--t`: The length of the watermark (default value is 10).
  - `--wm_num`: The number of watermarks (default value is 50).
  - `--scale_wm`: The watermark scaling parameter (default value is 1000).
  - `--output_dir`: The output directory (default value is `./output`).
  - `--command`: The command to execute, set to `insert`.

### Watermark Extraction

To extract a watermark, use the following command:

```bash
python watermark_v2.py --command extract --userid <userid> --model_dir <model_directory> --ori_model_dir <ori_model_directory> --model_type <model_type> --t <t> --wm_num <wm_num> --numit <numit> --beta <beta> --rho <rho> --scale_wm <scale_wm> --key_dir <key_directory>
```
- **Parameter Explanation**:
  - `--userid`: The user ID.
  - `--model_dir`: The directory path of the watermark model.
  - `--model_type`: The type of the model (e.g., "Llama-3-8B").
  - `--ori_model_dir`: The directory path of the original model.
  - `--t`: The length of the watermark.
  - `--wm_num`: The number of watermarks (default value is 50).
  - `--numit`: The number of iterations (default value is 10000).
  - `--beta`: The threshold for watermark detection (default value is 35).
  - `--rho`: The threshold for watermark detection (default value is 0.3).
  - `--scale_wm`: The watermark scaling parameter (default value is 1000).
  - `--key_dir`: The directory of the key parameters (default value is `./output`).
  - `--command`: The command to execute, set to `extract`.


## Citation
If you use this project or code in your research, please cite the following work:
```bibtex
@misc{your_paper,
      title={}, 
      author={},
      year={2025},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.CR}
}
```
