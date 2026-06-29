<p align="center">
  <h1 align="center">Contrastive and Adaptive Multi-modal Masked Autoencoder for Spatial Transcriptomics</h1>
  <h3 align="center"><b>MICCAI 2026</b></h3>
  <p align="center">
    <h3 align="center">
      <a href="https://kyyle2114.github.io/"><strong>Joohyeok Kim</strong></a> · 
      <a href="https://starfortj.github.io/"><strong>Taejin Jeong</strong></a> · 
      <a href="https://rubato-yeong.github.io/"><strong>Jinyeong Kim</strong></a> · 
      <a href="https://micv.yonsei.ac.kr/seongjae"><strong>Seong Jae Hwang</strong></a>
    </h3>
    <h3 align="center">
      Yonsei University
    </h3>
  </p>
</p>
<p align="center">
    <a href="https://arxiv.org/abs/2606.21156"><img alt='arXiv' src="https://img.shields.io/badge/arXiv-2606.21156-b31b1b.svg"></a>
</p>
 
<be>

## Overview

This repository contains the official implementation of **CAMMST (Contrastive and Adaptive Multi-modal Masked Autoencoder for Spatial Transcriptomics)**.

**Note:** The overall experimental protocol is based on the code from **[MERGE](https://github.com/ags3927/MERGE) (Multi-faceted Hierarchical Graph-based GNN for Gene Expression Prediction from Whole Slide Histopathology Images)**, which was published at CVPR 2025. We gratefully acknowledge the authors for making their code publicly available.

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended). The code was developed and tested with **CUDA 12.1**.

Install dependencies:

```bash
pip install -r requirements.txt
```

For PyTorch with CUDA, install the appropriate build from the PyTorch website before installing the rest. For CUDA 12.1, use the matching PyTorch wheel (e.g. `cu121` index).

## Stage 1: Data Preparation

Three public spatial transcriptomics datasets are used: Her2ST, SKIN (cSCC), and ST-Net.

The data file (`data.tar.gz`) can be downloaded from the **[MERGE](https://github.com/ags3927/MERGE) repository**. Extract it using:

```bash
tar -xvf data.tar.gz
```

After extraction, the directory structure should be:

```
data/
├── her2st/
│   ├── barcodes/
│   ├── counts_spcs_to_8n/
│   ├── ...
│   └── wsi/
├── skin/
│   └── ...
└── stnet/
    └── ...
```

## Stage 2: UNI Feature Extraction

Image embeddings are extracted from histopathology patches using a pretrained UNI2-h model. Run:

```bash
python extract_uni_features.py
```

This processes all three datasets and writes features under `uni_features/<dataset_name>/<slide_name>/uni_features.npy`. The script uses the UNI2-h model from the Hugging Face hub. 

**Note:** UNI2-h requires permission from the original authors; apply through the Hugging Face model page before use.

## Stage 3: CAMMST Training

Training uses 8-fold cross-validation. The main configuration is in `config/default.yaml`. Override paths and options via the command line or by editing the config.

**Single dataset (example: stnet):**

```bash
bash scripts/stnet.sh
```

Or run manually:

```bash
python main.py \
  --config config/default.yaml \
  --set data.dataset_path=./data/stnet \
  --set general.output_dir=./output_dir/stnet
```

**For all datasets:** run `scripts/her2st.sh`, `scripts/skin.sh`, and `scripts/stnet.sh` (set `CUDA_VISIBLE_DEVICES` in each script as needed).

Training outputs are saved under `output_dir/<dataset_name>/fold_<k>/`, including the best model checkpoint and config.

## Stage 4: Inference

Inference can be run after training. The provided scripts (e.g. `scripts/stnet.sh`) also run inference for visible ratios 0.0, 0.1, and 0.3. To run inference only:

```bash
python inference.py \
  --config ./output_dir/stnet/fold_0/config.yaml \
  --output_dir ./output_dir/stnet \
  --visible_ratio 0.1 \
  --inference_output_dir ./inference_results/stnet/0.1
```

Results are written to the specified inference output directory, including per-fold metrics and an aggregate summary.

## Project Structure

```
├── config/           # configuration schema and default YAML
├── engines/          # training and evaluation loop
├── models/           # CAMMST, UNI, and auxiliary modules
├── utils/            # data loaders, metrics, misc helpers
├── scripts/          # shell scripts for her2st, skin, stnet
├── main.py           # training entrypoint
├── inference.py      # inference entrypoint
├── extract_uni_features.py
└── requirements.txt
```

## Configuration

Key options in `config/default.yaml`:

- **data:** `dataset_path`, `feature_path` (UNI features), `folds` (default 8)
- **model:** `embed_dim`, `num_genes` (250), `visible_ratio`, `joint_depth`, `sampler_type`, etc.
- **training:** `epoch`, `patience`, `lr`, `weight_decay`, `warmup_epochs`
- **loss:** reconstruction (prediction), PCC, sampling, and contrastive loss weights

Override any value with `--set key.subkey=value` when calling `main.py`.

## Citation

If you found this code useful, please cite the following paper:

```bibtex
@article{cammst2026,
  title={Contrastive and Adaptive Multi-modal Masked Autoencoder for Spatial Transcriptomics},
  author={Kim, Joohyeok and Jeong, Taejin and Kim, Jinyeong and Hwang, Seong Jae},
  journal={arXiv preprint arXiv:2606.21156},
  year={2026}
}
```