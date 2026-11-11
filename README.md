# Khmer Sala OCR: DeepSeek-OCR-2 LoRA

Training, inference, and evaluation code for the Khmer Sala OCR model, a
DeepSeek-OCR-2 LoRA adapter trained on the **Khmer Sala OCR** dataset.

- Dataset: [CADT-IDRI/khmer-sala-ocr-16k](https://huggingface.co/datasets/CADT-IDRI/khmer-sala-ocr-16k)
- LoRA adapter: [CADT-IDRI/deepseek-ocr-2-khmer-lora-16k](https://huggingface.co/CADT-IDRI/deepseek-ocr-2-khmer-lora-16k)
- Base model: [unsloth/DeepSeek-OCR-2](https://huggingface.co/unsloth/DeepSeek-OCR-2)

This repository does not host the dataset or model weights.

## Repository layout

```text
.
├── configs/
│   └── lora.yaml              # Model, LoRA, data, inference, and training config
├── requirements.txt
└── src/
    ├── train.py               # Fine-tuning entry point
    ├── data.py                # Dataset loading and collator
    ├── inference.py           # Baseline/LoRA inference + evaluation CLI
    └── utils/
        ├── config.py          # Config loading and path resolution
        ├── preprocessing.py   # Shared text cleaning
        ├── test_dataset.py    # Hugging Face test split loader
        ├── inference_engine.py# Model loading and inference
        ├── evaluation.py      # CER/WER and micro-average
        ├── khmer_norm.py      # Khmer normalization
        └── cer_wer_calculation.py
```

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/IDRI-LAB/khmer-sala-ocr.git
cd khmer-sala-ocr
```

### 2. Create the environment

```bash
conda create -n khmer-ocr python=3.11 -y
conda activate khmer-ocr
```

### 3. Install requirements

```bash
pip install -r requirements.txt
```

The pinned versions target CUDA 12.4, PyTorch 2.6.0, Transformers 4.56.2, and
Unsloth 2026.6.1. Requirements: Python 3.10+ and a CUDA GPU.

If pip tries to downgrade `numpy`, `tqdm`, or `requests` because of
`khmer-nlp-toolkits` metadata, install the toolkit without touching the
already-verified environment:

```bash
pip install --no-deps khmer-nlp-toolkits==0.2.2
```

### 4. Download the released adapter (optional)

Only needed if you want to run inference without training first:

```bash
huggingface-cli download CADT-IDRI/deepseek-ocr-2-khmer-lora-16k \
  --local-dir src/models/deepseek_ocr2_lora_adapter
```

Then set `inference.adapter_path` in `configs/lora.yaml` to
`../models/deepseek_ocr2_lora_adapter`.

### 5. Run

```bash
# Fine-tune DeepSeek-OCR-2
python src/train.py

# Evaluate one test row with baseline + LoRA
python src/inference.py --row 0

# Evaluate LoRA only
python src/inference.py --row 10 --model lora
```

## Configuration

All settings live in `configs/lora.yaml`. Paths in `model.local_dir`,
`inference.adapter_path`, and `inference.output_dir` are resolved relative to
`src/`. Edit the YAML instead of changing Python code when tuning a run.

```yaml
model:
  name: unsloth/DeepSeek-OCR-2
  load_in_4bit: false
  output_path: ../models/deepseek_ocr2_lora_adapter

lora:
  r: 16
  lora_alpha: 16
  lora_dropout: 0

data:
  dataset_name: CADT-IDRI/khmer-sala-ocr-16k
  subset: default
  test_split: test
  instruction: "<image>\nFree OCR."

training:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 6
  learning_rate: 2.0e-4
  num_train_epochs: 2
```

## Training

Run from the repository root:

```bash
python src/train.py
```

Training writes:

- Adapter: `src/models/deepseek_ocr2_lora_adapter/`
- Logs: `training_summary.log`
- Metrics: `training_results.json`
- Loss curves and trainer state in the adapter directory

## Inference and evaluation

Run from the repository root. By default the script uses the `test` split and
writes results to `src/output/`.

```bash
# Run both baseline and LoRA on one row
python src/inference.py --row 0

# Run only the baseline
python src/inference.py --row 0 --model baseline

# Run only the LoRA adapter
python src/inference.py --row 10 --model lora

# Run every row in the test split
python src/inference.py

# Custom output directory or config
python src/inference.py --row 5 --output-dir src/output
python src/inference.py --row 5 --config configs/lora.yaml
```

| Option                         | Description                                                  |
| ------------------------------ | ------------------------------------------------------------ |
| `--row N`                      | Test row index. Negative indices accepted. Omit to run all.  |
| `--model baseline\|lora\|both` | Model(s) to run. Default: `both`.                            |
| `--output-dir PATH`            | Output directory. Default from config (`src/output`).        |
| `--config PATH`                | Alternative YAML config. Default: `configs/lora.yaml`.       |

Outputs are written to `src/output/`:

- `baseline_predictions.jsonl` / `lora_predictions.jsonl`
- `report.json` with predictions and micro-averaged CER/WER

## Metrics

- Ground truth comes from the `test` split.
- The same text cleaning is applied to ground truth and prediction.
- CER uses Khmer-normalized character Levenshtein distance.
- WER uses Khmer word segmentation, then normalized word-level Levenshtein
  distance.
- Only Khmer-normalized CER/WER are reported.
- Corpus scores are always micro-averages.

## Dataset

[CADT-IDRI/khmer-sala-ocr-16k](https://huggingface.co/datasets/CADT-IDRI/khmer-sala-ocr-16k)
(`default` subset):

- `train` and `validation` splits for training
- `test` split (100 human-verified pages) for inference and evaluation

Columns: `image`, `text`, `book_id`, `page`, `label_source`.

## References

- DeepSeek-OCR-2 fine-tuning is adapted from the
  [Unsloth DeepSeek-OCR-2 notebook](<https://github.com/unslothai/notebooks/blob/main/nb/Deepseek_OCR_2_(3B).ipynb>).
- Khmer normalization in `src/utils/khmer_norm.py` is from
  [KHCWER](https://github.com/keosaly/KHCWER).
- Khmer word segmentation is provided by
  [khmer-nlp-toolkits](https://github.com/IDRI-LAB/khmer-nlp-toolkits).

## Contribution

- [Kimleangchhay SONG](https://github.com/leangchhay1523)
