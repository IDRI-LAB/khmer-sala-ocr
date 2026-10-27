# Khmer Sala OCR: DeepSeek-OCR-2 LoRA

This repository contains the training, inference, and evaluation code for the
DeepSeek-OCR-2 Khmer LoRA adapter described in the paper **"Distilling
Multimodal Foundation Models into Lightweight OCR for Low-Resource Khmer"**.

The code is responsible for:

- Loading the Khmer Sala OCR dataset.
- Fine-tuning DeepSeek-OCR-2 with LoRA using Unsloth.
- Running baseline or LoRA inference on any row of the test split.
- Reporting Khmer-normalized CER/WER with micro-averaging.

This repository does **not** host the dataset or the trained adapter. They are
published separately on Hugging Face:

- Dataset: [CADT-IDRI/khmer-sala-ocr-16k](https://huggingface.co/datasets/CADT-IDRI/khmer-sala-ocr-16k)
- LoRA adapter: [CADT-IDRI/deepseek-ocr-2-khmer-lora-16k](https://huggingface.co/CADT-IDRI/deepseek-ocr-2-khmer-lora-16k)
- Base model: [deepseek-ai/DeepSeek-OCR-2](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2) (re-uploaded by [unsloth/DeepSeek-OCR-2](https://huggingface.co/unsloth/DeepSeek-OCR-2))

## Repository structure

```text
.
├── configs/
│   └── lora.yaml              # Model, LoRA, data, inference, and training config
├── src/
│   ├── train.py               # Fine-tuning entry point
│   ├── data.py                # Dataset loading and DeepSeek-OCR-2 data collator
│   ├── inference.py           # Baseline/LoRA inference + evaluation CLI
│   └── utils/
│       ├── config.py          # Shared config loading and path resolution
│       ├── preprocessing.py   # Shared text cleaning for GT and prediction
│       ├── test_dataset.py    # Loads the Hugging Face test split
│       ├── inference_engine.py# Loads baseline/LoRA and runs model.infer
│       ├── evaluation.py      # CER/WER per sample and micro-average
│       ├── khmer_norm.py      # Khmer normalization from KHCWER
│       └── cer_wer_calculation.py  # CER/WER calculation utilities
├── requirements.txt
├── temp.py                    # Legacy reference evaluation script (not used)
└── README.md
```

## Installation

Requirements: Python 3.10+, a CUDA GPU, and PyTorch CUDA 12.4 wheels.

```bash
conda create -n khmer-ocr python=3.11 -y
conda activate khmer-ocr
pip install -r requirements.txt
```

`requirements.txt` pins the versions used by the paper. If pip tries to
downgrade `numpy`, `tqdm`, or `requests` because of `khmer-nlp-toolkits`
metadata, install the toolkit without touching the already-verified environment:

```bash
pip install --no-deps khmer-nlp-toolkits==0.2.2
```

## Configuration

All important settings live in `configs/lora.yaml`:

```yaml
model:
  name: unsloth/DeepSeek-OCR-2  # Base model ID on Hugging Face
  local_dir: deepseek_ocr2  # Local folder to download the base model into (relative to src/)
  load_in_4bit: false  # Load in 4-bit; false means 16-bit LoRA training
  output_path: ../models/deepseek_ocr2_lora_adapter  # Where to save the trained LoRA adapter

lora:
  r: 16  # LoRA rank; higher gives more capacity but may overfit
  lora_alpha: 16  # LoRA scaling factor; usually equal to or greater than r
  lora_dropout: 0  # Dropout probability for LoRA layers
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]  # Layers that get LoRA adapters

data:
  dataset_name: CADT-IDRI/khmer-sala-ocr-16k  # Hugging Face dataset ID
  subset: default  # Dataset subset/config name
  splits: [train, validation]  # Splits used for training and validation
  test_split: test  # Split used by inference and evaluation
  instruction: "<image>\nFree OCR."  # Prompt paired with each input image

inference:
  baseline_model: unsloth/DeepSeek-OCR-2  # Model used when running baseline inference
  adapter_path: ../models/deepseek_ocr2_lora_adapter  # LoRA adapter to load for adapter inference
  output_dir: output  # Folder for predictions and report (relative to src/)

training:
  per_device_train_batch_size: 4  # Training batch size per GPU
  gradient_accumulation_steps: 6  # Steps to accumulate before updating weights
  learning_rate: 2.0e-4  # Peak learning rate
  num_train_epochs: 2  # Number of passes over the training split
  ...  # Remaining TrainingArguments settings are passed directly to the Trainer
```

Paths in `model.local_dir`, `inference.adapter_path`, and
`inference.output_dir` are resolved relative to `src/`. Edit this YAML file
instead of changing Python code when tuning a run.

## Training

Run from the repository root:

```bash
python src/train.py
```

What happens:

1. The base model is downloaded to `src/deepseek_ocr2/`.
2. The `train` and `validation` splits are loaded from the configured dataset.
3. LoRA is applied and training starts with the settings in `training:`.
4. The adapter is saved to `models/deepseek_ocr2_lora_adapter/`.
5. `training_summary.log` and `training_results.json` are written to the
   current working directory.

To save the adapter somewhere else, change `model.output_path` in
`configs/lora.yaml`.

## Inference and evaluation

Run from the repository root. By default the script uses the test split
(`data.test_split`) and writes results to `src/output/`.

```bash
# Run both baseline and LoRA on one row
python src/inference.py --row 0

# Run only the baseline
python src/inference.py --row 0 --model baseline

# Run only the LoRA adapter
python src/inference.py --row 10 --model lora

# Run both models explicitly
python src/inference.py --row 10 --model both

# Run every row in the test split
python src/inference.py

# Choose a custom output directory
python src/inference.py --row 5 --output-dir src/output

# Use a different config file
python src/inference.py --row 5 --config configs/lora.yaml
```

### CLI options

| Option                         | Description                                                                                  |
| ------------------------------ | -------------------------------------------------------------------------------------------- |
| `--row N`                      | Test-set row index. Negative indices are also accepted. Omit to run all rows.                |
| `--model baseline\|lora\|both` | Which model(s) to run. Default: `both`.                                                      |
| `--output-dir PATH`            | Output directory. Default: `inference.output_dir` from the config, resolved to `src/output`. |
| `--config PATH`                | Alternative YAML config. Default: `configs/lora.yaml`.                                       |

### Output files

For each model that runs, `src/output/` receives:

- `baseline_predictions.jsonl` and/or `lora_predictions.jsonl` — one JSON object
  per row containing `index`, `id`, `ground_truth`, `prediction`, and per-sample
  `metrics`.
- `report.json` — dataset metadata, all predictions, and the micro-averaged
  CER/WER for each model.

The console prints the ground truth, prediction, normalized CER/WER for every
row, followed by the micro-average.

## Evaluation protocol

- Ground truth comes from the `test` split of the configured Hugging Face
  dataset.
- The same `clean_text_pipeline` is applied to ground truth and prediction.
- Metrics are computed on the full overall text; there is no LaTeX-specific
  handling or ignored-row list in this pipeline.
- CER applies `clean_and_normalize` from `khmer_norm.py`, then computes
  character-level Levenshtein distance divided by reference length.
- WER uses Khmer word segmentation (`word_type="com"`) from
  `khmer-nlp-toolkits`, then normalizes each word and computes word-level
  Levenshtein distance.
- All reported scores are Khmer-normalized.
- Reported corpus scores are always **micro-averages**: distances and lengths
  are summed across samples and divided once at the end.

## Dataset

The dataset is [CADT-IDRI/khmer-sala-ocr-16k](https://huggingface.co/datasets/CADT-IDRI/khmer-sala-ocr-16k),
using the `default` subset. Rows contain:

- `image`: page image
- `text`: OCR ground truth
- `book_id`, `page`: identifiers
- `label_source`: provenance of the label

Training uses the `train` and `validation` splits. Inference and evaluation use
the `test` split, which contains 100 human-verified pages.

## References and acknowledgements

- Paper: **"Distilling Multimodal Foundation Models into Lightweight OCR for
  Low-Resource Khmer"**. See the [LoRA model card](https://huggingface.co CADT-IDRI/deepseek-ocr-2-khmer-lora-16k) for the latest citation information.
- Dataset: [CADT-IDRI/khmer-sala-ocr-16k](https://huggingface.co/datasets/CADT-IDRI/khmer-sala-ocr-16k)
- Trained adapter: [CADT-IDRI/deepseek-ocr-2-khmer-lora-16k](https://huggingface.co/CADT-IDRI/deepseek-ocr-2-khmer-lora-16k)
- Original Unsloth DeepSeek-OCR-2 notebook:
  [unslothai/notebooks — Deepseek*OCR_2*(3B).ipynb](<https://github.com/unslothai/notebooks/blob/main/nb/Deepseek_OCR_2_(3B).ipynb>)
- Khmer normalization (`src/utils/khmer_norm.py`) is from
  [KHCWER](https://github.com/keosaly/KHCWER), introduced in Nom et al.,
  "WildKhmerST: A Comprehensive Dataset and Benchmark for Khmer Scene Text
  Detection and Recognition in the Wild".
- Khmer word segmentation is provided by
  [khmer-nlp-toolkits](https://github.com/IDRI-LAB/khmer-nlp-toolkits).

## Contributors

- [Kimleangchhay SONG](https://github.com/leangchhay1523)
