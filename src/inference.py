"""Run DeepSeek-OCR-2 baseline and/or LoRA inference on a test-set row.

Examples:
    python src/inference.py --row 0
    python src/inference.py --row 10 --model lora
    python src/inference.py --model both --output-dir src/output

If --row is omitted, every row in the test split is processed. Predictions,
per-sample metrics, and the micro-average report are written to the output
directory (src/output by default).
"""

import argparse
import json
from pathlib import Path

from utils.config import CONFIG_PATH, load_config, resolve_path
from utils.evaluation import (
    aggregate_micro,
    evaluate_sample,
    print_micro_average,
    print_sample_result,
)
from utils.inference_engine import DeepSeekOCREngine
from utils.test_dataset import (
    get_ground_truth,
    get_image,
    get_row_id,
    load_test_split,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--row",
        type=int,
        default=None,
        help="Test-set row index. Omit to run every row.",
    )
    parser.add_argument(
        "--model",
        choices=("baseline", "lora", "both"),
        default="both",
        help="Which model(s) to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write predictions/report. Defaults to config inference.output_dir.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to the YAML config.",
    )
    return parser.parse_args()


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    config = load_config(args.config)
    data_config = config["data"]
    inference_config = config.get("inference", {})
    model_config = config["model"]

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else resolve_path(inference_config.get("output_dir", "output"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_test_split(
        dataset_name=data_config["dataset_name"],
        subset=data_config["subset"],
        split=data_config.get("test_split", "test"),
    )

    if args.row is not None:
        if not -len(dataset) <= args.row < len(dataset):
            raise IndexError(
                f"Row {args.row} is out of range for a test set of size {len(dataset)}."
            )
        indices = [args.row]
    else:
        indices = list(range(len(dataset)))

    local_model = resolve_path(model_config["local_dir"])
    baseline_source = (
        str(local_model)
        if local_model.exists()
        else inference_config.get("baseline_model", model_config["name"])
    )
    adapter_path = resolve_path(
        inference_config.get("adapter_path", model_config["output_path"])
    )

    prompt = inference_config.get("prompt", data_config["instruction"])
    image_size = inference_config.get("image_size", data_config["image_size"])
    base_size = inference_config.get("base_size", data_config["base_size"])
    crop_mode = inference_config.get("crop_mode", data_config["crop_mode"])

    models = ["baseline", "lora"] if args.model == "both" else [args.model]
    if "lora" in models and not adapter_path.exists():
        raise FileNotFoundError(f"LoRA adapter not found: {adapter_path}")

    report = {
        "dataset": data_config["dataset_name"],
        "subset": data_config["subset"],
        "split": data_config.get("test_split", "test"),
        "row_indices": indices,
        "models": {},
    }

    for model_name in models:
        engine = DeepSeekOCREngine(
            baseline_source,
            adapter_path if model_name == "lora" else None,
        )
        records = []
        stats = []
        try:
            for index in indices:
                row = dataset[index]
                prediction = engine.predict(
                    get_image(row),
                    prompt=prompt,
                    image_size=image_size,
                    base_size=base_size,
                    crop_mode=crop_mode,
                )
                record, sample_stats = evaluate_sample(
                    get_ground_truth(row), prediction
                )
                record = {
                    "index": index,
                    "id": get_row_id(row, index),
                    **record,
                }
                records.append(record)
                stats.append(sample_stats)
                print_sample_result(model_name, record)
        finally:
            engine.close()

        summary = aggregate_micro(stats)
        print_micro_average(model_name, summary)

        predictions_path = output_dir / f"{model_name}_predictions.jsonl"
        write_jsonl(predictions_path, records)
        report["models"][model_name] = {
            "model_source": baseline_source,
            "adapter_path": str(adapter_path) if model_name == "lora" else None,
            "micro_average": summary,
            "predictions": records,
        }
        with open(output_dir / "report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"Predictions saved to: {predictions_path}")

    print(f"Report saved to:      {output_dir / 'report.json'}")


if __name__ == "__main__":
    main()
