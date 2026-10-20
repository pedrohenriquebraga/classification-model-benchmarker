"""Benchmark de modelos de classificacao em TensorFlow Lite.

Formatos de dataset suportados:
  1. Diretorios: dataset/<classe>/*.jpg
  2. CSV com cabecalho contendo ``path`` e ``label``. Caminhos relativos sao
     resolvidos em relacao ao diretorio do CSV.

Exemplo:
    python main.py --model modelo.tflite --dataset dataset --output resultados
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image
from PIL import ImageDraw, ImageFont


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str


def load_interpreter(model_path: Path, threads: int) -> Any:
    """Carrega um Interpreter sem importar TensorFlow antes de ser necessario."""
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            from tensorflow.lite.python.interpreter import Interpreter
        except ImportError as error:
            try:
                from tensorflow.lite import Interpreter
            except ImportError:
                raise RuntimeError(
                    "Nenhum backend TensorFlow Lite foi encontrado. Execute "
                    "'python -m pip install -r requirements.txt' usando o mesmo "
                    "interpretador que executa este script."
                ) from error

    return Interpreter(model_path=str(model_path), num_threads=threads)


def read_dataset(dataset_path: Path) -> list[Sample]:
    if dataset_path.is_file():
        with dataset_path.open(newline="", encoding="utf-8-sig") as file:
            rows = list(csv.DictReader(file))
        if not rows or "path" not in rows[0] or "label" not in rows[0]:
            raise ValueError("O CSV deve possuir as colunas 'path' e 'label'.")
        base = dataset_path.parent
        return [
            Sample(
                path=(base / row["path"]).resolve()
                if not Path(row["path"]).is_absolute()
                else Path(row["path"]),
                label=row["label"],
            )
            for row in rows
        ]

    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset nao encontrado: {dataset_path}")

    samples = [
        Sample(path=image_path, label=class_dir.name)
        for class_dir in sorted(path for path in dataset_path.iterdir() if path.is_dir())
        for image_path in sorted(class_dir.rglob("*"))
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not samples:
        raise ValueError(
            "Nenhuma imagem encontrada. Use dataset/<classe>/*.jpg ou um CSV."
        )
    return samples


def input_spec(interpreter: Any) -> tuple[tuple[int, int], int, np.dtype[Any], tuple[float, int]]:
    details = interpreter.get_input_details()
    if len(details) != 1:
        raise ValueError("O benchmark espera exatamente uma entrada no modelo.")
    shape = tuple(int(value) for value in details[0]["shape"])
    if len(shape) != 4 or shape[0] != 1:
        raise ValueError(f"Entrada esperada no formato [1, altura, largura, canais]: {shape}")
    if shape[1] <= 0 or shape[2] <= 0 or shape[3] not in (1, 3, 4):
        raise ValueError(f"Shape de entrada nao suportado: {shape}")
    quantization = details[0].get("quantization", (0.0, 0))
    return (shape[1], shape[2]), shape[3], np.dtype(details[0]["dtype"]), quantization


def preprocess_image(
    image_path: Path,
    size: tuple[int, int],
    channels: int,
    dtype: np.dtype[Any],
    normalization: str,
    quantization: tuple[float, int],
) -> np.ndarray:
    if channels == 1:
        mode = "L"
    elif channels == 3:
        mode = "RGB"
    elif channels == 4:
        mode = "RGBA"
    else:
        raise ValueError(f"Quantidade de canais nao suportada: {channels}")
    with Image.open(image_path) as image:
        image = image.convert(mode).resize((size[1], size[0]), Image.Resampling.BILINEAR)
        array = np.asarray(image)

    if array.ndim == 2:
        array = array[:, :, np.newaxis]
    if normalization == "zero_one":
        array = array.astype(np.float32) / 255.0
    elif normalization == "minus_one_one":
        array = array.astype(np.float32) / 127.5 - 1.0
    elif normalization == "none":
        array = array.astype(np.float32)
    else:
        raise ValueError(f"Normalizacao desconhecida: {normalization}")

    if np.issubdtype(dtype, np.floating):
        result = array
    elif np.issubdtype(dtype, np.integer):
        scale, zero_point = quantization
        if scale <= 0:
            raise ValueError("O modelo possui quantizacao de entrada invalida.")
        result = np.round(array / scale + zero_point).astype(dtype)
    else:
        raise ValueError(f"Tipo de entrada nao suportado: {dtype}")
    return result[np.newaxis, ...]


def output_class(output: np.ndarray) -> int:
    values = np.asarray(output).reshape(-1)
    if values.size == 0:
        raise ValueError("O modelo retornou uma saida vazia.")
    return int(np.argmax(values))


def class_names_from_samples(samples: Sequence[Sample]) -> list[str]:
    return sorted({sample.label for sample in samples})


def model_labels_from_file(labels_file: Path | None) -> list[str] | None:
    if labels_file is None:
        return None
    with labels_file.open(encoding="utf-8") as file:
        labels = [line.strip() for line in file if line.strip()]
    if not labels:
        raise ValueError("O arquivo de labels esta vazio.")
    return labels


def benchmark_model(
    model_path: Path,
    samples: Sequence[Sample],
    class_names: Sequence[str],
    model_labels: Sequence[str] | None,
    normalization: str,
    warmup: int,
    threads: int,
) -> dict[str, Any]:
    interpreter = load_interpreter(model_path, threads)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    if len(output_details) != 1:
        raise ValueError("O benchmark espera exatamente uma saida no modelo.")
    size, channels, input_dtype, quantization = input_spec(interpreter)
    label_to_index = {label: index for index, label in enumerate(class_names)}
    if model_labels is not None:
        missing = sorted(set(class_names) - set(model_labels))
        if missing:
            raise ValueError(
                "As classes do dataset nao foram encontradas no arquivo de labels "
                f"do modelo: {', '.join(missing)}"
            )
        model_label_to_indices: dict[str, set[int]] = {}
        for index, label in enumerate(model_labels):
            model_label_to_indices.setdefault(label, set()).add(index)
        selected_model_indices = {
            index
            for label in class_names
            for index in model_label_to_indices.get(label, set())
        }
    else:
        selected_model_indices = set(range(len(class_names)))

    first_input = preprocess_image(
        samples[0].path, size, channels, input_dtype, normalization, quantization
    )
    interpreter.set_tensor(input_details["index"], first_input)
    for _ in range(max(0, warmup)):
        interpreter.invoke()

    matrix = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    correct = 0
    inference_seconds = 0.0
    started_total = time.perf_counter()
    progress_interval = max(1, len(samples) // 100)
    for sample_index, sample in enumerate(samples, start=1):
        tensor = preprocess_image(
            sample.path, size, channels, input_dtype, normalization, quantization
        )
        interpreter.set_tensor(input_details["index"], tensor)
        started = time.perf_counter()
        interpreter.invoke()
        inference_seconds += time.perf_counter() - started
        predicted = output_class(interpreter.get_tensor(output_details[0]["index"]))
        expected = label_to_index[sample.label]
        if predicted in selected_model_indices:
            if model_labels is None:
                predicted_label_index = predicted
            else:
                predicted_label = model_labels[predicted]
                predicted_label_index = label_to_index.get(predicted_label)
            if predicted_label_index is not None:
                matrix[expected, predicted_label_index] += 1
                correct += int(expected == predicted_label_index)
        if sample_index == 1 or sample_index % progress_interval == 0:
            elapsed = time.perf_counter() - started_total
            rate = sample_index / elapsed if elapsed else 0.0
            remaining = (len(samples) - sample_index) / rate if rate else 0.0
            print(
                f"\rAvaliando {sample_index}/{len(samples)} "
                f"({sample_index / len(samples):.0%}) - "
                f"{rate:.2f} imagens/s - ETA {remaining / 60:.1f} min",
                end="",
                flush=True,
            )
    print()

    total = len(samples)
    return {
        "model": str(model_path),
        "samples": total,
        "correct": correct,
        "errors": total - correct,
        "accuracy": correct / total,
        "inference_seconds": inference_seconds,
        "inferences_per_second": total / inference_seconds if inference_seconds else None,
        "input_shape": [1, size[0], size[1], channels],
        "input_dtype": str(input_dtype),
        "normalization": normalization,
        "threads": threads,
        "labels": list(class_names),
        "confusion_matrix": matrix.tolist(),
    }


def write_results(results: Sequence[dict[str, Any]], output_dir: Path) -> None:
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    for result in results:
        model_name = Path(result["model"]).stem
        model_output_dir = output_dir / f"{model_name}-{timestamp}"
        model_output_dir.mkdir(parents=True, exist_ok=True)
        (model_output_dir / "results.json").write_text(
            json.dumps([result], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        filename = model_name + "_confusion_matrix.csv"
        with (model_output_dir / filename).open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["actual\\predicted", *result["labels"]])
            writer.writerows(
                [label, *row]
                for label, row in zip(result["labels"], result["confusion_matrix"])
            )
        (model_output_dir / "results.txt").write_text(
            "\n".join(format_text_report(result)), encoding="utf-8"
        )
        write_confusion_matrix_image(
            result, model_output_dir / (model_name + "_confusion_matrix.png")
        )


def format_text_report(result: dict[str, Any]) -> list[str]:
    lines = [
        f"Modelo: {result['model']}",
        f"Amostras: {result['samples']}",
        f"Acertos: {result['correct']}",
        f"Erros: {result['errors']}",
        f"Acuracia: {result['accuracy']:.4%}",
        f"Tempo de inferencia: {result['inference_seconds']:.3f} s",
        f"Inferencias por segundo: {result['inferences_per_second']:.3f}",
        f"Formato de entrada: {result['input_shape']}",
        f"Tipo de entrada: {result['input_dtype']}",
        f"Normalizacao: {result['normalization']}",
        f"Threads: {result['threads']}",
        "",
        "Matriz de confusao (real nas linhas, previsto nas colunas):",
    ]
    labels = result["labels"]
    matrix = result["confusion_matrix"]
    lines.append("real\\previsto\t" + "\t".join(labels))
    lines.extend(
        label + "\t" + "\t".join(str(value) for value in row)
        for label, row in zip(labels, matrix)
    )
    lines.extend(["", "-" * 80, ""])
    return lines


def write_confusion_matrix_image(result: dict[str, Any], output_path: Path) -> None:
    labels = result["labels"]
    matrix = np.asarray(result["confusion_matrix"], dtype=np.int64)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    bold_font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    font = ImageFont.truetype(font_path, 18) if font_path.exists() else ImageFont.load_default()
    bold_font = (
        ImageFont.truetype(bold_font_path, 18)
        if bold_font_path.exists()
        else font
    )
    cell_size = 90
    margin_left = 150
    margin_top = 115
    image = Image.new(
        "RGB",
        (
            margin_left + cell_size * len(labels) + 20,
            margin_top + cell_size * len(labels) + 20,
        ),
        "white",
    )
    draw = ImageDraw.Draw(image)
    maximum = int(matrix.max()) if matrix.size else 0
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            intensity = int(255 - (value / maximum * 200)) if maximum else 255
            color = (intensity, intensity, 255)
            left = margin_left + column_index * cell_size
            top = margin_top + row_index * cell_size
            draw.rectangle(
                (left, top, left + cell_size, top + cell_size),
                fill=color,
                outline=(80, 80, 80),
            )
            text = str(int(value))
            bounds = draw.textbbox((0, 0), text, font=bold_font)
            draw.text(
                (
                    left + (cell_size - bounds[2] + bounds[0]) / 2,
                    top + (cell_size - bounds[3] + bounds[1]) / 2,
                ),
                text,
                fill="black",
                font=bold_font,
            )
    draw.text((margin_left, 20), "Matriz de confusao", fill="black", font=bold_font)
    draw.text((margin_left, 50), "Previsto", fill="black", font=font)
    draw.text((20, margin_top - 35), "Real", fill="black", font=font)
    for index, label in enumerate(labels):
        column_left = margin_left + index * cell_size
        row_top = margin_top + index * cell_size
        draw.text((column_left + 8, margin_top - 30), label, fill="black", font=font)
        draw.text((20, row_top + 32), label, fill="black", font=font)
    image.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", nargs="+", type=Path, required=True, help="Modelos .tflite")
    parser.add_argument("--dataset", type=Path, required=True, help="Pasta ou CSV de avaliacao")
    parser.add_argument("--labels", type=Path, help="Um label por linha, na ordem da saida do modelo")
    parser.add_argument(
        "--normalization",
        choices=("zero_one", "minus_one_one", "none"),
        default="zero_one",
        help="Pre-processamento para modelos float (padrao: zero_one)",
    )
    parser.add_argument("--warmup", type=int, default=5, help="Invocacoes descartadas antes da medicao")
    parser.add_argument(
        "--threads",
        type=int,
        default=os.cpu_count() or 1,
        help="Threads usadas pelo TensorFlow Lite (padrao: todas as CPUs)",
    )
    parser.add_argument("--output", type=Path, default=Path("benchmark_results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup < 0:
        raise ValueError("--warmup nao pode ser negativo.")
    if args.threads < 1:
        raise ValueError("--threads deve ser positivo.")
    samples = read_dataset(args.dataset)
    labels = class_names_from_samples(samples)
    model_labels = model_labels_from_file(args.labels)
    results = [
        benchmark_model(
            model,
            samples,
            labels,
            model_labels,
            args.normalization,
            args.warmup,
            args.threads,
        )
        for model in args.model
    ]
    write_results(results, args.output)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()