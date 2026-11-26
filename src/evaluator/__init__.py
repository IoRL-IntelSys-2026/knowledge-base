"""Тестирует RAG с помощью LLM-судьи."""

import argparse
import pathlib

from ragas import Dataset

# Dataset(name="dataset", backend="local/csv", root_dir="../../data")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Оценивает качество ответов RAG-системы с использованием LLM-судьи.",
    )
    parser.add_argument(
        "--dataset",
        help="Путь до датасета с вопросами и ответами (по умолчанию %(default)s).",
        default=pathlib.Path(__file__).parents[2] / "data" / "dataset.csv",
        type=pathlib.Path
    )
    
def main() -> None:
    pass
