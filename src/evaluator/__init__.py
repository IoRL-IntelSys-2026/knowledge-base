"""Тестирует RAG с помощью LLM-судьи."""

import argparse

from pathlib import Path
from typing import Final
from urllib.parse import urlsplit, urlunsplit

from ragas import Dataset
from ragas.cache import DiskCacheBackend
from ragas.llms import llm_factory

def parse_args(
        args: list[str] | None,
        *,
        exit_on_error: bool = False
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Оценивает качество ответов RAG-системы с использованием LLM-судьи."
        ),
    )
    parser.add_argument(
        "--dataset",
        help=(
            "Путь до датасета с вопросами и ответами (по умолчанию %(default)s)."
        ),
        default=pathlib.Path(__file__).parents[2] / "data" / "dataset.csv",
        type=Path
    )
    parser.add_argument(
        "--model",
        help=(
            'Название модели-судьи. Специальное значение "openai"'
            "использует встроенный режим запросов для OpenAI."
        ),
        required=True,
    )
    parser.add_argument(
        "--llm-address",
        help="Адрес LLM-интерфейса",
        # Валидацию можно добавить, если нужна
        type=lambda x: urlunsplit(urlsplit(x, scheme="http"))
    )
    return parser.parse_args(args, exit_on_error=exit_on_error)
    
def main() -> None:
    args: Final = parse_args(None)
    llm: Final = llm_factory(
        cache=DiskCacheBackend(str(Path(__file__).parents[2] / ".ragas_cache"))
    )
