"""Тестирует RAG с помощью LLM-судьи."""

import argparse
import asyncio
import datetime as dt
from os import environ
from pathlib import Path
from typing import Final, NamedTuple, cast

environ["RAGAS_DO_NOT_TRACK"] = "true"

from openai import AsyncOpenAI
from pydantic import BaseModel, HttpUrl, TypeAdapter
from ragas import Dataset, experiment
from ragas.cache import DiskCacheBackend
from ragas.embeddings.base import BaseRagasEmbedding, embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerCorrectness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)


class MetricGroup(NamedTuple):
    """Критерии оценивания ответов RAG-системы."""

    correctness: AnswerCorrectness
    relevancy: AnswerRelevancy
    precision: ContextPrecision
    recall: ContextRecall
    faithfulness: Faithfulness


class EvalRow(BaseModel):
    """Формат тестового датасета."""

    user_input: str
    reference: str


class ModelServerURL(BaseModel):
    """Адрес сервера с моделью-судьёй."""

    url: HttpUrl


def parse_args(
    args: list[str] | None, *, exit_on_error: bool = False
) -> argparse.Namespace:
    """Обрабатывает и проверяет переданные аргументы.

    Parameters
    ----------
    args : list of str, optional
        Список аргументов. Если None, получает аргументы из командной строки.
    exit_on_error : bool
        Выбирает, выходить или бросать исключение, если при обработке
        аргументов возникла ошибка. Нужно для программного использования
        обработчика.

    Returns
    -------
    argparse.Namespace
        Именное пространство с обработанными аргументами.
    """
    adapter: Final = TypeAdapter(ModelServerURL)
    parser: Final = argparse.ArgumentParser(
        description=(
            "Оценивает качество ответов RAG-системы с использованием LLM-судьи."
        ),
        exit_on_error=exit_on_error,
    )
    parser.add_argument(
        "--dataset",
        help=(
            "Путь до датасета с вопросами и ответами (по умолчанию %(default)s)."
        ),
        default=Path(__file__).parents[2] / "data" / "dataset.csv",
        type=Path,
    )
    parser.add_argument(
        "--model",
        help="Название модели-судьи.",
        required=True,
    )
    parser.add_argument(
        "--address",
        help="Адрес LLM-интерфейса.",
        type=lambda x: str(adapter.validate_python(x).url),
    )
    return parser.parse_args(args)


@experiment()
async def run_rag(
    row: EvalRow, model_name: str, metrics: MetricGroup, timestamp: str
) -> dict[str, float | str]:
    """Проверяет ответ RAG-системы.

    Проверяет ответ на текущий запрос по нескольким метрикам.

    Parameters
    ----------
    row : EvalRow
        Текущий вопрос и эталонный ответ из датасета.
    model_name : str
        Имя тестируемой RAG-системы, используется в имени файла
        с результатами.
    metrics : MetricGroup
        Экземпляры метрик, по которым измеряется RAG-система.
    timestamp : str
        Текущие дата и время. Используются в имени файла с
        результатами.

    Returns
    -------
    dict of {str : float or str}
        Оценки ответа на текущий вопрос.
    """
    # FIXME: здесь запрос к RAG
    response, contexts = "", [""]
    async with asyncio.TaskGroup() as tg:
        correctness: Final = tg.create_task(
            metrics.correctness.ascore(
                user_input=row.user_input,
                response=response,
                reference=row.reference,
            )
        )
        relevancy: Final = tg.create_task(
            metrics.relevancy.ascore(
                user_input=row.user_input,
                response=response,
            )
        )
        precision: Final = tg.create_task(
            metrics.precision.ascore(
                user_input=row.user_input,
                reference=row.reference,
                retrieved_contexts=contexts,
            )
        )
        recall: Final = tg.create_task(
            metrics.recall.ascore(
                user_input=row.user_input,
                reference=row.reference,
                retrieved_contexts=contexts,
            )
        )
        faithfulness: Final = tg.create_task(
            metrics.faithfulness.ascore(
                user_input=row.user_input,
                response=response,
                retrieved_contexts=contexts,
            )
        )
    return {
        "user_input": row.user_input,
        "reference": row.reference,
        "correctness": correctness.result().value,
        "relevancy": relevancy.result().value,
        "precision": precision.result().value,
        "recall": recall.result().value,
        "faithfulness": faithfulness.result().value,
        "experiment_name": f"{model_name}_{timestamp}",
    }


def main() -> None:
    """Точка входа в скрипт. Собирает аргументы и запускает проверку RAG."""
    args: Final = parse_args(None)
    dataset: Final[Dataset[EvalRow]] = Dataset(
        name=args.dataset.stem,
        backend="local/csv",
        root_dir=args.dataset.parent,
        data_model=EvalRow,
    )
    client: Final = AsyncOpenAI(base_url=args.address, api_key="ollama")
    cache: Final = DiskCacheBackend(
        str(Path(__file__).parents[2] / ".ragas_cache")
    )
    llm: Final = llm_factory(
        args.model,
        # FIXME: какой формат? OpenAI?
        client=client,
        cache=cache,
    )
    embeddings: Final = cast(
        "BaseRagasEmbedding", embedding_factory(client=client, cache=cache)
    )
    asyncio.run(
        run_rag.arun(
            dataset,
            metrics=MetricGroup(
                AnswerCorrectness(llm=llm),
                AnswerRelevancy(llm=llm, embeddings=embeddings),
                ContextPrecision(llm=llm),
                ContextRecall(llm=llm),
                Faithfulness(llm=llm),
            ),
            model_name="ИМЯ",
            timestamp=dt.datetime.now(dt.UTC).strftime("%F_%H_%M_%S"),
        )
    )
