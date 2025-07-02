import asyncio
import copy
import json
from pathlib import Path

import pandas as pd
import ragas
from langchain_openai import ChatOpenAI
from loguru import logger
from ragas import EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from src.eval.utils import LangMap
from src.mocks import RagPipeline
from src.models import TestConfig


class RagEvaluation:
    """Evaluation pipeline with Ragas."""

    def __init__(
        self,
        cfg: TestConfig,
        rag: RagPipeline,
        metrics: list | None = None,
    ):
        self.rag = rag
        self.cfg = cfg
        self.llm = LangchainLLMWrapper(ChatOpenAI(model=cfg.eval_llm))
        self.metrics = metrics or [
            AnswerRelevancy(llm=self.llm),
            Faithfulness(llm=self.llm),
            ContextRecall(llm=self.llm),
            ContextPrecision(llm=self.llm),
        ]

    def adapt_scorers(self, lang: str) -> list:
        """Adapt the metrics to the specified language."""
        metrics = copy.deepcopy(self.metrics)
        lang = LangMap.normalize_lang(lang)
        if lang == "en":
            return metrics

        for scorer in metrics:
            adapted_prompts = asyncio.run(
                scorer.adapt_prompts(language=lang, llm=self.llm)
            )
            scorer = scorer.set_prompts(**adapted_prompts)
        return metrics

    def evaluate_dataset(
        self, data: Path, ignore_override: bool = False
    ) -> None:
        """Evaluate the RAG by calling it on the QA data."""
        eval_path = data.parent / f"eval_{data.stem}.json"
        if not ignore_override and eval_path.exists():
            logger.warning(f"Skipping not to override: {eval_path}")
            return

        logger.info(f"Collecting RAG answers for the dataset: {data}")
        df = pd.read_csv(data)
        if self.cfg.debug_limit:
            df = df.iloc[: self.cfg.debug_limit]

        dataset = []
        for _, row in df.iterrows():
            answer, context = self.rag.answer(  # type: ignore[misc]
                question=row["user_input"],
                return_retrieved_context=True,
            )
            dataset.append(
                {
                    "user_input": row["user_input"],
                    "retrieved_contexts": context,
                    "response": answer,
                    "reference": row["reference"],
                }
            )
        logger.info("Evaluating dataset..")
        ## FIXME: currently causes something like a thread lock:
        # lang = data.stem.split("-")[1]
        # metrics = self.adapt_scorers(lang)
        ## Likely a problem with sync-async execution.
        eval_results = ragas.evaluate(
            dataset=EvaluationDataset.from_list(dataset),
            metrics=self.metrics,
        )
        with open(eval_path, "w", encoding="utf-8") as fd:
            json.dump(str(eval_results), fd, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {eval_path}")

    def evaluate(
        self, data_paths: list[Path], ignore_override: bool = False
    ) -> None:
        """Evaluate separated language datasets."""
        for path in data_paths:
            self.evaluate_dataset(path, ignore_override)
