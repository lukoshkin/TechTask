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

from src.eval.datagen.synthetic import SyntheticDataset
from src.main import build_rag
from src.models import TestConfig


def load_config() -> TestConfig:
    """Load the test configuration from 'test-config.yaml'."""
    test_cfg_file = "test-config.yaml"
    test_cfg = TestConfig.from_yaml(test_cfg_file)
    logger.debug(f"Loaded configuration: {test_cfg}")
    return test_cfg


def build_dataset() -> list[Path]:
    cfg = load_config()

    dgen = SyntheticDataset(cfg.datagen)
    dgen.sort_by_lang(cfg.chunked_data)
    return dgen.generate_dataset()[0]


def evaluate(data_paths: list[Path]) -> None:
    """Evaluate the dataset."""
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    metrics = [
        AnswerRelevancy(llm=llm),
        Faithfulness(llm=llm),
        ContextRecall(llm=llm),
        ContextPrecision(llm=llm),
    ]

    def adapt_scorers(metrics: list, lang: str) -> list:
        metrics = copy.deepcopy(metrics)
        lang_map = SyntheticDataset.lang_map
        for metric in metrics:
            adapted_prompts = asyncio.run(
                metric.adapt_prompts(language=lang_map[lang], llm=llm)
            )
            metric = metric.set_prompts(**adapted_prompts)
        return metrics

    rag = build_rag()
    for path in data_paths:
        dataset = []
        df = pd.read_csv(path)
        logger.info(f"Collecting answers for the dataset: {path}")
        for _, row in df.iterrows():
            answer, context = rag.answer(  # type: ignore[misc]
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
        # lang = path.stem.split("-")[1]
        # metrics = adapt_scorers(metrics, lang)
        ## Likely a problem with sync-async execution.
        eval_results = ragas.evaluate(
            dataset=EvaluationDataset.from_list(dataset), metrics=metrics
        )
        eval_path = path.parent / f"eval_{path.stem}.json"
        with open(eval_path, "w", encoding="utf-8") as fd:
            json.dump(eval_results, fd, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {eval_path}")


def main() -> None:
    """Run evaluation pipeline."""
    dataset = build_dataset()
    evaluate(dataset)


if __name__ == "__main__":
    main()
