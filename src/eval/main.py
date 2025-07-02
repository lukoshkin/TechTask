import argparse
from pathlib import Path

from loguru import logger

from src.eval.datagen.synthetic import SyntheticDataset
from src.eval.evaluate import RagEvaluation
from src.main import build_rag
from src.models import TestConfig


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns
    -------
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Generate dataset and evaluate RAG pipeline on it."
    )
    parser.add_argument(
        "-0",
        dest="zero_count",
        action="count",
        default=0,
        help=(
            "Use -0 to drop evaluation results,"
            " -00 to drop knowledge graph files,"
            " -000 to drop data portions split by language,"
        ),
    )
    return parser.parse_args()


def load_config() -> TestConfig:
    """Load the test configuration from 'test-config.yaml'."""
    test_cfg_file = "test-config.yaml"
    test_cfg = TestConfig.from_yaml(test_cfg_file)
    logger.debug(f"Loaded configuration: {test_cfg}")
    return test_cfg


def build_dataset(cfg: TestConfig, reset_lvl: int = 0) -> list[Path]:
    """Build the 'question - ground truth answer' dataset with Ragas."""
    dgen = SyntheticDataset(cfg.datagen)
    if dgen.interim_dir.exists():
        if reset_lvl >= 2:
            for path in dgen.output_dir.glob("dset-*.csv"):
                path.unlink()
        if reset_lvl >= 3:
            for path in dgen.interim_dir.iterdir():
                path.unlink()

    dgen.sort_by_lang(cfg.chunked_data)
    return dgen.generate_dataset()[0]


def evaluate(reset_lvl: int = 0) -> None:
    """Evaluate the dataset."""
    cfg = load_config()
    data_paths = build_dataset(cfg, reset_lvl)

    rag = build_rag()
    evaluation = RagEvaluation(cfg=cfg, rag=rag)
    evaluation.evaluate(data_paths, reset_lvl >= 1)


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.zero_count)
