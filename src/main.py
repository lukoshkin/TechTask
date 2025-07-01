import argparse

from dotenv import load_dotenv
from loguru import logger

from src.db import MilvusDB
from src.mocks import DbConnector
from src.models import RagConfig
from src.preproc import DataProcessor
from src.rag import RagPipeline
from src.retriever import HybridRetriever


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns
    -------
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Process knowledge base data and manage Milvus collection."
    )
    parser.add_argument(
        "-0",
        dest="zero_count",
        action="count",
        default=0,
        help=(
            "Use -0 to drop collection,"
            " -00 to drop collection and preprocessed data"
        ),
    )
    parser.add_argument(
        "question",
        type=str,
        help="The question to ask the RAg pipeline",
    )
    return parser.parse_args()


def load_config() -> RagConfig:
    """Load the RAG configuration from 'rag-config.yaml'."""
    load_dotenv()

    rag_cfg_file = "rag-config.yaml"
    rag_cfg = RagConfig.from_yaml(rag_cfg_file)
    logger.debug(f"Loaded configuration: {rag_cfg}")
    return rag_cfg


def build_db(rag_cfg: RagConfig, reset_lvl: int = 0) -> DbConnector:
    """Preprocess data, build DB, ingest data."""
    preproc = DataProcessor(rag_cfg.preprocessing)
    output_path = preproc.output_path(rag_cfg.input_data)
    if reset_lvl >= 2 and output_path.exists():
        output_path.unlink()

    if not output_path.exists():
        preproc(rag_cfg.input_data)

    import sys
    sys.exit()

    db = MilvusDB(rag_cfg.database)
    if reset_lvl >= 1:
        db.drop_collection()

    db.create_collection()
    if db.count() == 0:
        preproc.insert_data(db, str(output_path))

    return db


def build_rag(reset_lvl: int) -> RagPipeline:
    """Build the RAG pipeline."""
    cfg = load_config()
    db = build_db(cfg, reset_lvl)
    retriever = HybridRetriever(cfg.retrieval, db)
    return RagPipeline(cfg, retriever)


if __name__ == "__main__":
    args = parse_args()
    rag = build_rag(args.zero_count)
    print(rag.answer(args.question))
