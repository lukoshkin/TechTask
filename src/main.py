import argparse

from dotenv import load_dotenv
from loguru import logger

from src.db import MilvusDB
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


def main() -> None:
    load_dotenv()

    args = parse_args()
    rag_cfg_file = "rag-config.yaml"
    drop_collection = args.zero_count >= 1
    drop_preprocessed = args.zero_count >= 2

    rag_cfg = RagConfig.from_yaml(rag_cfg_file)
    logger.debug(f"Loaded configuration: {rag_cfg}")

    preproc = DataProcessor(rag_cfg.preprocessing)
    output_path = preproc.output_path(rag_cfg.input_data)
    if drop_preprocessed and output_path.exists():
        output_path.unlink()

    if not output_path.exists():
        preproc(rag_cfg.input_data)

    db = MilvusDB(rag_cfg.database)
    if drop_collection:
        db.drop_collection()

    db.create_collection()
    if db.count() == 0:
        preproc.insert_data(db, str(output_path))

    retriever = HybridRetriever(rag_cfg.retrieval, db)
    rag = RagPipeline(rag_cfg, retriever)
    print(rag.answer(args.question))


if __name__ == "__main__":
    main()
