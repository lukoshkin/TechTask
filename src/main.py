import argparse

from dotenv import load_dotenv

from src.db import MilvusDB
from src.models import MilvusConfig
from src.preproc import DataProcessor


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
    return parser.parse_args()


load_dotenv()

args = parse_args()
csv_file = "data/knowledge_base.csv"
drop_collection = args.zero_count >= 1
drop_preprocessed = args.zero_count >= 2

preproc = DataProcessor(chunk_overlap=0)
output_path = preproc.output_path(csv_file)
if drop_preprocessed and output_path.exists():
    output_path.unlink()

if not output_path.exists():
    preproc(csv_file)

cfg = MilvusConfig(
    collection="techtask_kb",
    vector_dim=768,
    text_chunk_size=1024,
    lang_length=5,
)
db = MilvusDB(cfg)
if drop_collection:
    db.drop_collection()

db.create_collection()
if db.count() == 0:
    preproc.insert_data(db, str(output_path))
