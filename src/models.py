from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class RetrievedDocument(BaseModel):
    """Model representing a retrieved document."""

    id: int
    distance: float
    title: str
    category: int
    folder: int
    chunk: int

    @classmethod
    def from_milvus_hit(cls, hit: dict) -> "RetrievedDocument":
        """Create a RetrievedDocument from a Milvus hit."""
        return RetrievedDocument(
            id=hit["id"],
            distance=hit["distance"],
            title=hit["entity"]["title"],
            category=hit["entity"]["categoryId"],
            folder=hit["entity"]["folderId"],
            chunk=hit["entity"]["chunk_id"],
        )


class PreprocessingConfig(BaseModel):
    """Configuration for preprocessing the knowledge base data."""

    text_chunk_size: int = 512
    text_chunk_overlap: int = 64
    data_dir: str = "data"
    processed_data: str = "processed_knowledge_base.csv"
    html_dir: str = "html_content"
    save_html_txt: bool = True


class DatabaseConfig(BaseModel):
    """Configuration for the database."""

    uri: str = "http://localhost:19530"
    token: str | None = None

    collection: str = "techtask"
    lang_length: int = 5
    text_size: int = 1024
    vector_dim: int = 768


class RetrievalConfig(BaseModel):
    """Configuration for the retrieval pipeline."""

    ranker: str = "rrf"
    weights: list[float] = [0.5, 0.5]
    top_k: int = 5
    nprobe: int = 10
    rrf_k: int = 60


class LLMConfig(BaseModel):
    """Configuration for the LLM model."""

    model: str = "gpt-3.5-turbo"
    temperature: float = 0.3
    max_tokens: int = 2048


class RagConfig(BaseModel):
    """Configuration for the RAG pipeline."""

    input_data: str
    llm: LLMConfig = Field(default_factory=LLMConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    preprocessing: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig
    )

    @classmethod
    def from_yaml(cls, file_path: str | Path) -> "RagConfig":
        """Load configuration from YAML file.

        Args:
            file_path: Path to the YAML file.

        Returns
        -------
            RagConfig: Configuration object.
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(
                f"Configuration file {file_path} not found."
            )
        with open(file_path, encoding="utf-8") as fd:
            config_data = yaml.safe_load(fd)
            return cls.model_validate(config_data)
