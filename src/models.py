from pydantic import BaseModel


class MilvusConfig(BaseModel):
    """Configuration for connecting to a Milvus database."""

    collection: str
    uri: str = "http://localhost:19530"
    token: str | None = None

    text_chunk_size: int = 512
    lang_length: int = 5
    vector_dim: int = 768
