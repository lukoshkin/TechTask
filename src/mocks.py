from typing import Any, Protocol

import pandas as pd

from src.models import DatabaseConfig, RetrievedDocument


class DbConnector(Protocol):
    """Mock-interface for database connectors."""

    client: Any
    cfg: DatabaseConfig

    def insert_data(self, df: pd.DataFrame) -> None:
        """
        Insert data into the database.

        Args:
            data: DataFrame containing the data to be inserted
        """


class Retriever(Protocol):
    """Retriever protocol."""

    def __init__(self, db_connector: DbConnector) -> None:
        """Initialize the retriever object."""

    def __call__(
        self,
        question: str | list[str],
        top_k: int = 5,
    ) -> list[list[RetrievedDocument]]:
        """Retrieve relevant documents using hybrid search.

        This method performs a hybrid search using both question and answer
        vectors to retrieve the most relevant documents for a given question.

        Args:
            question: A question or list of questions to search for.
            top_k: Maximum number of documents to retrieve.

        Returns
        -------
            List of dicts containing retrieved documents and their metadata.
        """


class RagPipeline(Protocol):
    """Mock interface for the RAG pipeline."""

    def answer(
        self,
        question: str,
        retrieval_top_k: int | None = None,
        temperature: float | None = None,
        return_retrieved_context: bool = False,
    ) -> tuple[str, list[str]]:
        """Generate a source-referenced answer to a question.

        Args:
            question: The question to answer.
            retrieval_top_k: Number of documents to retrieve.
            temperature: Temperature for the LLM response.

        Returns
        -------
            HTML answer with expandable sections for retrieved documents.
        """
