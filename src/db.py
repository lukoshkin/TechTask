"""MilvusDB Connector for TechTask Knowledge Base.

This module provides a connector to the Milvus database for managing a
knowledge base related to TechTask. It includes functionality to create a
collection, insert data, and manage vector fields for efficient querying.
"""

from collections.abc import Generator
from typing import Any, Literal

import numpy as np
import pandas as pd
from loguru import logger
from pymilvus import DataType, Function, FunctionType, MilvusClient

from src.mocks import DbConnector
from src.models import DatabaseConfig


def batch_generator(
    data: pd.DataFrame, batch_size: int
) -> Generator[np.ndarray, None, None]:
    """
    Generate batches of data from a numpy array.

    Args:
        data (np.ndarray): The input data array.
        batch_size (int): The size of each batch.

    Yields
    ------
        np.ndarray: A batch of data.
    """
    data = data.to_numpy()
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


class MilvusDB(DbConnector):
    """Milvus Connector for TechTask Knowledge Base."""

    def __init__(self, cfg: DatabaseConfig) -> None:
        """
        Initialize the MilvusDB class with connection parameters.

        Args:
            uri (str): The URI of the Milvus server.
            token (str): Authentication token for Milvus.
        """
        self.cfg = cfg
        self.client = MilvusClient(uri=cfg.uri, token=cfg.token)
        self._metadata_fields = [
            "id",
            "categoryId",
            "folderId",
            "lang",
            "title",
            "description_text",
            "chunk_id",
        ]

    def create_collection(self) -> None:
        """
        Create a collection for the TechTask knowledge base.

        The schema includes fields for metadata and two vector fields:
        - question_vector: For the title field (question)
        - answer_vector: For the description_text field (answer)
        """
        if self.client.has_collection(self.cfg.collection):
            logger.info(f"Collection '{self.cfg.collection}' already exists.")
            return

        schema = MilvusClient.create_schema(auto_id=False)
        schema.add_field(
            field_name="id",
            datatype=DataType.INT64,
            is_primary=True,
        )
        schema.add_field(
            field_name="categoryId",
            datatype=DataType.INT64,
        )
        schema.add_field(
            field_name="folderId",
            datatype=DataType.INT64,
        )
        schema.add_field(
            field_name="lang",
            datatype=DataType.VARCHAR,
            max_length=self.cfg.lang_length,
        )
        schema.add_field(
            field_name="title",
            datatype=DataType.VARCHAR,
            max_length=self.cfg.text_size,
            enable_analyzer=True,
        )
        schema.add_field(
            field_name="description_text",
            datatype=DataType.VARCHAR,
            max_length=self.cfg.text_size,
            enable_analyzer=True,
        )
        schema.add_field(
            field_name="chunk_id",  # Inferred field; thus, python snakecase
            datatype=DataType.INT64,
        )
        schema.add_field(
            field_name="question_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.cfg.vector_dim,
        )
        schema.add_field(
            field_name="answer_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.cfg.vector_dim,
        )
        # schema.add_field(
        #     field_name="answer_vector_sparse",
        #     datatype=DataType.SPARSE_FLOAT_VECTOR,
        # )
        # schema.add_function(
        #     Function(
        #         name="answer_bm25_emb",
        #         input_field_names=["description_text"],
        #         output_field_names=["answer_vector_sparse"],
        #         function_type=FunctionType.BM25,
        #     )
        # )
        schema.add_function(
            Function(
                name="title_embedding",
                function_type=FunctionType.TEXTEMBEDDING,
                input_field_names=["title"],
                output_field_names=["question_vector"],
                params={
                    "provider": "openai",
                    "model_name": "text-embedding-3-small",
                    # "credential": <alias_of_credential_in_milvus.yaml>
                    "dim": self.cfg.vector_dim,
                },
            )
        )
        schema.add_function(
            Function(
                name="description_embedding",
                function_type=FunctionType.TEXTEMBEDDING,
                input_field_names=["description_text"],
                output_field_names=["answer_vector"],
                params={
                    "provider": "openai",
                    "model_name": "text-embedding-3-small",
                    # "credential": <alias_of_credential_in_milvus.yaml>
                    "dim": self.cfg.vector_dim,
                },
            )
        )
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="question_vector",
            index_type="AUTOINDEX",
            metric_type="IP",
        )
        index_params.add_index(
            field_name="answer_vector",
            index_type="AUTOINDEX",
            metric_type="IP",
        )
        # index_params.add_index(
        #     field_name="answer_vector_sparse",
        #     index_type="SPARSE_INVERTED_INDEX",
        #     metric_type="BM25",
        #     params={"inverted_index_algo": "DAAT_MAXSCORE"},
        # )
        self.client.create_collection(
            collection_name=self.cfg.collection,
            schema=schema,
            index_params=index_params,
            consistency_level="Strong",
        )
        logger.info(f"Created collection: '{self.cfg.collection}'")

    def insert_data(
        self,
        df: pd.DataFrame,
        batch_size: int | None = None,
    ) -> None:
        """
        Insert data into the Milvus collection.

        Args:
            df (pandas.DataFrame): DataFrame with the knowledge base data
        """
        if batch_size is None:
            try:
                self.client.insert(
                    collection_name=self.cfg.collection,
                    data=df[self._metadata_fields].to_dict(orient="records"),
                )
                return
            except Exception as exc:
                logger.error(f"Failed to insert data: {exc}")
                msg = getattr(exc, "message", None)
                if msg:
                    row_number = msg.split("row number:")
                    if len(row_number) > 1:
                        row_number = row_number[1].split(",")[0].strip()
                        row = df[self._metadata_fields].iloc[int(row_number)]
                        logger.debug(f"Problematic row: {row.to_dict()}")

                raise exc

        gen = batch_generator(df[self._metadata_fields], batch_size)
        for idx, batch in enumerate(gen):
            self.client.insert(
                collection_name=self.cfg.collection,
                data=batch.to_dict(orient="records"),
            )
            logger.info(f"Inserted batch #{idx + 1}")

    def count(
        self,
        consistency_level: (
            Literal["Session", "Strong", "Bounded", "Eventual"]
        ) = "Strong",
    ) -> int:
        """
        Count the number of items in the collection.

        The consistency level is set to `register_consistency` by default.
        """
        return self.client.query(
            collection_name=self.cfg.collection,
            output_fields=["count(*)"],
            consistency_level=consistency_level,
        )[0]["count(*)"]

    def drop_collection(self, **kwargs: Any) -> None:
        """Drop the collection."""
        if not self.client.has_collection(self.cfg.collection):
            logger.warning(
                f"Collection '{self.cfg.collection}' cannot be removed"
                ", since not found."
            )
            return
        self.client.release_collection(self.cfg.collection, **kwargs)
        self.client.drop_collection(self.cfg.collection)
