"""
Synthetic data generation module for RAG pipeline evaluation.

This module provides tools for generating synthetic test data using Ragas library
to evaluate the RAG pipeline created for the knowledge base.
"""

import asyncio
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from loguru import logger
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.synthesizers import default_query_distribution
from ragas.testset.transforms import (
    # KeyphrasesExtractor,
    # Parallel,
    apply_transforms,
    default_transforms,
)

# from ragas.testset.transforms.extractors import NERExtractor
# from ragas.testset.transforms.relationship_builders.traditional import (
#     JaccardSimilarityBuilder,
# )
from src.models import DataGenConfig


class KGGenerationInputError(ValueError):
    """Custom exception for errors in knowledge graph generation input."""


class SyntheticDataset:
    """Interface for generating synthetic test data for RAG evaluation.

    This class uses the Ragas library to generate synthetic test data based on
    the knowledge base provided. It can be configured through the TestConfig.
    """

    def __init__(self, cfg: DataGenConfig):
        """Initialize the SyntheticDataset.

        Args:
            config: Configuration for the RAG pipeline.
        """
        self.cfg = cfg
        self.langs: list[str]
        self.output_dir = Path(cfg.output_dir) / cfg.synthetic_dir
        self.interim_dir = self.output_dir / "interim"
        self.interim_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.output_dir / self.cfg.output_filename

        self.llm = LangchainLLMWrapper(ChatOpenAI(model=self.cfg.llm_model))
        self.embedding_model = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model=self.cfg.embedding_model,
                dimensions=self.cfg.embedding_dimension,
            )
        )

    def sort_by_lang(self, source_csv_path: str) -> None:
        """Sort a small number of documents by language.

        Args:
            source_csv_path: Path to the source CSV data.
        """
        df = pd.read_csv(source_csv_path)
        if "lang" not in df.columns:
            raise ValueError("CSV file must contain 'lang' column.")

        self.langs = df["lang"].unique().tolist()
        for lang in self.langs:
            subsample = df[df["lang"] == lang].sample(
                n=min(self.cfg.lang_cardinality, len(df[df["lang"] == lang])),
                random_state=self.cfg.random_state,
            )
            if len(subsample) < self.cfg.lang_cardinality:
                logger.warning(
                    f"Not enough samples for language '{lang}'. Requested"
                    f" {self.cfg.lang_cardinality}, got {len(subsample)}."
                )
            subsample.to_csv(
                self.interim_dir / f"{lang}-{len(subsample)}.csv",
                index=False,
            )

    def _langchain_documents(self, csv_path: str | Path) -> list[Document]:
        """Load documents from a CSV file into Langchain format.

        Args:
            csv_path: Path to the CSV file containing documents.

        Returns
        -------
            list[dict]: List of documents in Langchain format.
        """
        return CSVLoader(
            file_path=str(csv_path),
            source_column="description_text",
            metadata_columns=["title", "lang"],
        ).load()

    def knowledge_graph(
        self,
        docs: list[Document] | None = None,
        save_kg_at: str | Path | None = None,
        ignore_input_if_kg_exists: bool = False,
    ) -> KnowledgeGraph:
        """Generate or load a knowledge graph from documents.

        Args:
            docs: List of documents to build the knowledge graph from.
            ignore_input_if_kg_exists: If True, ignore input documents if a
                knowledge graph already exists.
        """
        if docs is None or ignore_input_if_kg_exists:
            if save_kg_at and Path(save_kg_at).exists():
                return KnowledgeGraph.load(save_kg_at)

            if not docs:
                raise KGGenerationInputError(
                    "No documents provided and no knowledge graph found."
                )
        kg = KnowledgeGraph()
        for doc in docs:
            kg.nodes.append(
                Node(
                    type=NodeType.DOCUMENT,
                    properties={
                        "page_content": doc.page_content,
                        "document_metadata": doc.metadata,
                    },
                )
            )
        # Among default_transforms, there are a lot of them, actually:
        # SummaryExtractor,
        # CustomNodeFilter,
        # EmbeddingExtractor,
        # ThemeExtractor,
        # NERExtractor,
        # and etc.
        transforms = default_transforms(
            documents=docs,
            llm=self.llm,
            embedding_model=self.embedding_model,
        )
        # transforms = [
        #     Parallel(KeyphrasesExtractor(), NERExtractor()),
        #     JaccardSimilarityBuilder(
        #         property_name="entities",
        #         key_name="PER",
        #         new_property_name="entity_jaccard_similarity",
        #     ),
        # ]
        apply_transforms(kg, transforms)
        if save_kg_at:
            kg.save(save_kg_at)
        return kg

    def adapt_queries(self, lang: str) -> Any:
        """Adapt queries for the specified language.

        Args:
            lang: Language code to adapt queries for.

        Returns
        -------
            distribution: Adapted query prompts.
        """
        distribution = default_query_distribution(self.llm)
        for query, _ in distribution:
            prompts = asyncio.run(query.adapt_prompts(lang, llm=self.llm))
            query.set_prompts(**prompts)
        return distribution

    def generate_dataset(
        self, ignore_previous_generation: bool = False
    ) -> pd.DataFrame | None:
        """Generate synthetic test dataset.

        Args:
            ignore_previous_generation: If True, ignore any previously
                generated datasets and regenerate them.

        Returns
        -------
            pd.DataFrame: Generated synthetic dataset if any documents were
                processed. Otherwise, None.

        """
        frames = []
        for csv_path in self.interim_dir.glob("*.csv"):
            logger.info(f"Working on CSV file: {csv_path}")
            lang = csv_path.stem.rsplit("-")[0]
            dset_path = self.output_dir / f"dset-{lang}.csv"
            if not ignore_previous_generation and dset_path.exists():
                logger.warning(
                    f"Dataset for language '{lang}' already exists. "
                    "Skipping regeneration."
                )
                continue

            docs = self._langchain_documents(csv_path)
            kg = self.knowledge_graph(
                docs,
                csv_path.parent / f"kg-{csv_path.stem}.json",
                ignore_input_if_kg_exists=True,
            )
            generator = TestsetGenerator(
                llm=self.llm,
                embedding_model=self.embedding_model,
                knowledge_graph=kg,
            )
            df = generator.generate(
                testset_size=self.cfg.lang_cardinality,
                query_distribution=self.adapt_queries(lang),
            ).to_pandas()
            df.to_csv(dset_path, index=False)
            frames.append(df)

        if frames:
            df_all = pd.concat(frames)
            df_all.to_csv(self.output_path, index=False)
            return df_all
        return None
