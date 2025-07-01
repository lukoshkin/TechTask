"""
Data preprocessing module for TechTask knowledge base.

This module provides functionality to prepare the data for embedding into
MilvusDB. It handles splitting long text into chunks and organizing HTML
content.
"""

from pathlib import Path

import pandas as pd
from loguru import logger

from src.mocks import DbConnector
from src.models import PreprocessingConfig
from src.text_splitter import RecursiveCharacterTextSplitter


class DataProcessor:
    """TechTask data processor."""

    def __init__(self, cfg: PreprocessingConfig):
        """
        Initialize the DataPreparator for processing knowledge base data.

        Args:
            cfg: Configuration for the database connection and processing
        """
        self.cfg = cfg
        self._save_also_txt = cfg.save_html_txt
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.text_chunk_size,
            chunk_overlap=cfg.text_chunk_overlap // 2,
            separators=["\n\n", "\n", r"\s{3,}", "  ", " "],
            strip_whitespace=True,
        )
        self.tsplit_map = {
            "ja": RecursiveCharacterTextSplitter(
                chunk_size=cfg.text_chunk_size // 2,
                chunk_overlap=cfg.text_chunk_overlap // 2,
                separators=["\n", r"\s{3,}", "。", "!", "?", "、", " "],
                strip_whitespace=True,
            )
        }
        self.required_fields = [
            "id",
            "lang",
            "folderId",
            "categoryId",
            "title",
            "description_text",
        ]
        self._stats: dict[str, dict[str, int]] = {
            "max_lengths": {},
            "max_display_widths": {},
        }

    def output_path(self, csv_path: str) -> Path:
        """Get the output path for the processed CSV file."""
        output_path = Path(self.cfg.data_dir or Path(csv_path).parent)
        return output_path / self.cfg.processed_data

    def __call__(self, csv_path: str, chunksize: int = 512) -> None:
        """
        Process data from CSV file for embedding into MilvusDB.

        Reads in chunks to mitigate the compute burden.

        Args:
            csv_path: Path to the CSV file containing the knowledge base data
            chunksize: Number of rows to process at once
        """
        self._stats = {
            "max_lengths": {},
            "max_display_widths": {},
        }
        logger.info(f"Reading data from {csv_path} in chunks of {chunksize}")
        output_csv_path = self.output_path(csv_path)
        output_path = output_csv_path.parent
        output_path.mkdir(exist_ok=True)
        html_path = output_path / self.cfg.html_dir
        html_path.mkdir(exist_ok=True)

        if output_csv_path.exists():
            output_csv_path.unlink()

        total_rows = 0
        for chunk_idx, chunk in enumerate(
            pd.read_csv(csv_path, chunksize=chunksize)
        ):
            rows_processed = self.process_chunk(
                chunk,
                html_path,
                output_csv_path,
                write_header=chunk_idx == 0,
            )
            total_rows += rows_processed
            logger.info(f"Processed chunk {chunk_idx + 1}")

        self._log_column_stats()
        logger.info(f"Total processed rows: {total_rows}")
        logger.info(f"Processed data saved to {output_csv_path}")

    def _save_html_content(self, row: pd.Series, html_dir: Path) -> None:
        """
        Save HTML content from a row to a file.

        Args:
            row: DataFrame row containing HTML content
            html_path: Base path for HTML files
        """
        folder_path = html_dir / str(row["folderId"])
        folder_path.mkdir(exist_ok=True)
        html_path = folder_path / f"{row['id']}.html"
        with open(html_path, "w", encoding="utf-8") as fd:
            fd.write(row["description"])

        if self._save_also_txt:
            txt_path = html_path.with_suffix(".txt")
            with open(txt_path, "w", encoding="utf-8") as fd:
                fd.write(row["description"])

    def process_chunk(
        self,
        df_chunk: pd.DataFrame,
        html_path: Path,
        output_csv_path: Path,
        write_header: bool = False,
    ) -> int:
        """
        Process a chunk of the DataFrame and write to output CSV incrementally.

        Args:
            df_chunk: Chunk of DataFrame to process
            html_path: Path to save HTML content
            output_csv_path: Path to the output CSV file
            write_header: Whether to write the header row (True for first chunk)

        Returns
        -------
            Number of processed rows written to the output
        """
        all_rows = []
        for _, row in df_chunk.iterrows():
            self._save_html_content(row, html_path)
            text = row["description_text"]
            text_utf8 = text.encode("utf-8", errors="ignore")
            if len(text_utf8) > self.cfg.text_chunk_size:
                tsplit = self.tsplit_map.get(row["lang"], self.text_splitter)
                chunks = tsplit.split_text(text)
                for idx, chunk in enumerate(chunks):
                    chunk_utf8 = chunk.encode("utf-8", errors="ignore")
                    if len(chunk_utf8) > self.cfg.text_chunk_size:
                        logger.warning(
                            "Chunk exceeds maximum size.\n"
                            f"Text: {chunk}"
                            "\n---------------\n"
                            f"Original text length: {len(text)},\n"
                            f"Chunk length: {len(chunk)},\n"
                            f"Chunk byte length: {len(chunk_utf8)}."
                        )
                    new_row = {
                        field: row[field]
                        for field in self.required_fields
                        if field != "description_text"
                    }
                    new_row["description_text"] = chunk
                    new_row["chunk_id"] = idx
                    all_rows.append(new_row)
            else:
                new_row = {field: row[field] for field in self.required_fields}
                new_row["chunk_id"] = 0
                all_rows.append(new_row)

        processed_df = pd.DataFrame(all_rows)
        self._update_column_stats(processed_df)

        processed_df.to_csv(
            output_csv_path,
            mode="a",
            header=write_header,
            index=False,
        )
        return len(processed_df)

    def _update_column_stats(self, df: pd.DataFrame) -> None:
        """
        Update the maximum lengths and display widths.

        Args:
            df: DataFrame to extract string lengths and display widths from
        """
        for column in df.columns:
            if df[column].dtype == "object":
                try:
                    self._stats["max_lengths"][column] = max(
                        self._stats["max_lengths"].get(column, 0),
                        df[column].astype(str).apply(len).max(),
                    )
                    self._stats["max_display_widths"][column] = max(
                        self._stats["max_display_widths"].get(column, 0),
                        df[column]
                        .astype(str)
                        .apply(
                            lambda x: len(x.encode("utf-8", errors="ignore"))
                        )
                        .max(),
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to calculate string metrics.\n"
                        f"Column {column}: {exc}"
                    )

    def _log_column_stats(self) -> None:
        """Log stats about maximum string lengths and display widths."""
        logger.info("Column length statistics:")
        for column, max_len in sorted(self._stats["max_lengths"].items()):
            max_dwidth = self._stats["max_display_widths"].get(column, 0)
            logger.info(
                f"  {column}: max length = {max_len},"
                f" max display width = {max_dwidth}"
            )

    @staticmethod
    def insert_data(
        db: DbConnector, csv_path: str, chunksize: int = 512
    ) -> None:
        """
        Insert data from a CSV file into the database.

        Args:
            db: Database connector instance
            csv_path: Path to the CSV file containing the knowledge base data
            chunksize: Number of rows to process at once
        """
        for idx, chunk in enumerate(
            pd.read_csv(csv_path, chunksize=chunksize)
        ):
            db.insert_data(chunk)
            logger.info(f"Inserted chunk #{idx + 1} into the database")
