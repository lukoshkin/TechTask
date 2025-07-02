"""Retrieval-Augmented Generation module.

This module provides a RAG (Retrieval-Augmented Generation) class that
extends the MilvusDB connector with methods for retrieving relevant documents
and generating answers using a language model.
"""

from pathlib import Path

import litellm
from bs4 import BeautifulSoup
from loguru import logger

from src.mocks import Retriever
from src.models import RagConfig
from src.prompt import SYSTEM_PROMPT, USER_PROMPT


class RagPipeline:
    """RAG class for retrieving documents and generating answers."""

    def __init__(self, cfg: RagConfig, retriever: Retriever) -> None:
        """Initialize the RAG class.

        Args:
            *args: Variable length argument list passed to MilvusDB.
            **kwargs: Arbitrary keyword arguments passed to MilvusDB.
        """
        self.cfg = cfg
        self.retriever = retriever

    def answer(
        self,
        question: str,
        retrieval_top_k: int | None = None,
        temperature: float | None = None,
        return_retrieved_context: bool = False,
    ) -> str | tuple[str, list[str]]:
        """Generate a source-referenced answer to a question.

        Args:
            question: The question to answer.
            retrieval_top_k: Number of documents to retrieve.
            temperature: Temperature for the LLM response.

        Returns
        -------
            HTML answer with expandable sections for retrieved documents.
        """
        retrieved_docs = self.retriever(
            question,
            top_k=retrieval_top_k or self.cfg.retrieval.top_k,
        )
        if not retrieved_docs:
            logger.warning("No documents retrieved to answer the question")
            return (
                "<p>I couldn't find any relevant information"
                " to answer your question.</p>"
            )
        context = []
        html_sections = []
        checked_docs: set[int] = set()
        for idx, doc in enumerate(retrieved_docs[0], 1):
            if doc.id in checked_docs:
                continue

            checked_docs.add(doc.id)
            html_content = self._load_html_content(doc.folder, doc.id)
            document_text = self._load_html_content(
                doc.folder, doc.id, as_plain_text=True
            )
            context.append(f"Document {idx}:\n{document_text}")
            html_section = f"""
            <details>
                <summary>Document {doc.id}: {doc.title}</summary>
                <div class="document-content">
                    {html_content}
                </div>
            </details>
            """
            html_sections.append(html_section)

        response = litellm.completion(
            model=self.cfg.llm.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_PROMPT.format(
                        context="\n\n".join(context), user_query=question
                    ),
                },
            ],
            temperature=temperature or self.cfg.llm.temperature,
            max_tokens=self.cfg.llm.max_tokens,
        )
        answer = f"""<div class="rag-answer">
<div class="answer-text">
    <p>{response.choices[0].message.content}</p>
</div>
<div class="source-documents">
    <h4>Source Documents:</h4>
    {"".join(html_sections)}
</div>
</div>"""
        return (answer, context) if return_retrieved_context else answer

    def _load_html_content(
        self,
        folder_id: int,
        file_id: int,
        as_plain_text: bool = False,
    ) -> str:
        """Load HTML content from a file based on folder ID.

        Args:
            folder_id: The folder ID to locate the HTML file.

        Returns
        -------
            The HTML content as a string
        """
        html_path = (
            Path(self.cfg.preprocessing.data_dir)
            / self.cfg.preprocessing.html_dir
            / str(folder_id)
            / f"{file_id}.html"
        )
        txt_path = html_path.with_suffix(".txt")
        extract_text = False
        if as_plain_text:
            if txt_path.exists():
                html_path = txt_path
            else:
                extract_text = True
        try:
            with open(html_path, encoding="utf-8") as html_file:
                content = html_file.read()
                if extract_text:
                    soup = BeautifulSoup(content, "html.parser")
                    return soup.get_text(separator="\n", strip=True)

                return content
        except Exception as exc:
            logger.error(f"Error loading the content from {html_path}")
            raise exc
