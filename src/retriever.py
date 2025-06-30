import traceback

from loguru import logger
from pymilvus import AnnSearchRequest, RRFRanker, WeightedRanker

from src.mocks import DbConnector
from src.models import RetrievalConfig, RetrievedDocument


class HybridRetriever:
    """Hybrid search retriever for Milvus database."""

    def __init__(
        self, cfg: RetrievalConfig, db_connector: DbConnector
    ) -> None:
        """Initialize the Retriever object.

        Args:
            db_connector: Database connector instance.
            cfg: Configuration for the Milvus database.
        """
        self.cfg = cfg
        self.collection = db_connector.cfg.collection
        self.client = db_connector.client

    def __call__(
        self,
        question: str | list[str],
        top_k: int | None = None,
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
        top_k = top_k or self.cfg.top_k
        questions = [question] if isinstance(question, str) else question
        # if not self.client.has_collection_loaded(self.collection):
        #     self.client.load_collection(self.collection)

        try:
            results = self._retrieve(questions, top_k=top_k)
        except Exception as exc:
            logger.debug(traceback.format_exc())
            logger.error(f"Error retrieving documents: {exc}")

        return results

    def _retrieve(
        self,
        questions: list[str],
        top_k: int = 5,
    ) -> list[list[RetrievedDocument]]:
        question_search_param = {
            "data": questions,
            "anns_field": "question_vector",
            "param": {"nprobe": self.cfg.nprobe},
            "limit": top_k,
        }
        request_question = AnnSearchRequest(**question_search_param)
        answer_search_param = {
            "data": questions,
            "anns_field": "answer_vector",
            "param": {"nprobe": self.cfg.nprobe},
            "limit": top_k,
        }
        request_answer = AnnSearchRequest(**answer_search_param)
        ranker = (
            RRFRanker(k=self.cfg.rrf_k)
            if self.cfg.ranker == "rrf"
            else WeightedRanker(*self.cfg.weights)
        )
        search_results = self.client.hybrid_search(
            collection_name=self.collection,
            reqs=[request_question, request_answer],
            ranker=ranker,
            limit=top_k,
            output_fields=[
                "id",
                "title",
                "categoryId",
                "folderId",
                "chunk_id",
            ],
        )
        results: list[list[RetrievedDocument]] = []
        for hits in search_results:
            result = []
            for hit in hits:
                logger.debug(hit)
                result.append(RetrievedDocument.from_milvus_hit(hit))

            results.append(result)
        return results
