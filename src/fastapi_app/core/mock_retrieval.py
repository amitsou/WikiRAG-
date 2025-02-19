import random


class MockRetriever:
    """A mock retrieval function to simulate similarity search."""

    def __init__(self):
        """Initialize the mock retriever with some fake documents."""
        self.documents = [
            "This is a test document about AI and machine learning.",
            "Deep learning is a subset of machine learning focused on neural networks.",
            "Qdrant is a vector database used for similarity search and RAG applications.",
            "The goal of retrieval-augmented generation is to enhance LLM responses with external knowledge.",
            "Embeddings help transform textual data into numerical vectors for similarity search.",
        ]

    def retrieve(self, query: str, top_k: int = 3) -> list:
        """
        Simulate retrieving relevant documents based on a query.

        Args:
            query (str): The user query.
            top_k (int): The number of results to return.

        Returns:
            list: A list of mock document results.
        """
        random.shuffle(self.documents)
        return self.documents[:top_k]  # Return a subset of fake results
