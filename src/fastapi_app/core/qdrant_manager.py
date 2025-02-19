"""Handles Qdrant vector database setup and operations."""

import gc
import re
from pathlib import Path

import numpy as np
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from summa import summarizer

from src.utils.general_utils import load_config, timeit


class QdrantManager:
    """Handles Qdrant vector database setup and operations."""

    def __init__(self):
        """Initializes Qdrant client and model configurations based on config.yaml."""
        project_root = Path(__file__).resolve().parents[3]
        self.config = load_config(
            "/".join([str(project_root), "config", "config.yaml"])
        )
        self.dataset_path = (
            project_root / "data" / self.config["dataset"]["dataset_filename"]
        )
        self.sample_size = self.config["dataset"]["sample_size"]

        self.client = QdrantClient(self.config["qdrant"]["host"])
        self.collection_name = self.config["qdrant"]["collection_name"]
        self.vector_size = self.config["qdrant"]["vector_size"]

        model_name = self.config["sentence_transformer"]["embedding_model"][
            "model_name"
        ]
        self.embedding_model = SentenceTransformer(model_name)

        self.distance_mapping = {
            "cosine": Distance.COSINE,
            "dot": Distance.DOT,
            "euclidean": Distance.EUCLID,
        }
        self.qdrant_distance = self.distance_mapping.get(
            self.config["qdrant"]["distance_metric"], Distance.COSINE
        )
        self.chunk_size = self.config["dataset"]["chunk_size"]
        self.chunk_overlap = self.config["dataset"]["chunk_overlap"]

        self._create_collection_if_not_exists()

    @timeit
    def ingest_data(self) -> None:
        """
        Ingests data into Qdrant by loading a dataset, processing the documents,
        and storing the embeddings.

        Steps:
        1. Loads dataset from `.parquet`
        2. Processes text into chunks
        3. Generates embeddings and stores them in Qdrant
        """
        try:
            df = self._load_dataset()
            processed_chunks, original_texts, metadata_list = self._process_documents(
                df
            )
            self._store_embeddings(processed_chunks, original_texts, metadata_list)
            print(f"Ingested {len(processed_chunks)} chunks into Qdrant.")
        except Exception as e:
            print(f"Error while ingesting data: {e}")

    @timeit
    def search_with_context(
        self, query_text: str, top_k: int = 3, max_tokens: int = 512
    ) -> str:
        """
        Searches for relevant documents in the Qdrant collection using the
        provided query text and returns a query with context.
        Args:
            query_text (str): The query text to search for.
            top_k (int, optional): The number of top results to retrieve.
            Defaults to 3.
            max_tokens (int, optional): The maximum number of tokens
            allowed in the context. Defaults to 512.
        Returns:
            str: A query string with the retrieved context or the
            original query text if no relevant documents are found.
        Raises:
            Exception: If an error occurs during the search process.
        """
        try:
            query_embedding = self.embedding_model.encode(
                query_text, convert_to_numpy=True
            )

            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True,
            )

            if not search_results:
                print("No relevant documents found. Sending query as-is.")
                return query_text

            retrieved_chunks = [
                hit.payload.get("chunk_text", "") for hit in search_results
            ]
            chunk_embeddings = self.embedding_model.encode(
                retrieved_chunks, convert_to_numpy=True
            )

            # #Calculate similarity scores between chunks and query
            similarities = np.dot(chunk_embeddings, query_embedding) / (
                np.linalg.norm(chunk_embeddings, axis=1)
                * np.linalg.norm(query_embedding)
            )

            # #Sort chunks based on similarity scores
            sorted_indices = np.argsort(similarities)[::-1]  # #Descending order
            ranked_chunks = [retrieved_chunks[i] for i in sorted_indices]

            all_sentences = []
            for chunk in ranked_chunks:
                sentences = chunk.split(". ")
                all_sentences.extend(sentences)

            sentence_embeddings = self.embedding_model.encode(
                all_sentences, convert_to_numpy=True
            )

            relevance_scores = np.dot(sentence_embeddings, query_embedding) / (
                np.linalg.norm(sentence_embeddings, axis=1)
                * np.linalg.norm(query_embedding)
            )

            # #Filter sentences based on relevance threshold
            filtered_sentences = [
                sentence
                for sentence, score in zip(all_sentences, relevance_scores)
                if score > 0.4
            ]

            # #Limit the total token count
            context = " ".join(filtered_sentences)
            if len(context.split()) > max_tokens:
                print("Summarizing retrieved context to fit within token limits...")
                context = summarizer.summarize(
                    context, words=100
                )  # #Summarize to 100 words

            query = f"""
            Context: {context}

            User Query: {query_text}

            Based on the above context, provide a helpful response.
            """

            del (
                query_embedding,
                search_results,
                retrieved_chunks,
                chunk_embeddings,
                similarities,
            )
            del (
                ranked_chunks,
                all_sentences,
                sentence_embeddings,
                relevance_scores,
                filtered_sentences,
            )
            gc.collect()
            return query

        except Exception as e:
            print(f"Error while searching Qdrant: {e}")
            return query_text

    @timeit
    def search(self, query_text: str, top_k: int = 3) -> list:
        """
        Searches Qdrant for the top-k most similar documents based on query text.
        Args:
            query_text (str): The user's input query.
            top_k (int, optional): Number of results to return. Defaults to 3.
        Returns:
            list: A list of retrieved documents with metadata.
        """
        try:
            query_embedding = self.embedding_model.encode([query_text])[0]

            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True,
            )

            results = []
            for hit in search_results:
                results.append(
                    {
                        "score": hit.score,
                        "chunk_text": hit.payload.get("chunk_text", "N/A"),
                        "full_text": hit.payload.get("full_text", "N/A"),
                        "prompt": hit.payload.get("prompt", "N/A"),
                        "seed_data": hit.payload.get("seed_data", "N/A"),
                        "format": hit.payload.get("format", "N/A"),
                        "audience": hit.payload.get("audience", "N/A"),
                    }
                )
            return results
        except Exception as e:
            print(f"Error while searching Qdrant: {e}")
            return []

    def _create_collection_if_not_exists(self) -> None:
        """
        Ensures that a collection exists in the Qdrant database.
        If the collection does not exist, it creates a new one.
        This method checks if a collection with the specified name
        exists in the Qdrant database. If the collection is not found,
        it creates a new collection with the given configuration parameters.
        If the collection already exists, it proceeds without
        creating a new one.

        Raises:
            Exception: If there is an error while checking for or creating the collection.
        Returns:
            None
        """
        try:
            existing_collections = self.client.get_collections()
            collection_names = [col.name for col in existing_collections.collections]

            if self.collection_name not in collection_names:
                print(f"Collection '{self.collection_name}' not found. Creating...")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size, distance=self.qdrant_distance
                    ),
                )
                print(f"Collection '{self.collection_name}' created successfully.")
            else:
                print(f"Collection '{self.collection_name}' already exists.")
        except Exception as e:
            print(f"Error while checking or creating the collection: {e}")

    def _preprocess_text(self, text: str) -> str:
        """Cleans and normalizes text."""
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\.{2,}", ".", text)  # Remove multiple dots
        text = re.sub(r"[^a-zA-Z0-9.,!? ]+", "", text)  # Remove non-ASCII chars
        return text

    def _chunk_text(self, text: str) -> list:
        """
        Splits the given text into chunks using RecursiveCharacterTextSplitter.
        Args:
            text (str): The text to be split into chunks.
        Returns:
            list: A list of text chunks.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        return splitter.split_text(text)

    def _load_dataset(self) -> pd.DataFrame:
        """
        Loads and samples the dataset from a Parquet file.

        Returns:
            DataFrame containing the loaded dataset.
        """
        try:
            if not self.dataset_path.exists():
                print(f"Error: Dataset file '{self.dataset_path}' was not found.")
                return pd.DataFrame()

            df = pd.read_parquet(self.dataset_path)
            if df.empty:
                print(f"Warning: The dataset '{self.dataset_path}' is empty.")
                return pd.DataFrame()

            df_sample = df.sample(n=self.sample_size, random_state=42)
            return df_sample
        except Exception as e:
            print(f"Error while loading dataset: {e}")
            return pd.DataFrame()

    def _process_documents(self, df: pd.DataFrame) -> tuple:
        """
        Processes the documents in the given DataFrame by preprocessing and chunking the text,
        and extracting relevant metadata.
        Args:
            df (pd.DataFrame): A DataFrame containing the documents to be processed. Each row
                               should have the columns 'text', 'prompt', 'seed_data', 'format',
                               and 'audience'.
        Returns:
            tuple: A tuple containing three lists:
                - processed_chunks (list): A list of text chunks obtained after preprocessing
                                           and chunking the text.
                - original_texts (list): A list of the original full document texts.
                - metadata_list (list): A list of dictionaries containing metadata for each
                                        text chunk, with keys 'prompt', 'seed_data', 'format',
                                        and 'audience'.
        """
        processed_chunks, original_texts, metadata_list = [], [], []

        for _, row in df.iterrows():
            text = self._preprocess_text(row["text"])
            chunks = self._chunk_text(text)

            for chunk in chunks:
                processed_chunks.append(chunk)
                original_texts.append(row["text"])  # Store full document text
                metadata_list.append(
                    {
                        "prompt": row["prompt"],
                        "seed_data": row["seed_data"],
                        "format": row["format"],
                        "audience": row["audience"],
                    }
                )

        return processed_chunks, original_texts, metadata_list

    def _store_embeddings(
        self, processed_chunks: list, original_texts: list, metadata_list: list
    ) -> None:
        """Generates embeddings in smaller batches and stores them in Qdrant."""
        batch_size = 32  # Reduce batch size to avoid timeouts

        for i in range(0, len(processed_chunks), batch_size):
            batch_chunks = processed_chunks[i: i + batch_size]
            batch_texts = original_texts[i: i + batch_size]
            batch_metadata = metadata_list[i: i + batch_size]

            embeddings = self.embedding_model.encode(
                batch_chunks, convert_to_numpy=True
            )

            points = [
                PointStruct(
                    id=i + idx,
                    vector=embeddings[idx],
                    payload={
                        "chunk_text": batch_chunks[idx],
                        "full_text": batch_texts[idx],
                        "prompt": batch_metadata[idx]["prompt"],
                        "seed_data": batch_metadata[idx]["seed_data"],
                        "format": batch_metadata[idx]["format"],
                        "audience": batch_metadata[idx]["audience"],
                    },
                )
                for idx in range(len(batch_chunks))
            ]

            self.client.upsert(self.collection_name, points)
            print(
                f"Stored {len(batch_chunks)} chunks in Qdrant (Batch {i // batch_size + 1})"
            )

        print(f"Successfully stored {len(processed_chunks)} chunks in Qdrant.")
