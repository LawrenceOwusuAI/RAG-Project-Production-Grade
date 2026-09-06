from __future__ import annotations
import os
import time
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from chunking import PDF_PATH
from chunking import run_chunking_pipeline
from exception import CustomException
from logger import logging
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_NAME = "rag-project"
NAMESPACE = PDF_PATH.stem


def get_pinecone_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY was not found in the .env file.")
    return api_key

def create_embedding_model():
    return HuggingFaceEmbeddings(
        model_name = EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={ "normalize_embeddings": True}
      )


def get_embedding_dimension(embeddings) -> int:
    test_embedding = embeddings.embed_query(
            "embedding dimension test")
    return len(test_embedding)

def prepare_documents_for_pinecone(langchain_documents:list[Document]) -> list[Document]:
    pinecone_documents: list[Document] = []
    for document in langchain_documents:
        metadata = document.metadata.copy()
        metadata["page_numbers"] = [
            str(page_number) for page_number in metadata.get("page_numbers",[])]

        metadata["headings"] = [
            str(heading) for heading in metadata.get("headings",[])]

        metadata["captions"] = [
            str(caption) for caption in metadata.get("captions",[])]

        pinecone_documents.append(Document(
                page_content = document.page_content,
                metadata=metadata))
    return pinecone_documents

def create_pinecone_client():
    api_key = get_pinecone_api_key()
    return Pinecone(
        api_key=api_key
    )

def create_or_get_index(pc, embedding_dimension: int,):
    if not pc.has_index(INDEX_NAME):
        print(f"Creating Pinecone index: " f"{INDEX_NAME}")
        pc.create_index(name=INDEX_NAME,dimension=embedding_dimension,
                 metric="cosine",
                 spec=ServerlessSpec( cloud="aws",region="us-east-1")
                  )
    while True:
        description = pc.describe_index(name=INDEX_NAME)
        if description.status["ready"]:
            break
        print("Waiting for Pinecone index to be ready")
        time.sleep(2)

    print(f"Pinecone index ready: "f"{INDEX_NAME}")
    return pc.Index(INDEX_NAME)

def create_vector_store(index, embeddings):
    return PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=NAMESPACE
    )

def store_documents(vector_store,documents:list[Document]) -> None:
    chunk_ids = [document.metadata["chunk_id"] for document in documents]
    vector_store.add_documents(documents=documents,ids=chunk_ids)
    print(
        f"\nSuccessfully embedded "
        f"and stored "

        f"{len(documents)} "
        f"chunks in Pinecone."
    )

def run_vectorstore_pipeline(langchain_documents:list[Document]):
    print("\nCreating embedding model...")
    embeddings = create_embedding_model()
    dimension = get_embedding_dimension(embeddings)
    print(f"Embedding dimension: {dimension}")
    pinecone_documents = prepare_documents_for_pinecone(langchain_documents)
    pc = create_pinecone_client()
    index = create_or_get_index(pc=pc,embedding_dimension=dimension)
    vector_store = create_vector_store(index=index,embeddings=embeddings)
    store_documents(vector_store=vector_store,documents=pinecone_documents)
    stats = index.describe_index_stats()
    print("\nPinecone index statistics:")
    print(stats)
    return vector_store

if __name__ == "__main__":
    logging.info("Chunking of document has started")
    try:
        chunkedDoclingDocument = run_chunking_pipeline()  
    except Exception as e:
        logging.exception("Document chucking failed")
        raise CustomException(e,sys)
    logging.info("Chucking has succeeded")

    logging.info("Upserting of embeddings in Pinecone has started")
    try:
        run_vectorstore_pipeline(chunkedDoclingDocument)
          
    except Exception as e:
        logging.exception("Document upsert into pinecone failed")
        raise CustomException(e,sys)
    logging.info("Upserting of embeddings in Pinecone has succeeded")

    