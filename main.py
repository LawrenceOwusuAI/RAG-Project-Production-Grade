from logger import logging
from exception import CustomException
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_core.prompts import ChatPromptTemplate
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
from reranker import create_advanced_retriever
from chunking import PDF_PATH

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_NAME = "rag-project"
NAMESPACE = PDF_PATH.stem

def main():
    load_dotenv(override=True)
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY was not found.")
    logging.info("Pinecone API key loaded successfully")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY was not found.")
    logging.info("OpenAI API key loaded successfully")

    logging.info("Loading embedding model")
    embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,  
            model_kwargs={
                "device": "cpu"},
            encode_kwargs={"normalize_embeddings": True})
    logging.info("Embedding model loaded successfully")
    logging.info("Connecting to Pinecone")
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(INDEX_NAME)
    logging.info("Connected to Pinecone index: %s",INDEX_NAME)
    stats = index.describe_index_stats()
    logging.info( "Pinecone index statistics: %s",stats)
    vector_store = PineconeVectorStore(index=index,embedding=embeddings,namespace=NAMESPACE)
    logging.info("Connected LangChain to Pinecone vector store")
    logging.info("Creating advanced retriever")
    retriever = create_advanced_retriever(vector_store)
    logging.info("Advanced retriever created successfully")
    query = input("\nEnter your question: ")
    logging.info("Retrieval started for query: %s",query)
    retrieved_documents = retriever.invoke(query)
    logging.info("Retrieval and reranking completed. Final documents retrieved: %d",len(retrieved_documents))
    context_parts = []
    for document in retrieved_documents:
        source = document.metadata.get("source","Unknown source")
        page_numbers = document.metadata.get("page_numbers",[])
        headings = document.metadata.get("headings",[])
        chunk_id = document.metadata.get("chunk_id","Unknown chunk")
        context_part = f"""
            SOURCE: {source}
            PAGE(S): {page_numbers}
            SECTION: {headings}
            CHUNK ID: {chunk_id}
            CONTENT:
            {document.page_content}
            """
        context_parts.append(context_part)
    context = "\n\n".join(context_parts)
            
    llm = ChatOpenAI(model="gpt-4.1-mini",temperature=0)
    answer_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
            You are a question-answering assistant for a
            Retrieval-Augmented Generation (RAG) system.

            Answer the user's question using only the provided
            retrieved document context.

            RULES:

            1. Use only information contained in the retrieved
            context.

            2. Do not introduce unsupported facts.

            3. Combine information from multiple retrieved chunks
            when necessary.

            4. Provide a clear, coherent, and concise answer.

            5. Cite the source file and page number for factual
            statements whenever that information is available.

            6. Use citations in this format:

            [Source: filename, Page: 5]

            For multiple pages:

            [Source: filename, Pages: 5-6]

            7. If information comes from multiple sources or
            different pages, cite the appropriate source after
            the corresponding statement.

            8. Use the SOURCE and PAGE(S) metadata supplied with
            each context chunk. Never invent a source or page
            number.

            9. Do not cite a source unless the corresponding
            retrieved context supports the statement.

            10. If the retrieved context does not contain enough
                information to answer the question, state that
                clearly.

            11. Do not use general knowledge to fill gaps in the
                retrieved context.
            """
                    ),

                    (
                        "human",
                        """
            USER QUESTION:

            {query}


            RETRIEVED DOCUMENT CONTEXT:

            {context}


            Using only the retrieved context above, provide a
            clear and coherent answer with source and page
            citations.
            """
                    ),
                ]
            )

    answer_chain = answer_prompt|llm
    logging.info("Generating final answer from retrieved context")
    response = answer_chain.invoke({"query": query,"context": context})
    logging.info("Final answer generated successfully")
    print(response.content)


if __name__ == "__main__":
    logging.info("RAG retrieval pipeline started")
    try:
        main()
    except Exception as e:
        logging.exception("RAG retrieval pipeline failed")
        raise CustomException(e,sys)
    logging.info("RAG retrieval pipeline completed successfully")