from contextlib import asynccontextmanager
import os
from exception import CustomException
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pydantic import BaseModel, Field
from chunking import PDF_PATH
from logger import logging
from reranker import create_advanced_retriever
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_NAME = "rag-project"
NAMESPACE = PDF_PATH.stem

class QueryRequest(BaseModel):
    query: str = Field(min_length=1)

class QueryResponse(BaseModel):
    query: str
    answer: str

def create_answer_prompt():
    return ChatPromptTemplate.from_messages(
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
            """,
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
            """,
            ),
        ]
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("FastAPI RAG application startup started")
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
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    logging.info("Embedding model loaded successfully")

    logging.info("Connecting to Pinecone")
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(INDEX_NAME)

    logging.info("Connected to Pinecone index: %s",INDEX_NAME)
    
    stats = index.describe_index_stats()

    logging.info("Pinecone index statistics: %s",stats)

    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=NAMESPACE,
    )

    logging.info("Connected LangChain to Pinecone vector store")

    logging.info("Creating advanced retriever")

    retriever = create_advanced_retriever(vector_store)

    logging.info("Advanced retriever created successfully")

    logging.info("Loading answer generation model")

    llm = ChatOpenAI(model="gpt-4.1-mini",temperature=0,)

    logging.info("Answer generation model loaded successfully")

    answer_prompt = create_answer_prompt()

    answer_chain = answer_prompt | llm

    app.state.embeddings = embeddings
    app.state.pc = pc
    app.state.index = index
    app.state.vector_store = vector_store
    app.state.retriever = retriever
    app.state.llm = llm
    app.state.answer_prompt = answer_prompt
    app.state.answer_chain = answer_chain
    app.state.ready = True

    logging.info("FastAPI RAG application startup completed successfully")

    yield

    logging.info("FastAPI RAG application shutdown started")

    app.state.ready = False

    logging.info( "FastAPI RAG application shutdown completed")


app = FastAPI(
    title="RAG API",
    description="FastAPI service for the RAG application",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/")
def root():
    return {
        "message": "RAG API is running",
        "docs": "/docs",
        "health": "/health",
        "query_endpoint": "/query",
    }


@app.get("/health")
def health():
    return {
        "status": (
            "healthy"
            if getattr(app.state, "ready", False)
            else "starting"
        ),
        "index": INDEX_NAME,
        "namespace": NAMESPACE,
        "embedding_model": EMBEDDING_MODEL_NAME,
    }

@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):

    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

    if not getattr(
        app.state,
        "ready",
        False,
    ):
        raise HTTPException(
            status_code=503,
            detail="RAG models are not ready.",
        )

    try:

        logging.info("Retrieval started for query: %s",query)

        retrieved_documents = app.state.retriever.invoke(query)
        
        logging.info(
            "Retrieval and reranking completed. "
            "Final documents retrieved: %d",
            len(retrieved_documents),
        )

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

        logging.info("Generating final answer from retrieved context")
        
        response = app.state.answer_chain.invoke(
                {
                    "query": query,
                    "context": context,
                }
            )
        
        logging.info("Final answer generated successfully")

        return QueryResponse(query=query,answer=response.content)
   
    except HTTPException:
        raise

    except Exception as e:
        logging.exception("RAG query processing failed")

        raise HTTPException(status_code=500,
            detail=(
                "RAG query processing failed: "
                f"{str(e)}"
            ),
        ) from e