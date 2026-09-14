from contextlib import asynccontextmanager
import os
import time

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

import logfire


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

# This must happen before logfire.configure()
# so LOGFIRE_TOKEN and OTEL variables are available.

load_dotenv(
    override=True
)


# =========================================================
# APPLICATION CONSTANTS
# =========================================================

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

INDEX_NAME = "rag-project"

NAMESPACE = PDF_PATH.stem


# =========================================================
# CONFIGURE LOGFIRE
# =========================================================

logfire.configure(
    service_name="rag-api",
    environment="development",
    send_to_logfire=True,
)


# =========================================================
# METRICS
# =========================================================


# Total number of /query requests received.
request_counter = logfire.metric_counter(
    "rag.requests",
    unit="1",
    description="Total number of RAG requests",
)


# Total failed RAG requests.
error_counter = logfire.metric_counter(
    "rag.errors",
    unit="1",
    description="Total number of failed RAG requests",
)


# Number of requests currently being processed.
active_requests = logfire.metric_up_down_counter(
    "rag.active_requests",
    unit="1",
    description="Number of active RAG requests",
)


# Total end-to-end request duration.
request_duration = logfire.metric_histogram(
    "rag.request.duration",
    unit="ms",
    description="End-to-end RAG request duration",
)


# Retrieval + reranking duration.
retrieval_duration = logfire.metric_histogram(
    "rag.retrieval.duration",
    unit="ms",
    description="Retrieval and reranking duration",
)


# Final answer generation duration.
llm_duration = logfire.metric_histogram(
    "rag.llm.duration",
    unit="ms",
    description="LLM answer generation duration",
)


# Number of documents returned by final retriever.
documents_retrieved = logfire.metric_histogram(
    "rag.documents.retrieved",
    unit="1",
    description="Number of documents returned after reranking",
)


# =========================================================
# REQUEST / RESPONSE SCHEMAS
# =========================================================


class QueryRequest(BaseModel):

    query: str = Field(
        min_length=1
    )


class QueryResponse(BaseModel):

    query: str

    answer: str


# =========================================================
# ANSWER GENERATION PROMPT
# =========================================================


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


# =========================================================
# FASTAPI LIFESPAN
# =========================================================


@asynccontextmanager
async def lifespan(app: FastAPI):

    logging.info(
        "FastAPI RAG application startup started"
    )

    logfire.info(
        "RAG application startup started"
    )

    # -----------------------------------------------------
    # LOAD API KEYS
    # -----------------------------------------------------

    pinecone_api_key = os.getenv(
        "PINECONE_API_KEY"
    )

    if not pinecone_api_key:

        raise ValueError(
            "PINECONE_API_KEY was not found."
        )


    openai_api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not openai_api_key:

        raise ValueError(
            "OPENAI_API_KEY was not found."
        )


    # -----------------------------------------------------
    # EMBEDDING MODEL
    # -----------------------------------------------------

    logging.info(
        "Loading embedding model"
    )

    with logfire.span(
        "rag.startup.load_embedding_model",
        model=EMBEDDING_MODEL_NAME,
    ):

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={
                "device": "cpu"
            },
            encode_kwargs={
                "normalize_embeddings": True
            },
        )


    logging.info(
        "Embedding model loaded successfully"
    )


    # -----------------------------------------------------
    # PINECONE
    # -----------------------------------------------------

    logging.info(
        "Connecting to Pinecone"
    )


    with logfire.span(
        "rag.startup.connect_pinecone",
        index_name=INDEX_NAME,
        namespace=NAMESPACE,
    ):

        pc = Pinecone(
            api_key=pinecone_api_key
        )

        index = pc.Index(
            INDEX_NAME
        )

        stats = index.describe_index_stats()


    logging.info(
        "Connected to Pinecone index: %s",
        INDEX_NAME,
    )


    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=NAMESPACE,
    )


    logging.info(
        "Connected LangChain to Pinecone vector store"
    )


    # -----------------------------------------------------
    # RETRIEVER
    # -----------------------------------------------------

    logging.info(
        "Creating advanced retriever"
    )


    with logfire.span(
        "rag.startup.create_retriever"
    ):

        retriever = create_advanced_retriever(
            vector_store
        )


    logging.info(
        "Advanced retriever created successfully"
    )


    # -----------------------------------------------------
    # ANSWER MODEL
    # -----------------------------------------------------

    logging.info(
        "Loading answer generation model"
    )


    with logfire.span(
        "rag.startup.create_llm",
        model="gpt-4.1-mini",
    ):

        llm = ChatOpenAI(
            model="gpt-4.1-mini",
            temperature=0,
        )


    logging.info(
        "Answer generation model loaded successfully"
    )


    answer_prompt = create_answer_prompt()

    answer_chain = (
        answer_prompt
        | llm
    )


    # -----------------------------------------------------
    # STORE RESOURCES IN APP.STATE
    # -----------------------------------------------------

    app.state.embeddings = embeddings

    app.state.pc = pc

    app.state.index = index

    app.state.vector_store = vector_store

    app.state.retriever = retriever

    app.state.llm = llm

    app.state.answer_prompt = answer_prompt

    app.state.answer_chain = answer_chain

    app.state.ready = True


    logging.info(
        "FastAPI RAG application startup completed successfully"
    )

    logfire.info(
        "RAG application startup completed",
        index_name=INDEX_NAME,
        namespace=NAMESPACE,
        embedding_model=EMBEDDING_MODEL_NAME,
    )


    yield


    # -----------------------------------------------------
    # SHUTDOWN
    # -----------------------------------------------------

    logging.info(
        "FastAPI RAG application shutdown started"
    )

    logfire.info(
        "RAG application shutdown started"
    )


    app.state.ready = False


    logging.info(
        "FastAPI RAG application shutdown completed"
    )

    logfire.info(
        "RAG application shutdown completed"
    )


# =========================================================
# CREATE FASTAPI APPLICATION
# =========================================================


app = FastAPI(
    title="RAG API",
    description="FastAPI service for the RAG application",
    version="1.0.0",
    lifespan=lifespan,
)


# =========================================================
# SAFE FASTAPI REQUEST INSTRUMENTATION
# =========================================================


def request_attributes_mapper(
    request,
    attributes,
):

    """
    Prevent Logfire from automatically recording
    successful endpoint argument values.

    This is important because QueryRequest contains
    the user's RAG question.
    """

    if attributes["errors"]:

        return {
            "errors": attributes["errors"]
        }

    return {}


logfire.instrument_fastapi(
    app,
    request_attributes_mapper=request_attributes_mapper,
)


# =========================================================
# ROOT ENDPOINT
# =========================================================


@app.get("/")
def root():

    return {
        "message": "RAG API is running",
        "docs": "/docs",
        "health": "/health",
        "query_endpoint": "/query",
    }


# =========================================================
# HEALTH ENDPOINT
# =========================================================


@app.get("/health")
def health():

    return {
        "status": (
            "healthy"
            if getattr(
                app.state,
                "ready",
                False,
            )
            else "starting"
        ),
        "index": INDEX_NAME,
        "namespace": NAMESPACE,
        "embedding_model": EMBEDDING_MODEL_NAME,
    }


# =========================================================
# QUERY ENDPOINT
# =========================================================


@app.post(
    "/query",
    response_model=QueryResponse,
)
def query_rag(
    request: QueryRequest,
):

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


    # -----------------------------------------------------
    # RECORD REQUEST ARRIVAL
    # -----------------------------------------------------

    request_counter.add(1)

    active_requests.add(1)

    request_start = time.perf_counter()


    # Do NOT send the complete user query.
    logfire.info(
        "RAG query received",
        query_length=len(query),
    )


    logging.info(
        "RAG query received. Query length: %d",
        len(query),
    )


    try:

        # =================================================
        # RETRIEVAL + RERANKING
        # =================================================

        retrieval_start = time.perf_counter()


        with logfire.span(
            "rag.retrieve_and_rerank",
            query_length=len(query),
        ) as retrieval_span:


            logging.info(
                "Retrieval and reranking started"
            )


            retrieved_documents = (
                app.state.retriever.invoke(
                    query
                )
            )


            document_count = len(
                retrieved_documents
            )


            retrieval_span.set_attribute(
                "rag.documents.count",
                document_count,
            )


            logging.info(
                "Retrieval and reranking completed. "
                "Final documents retrieved: %d",
                document_count,
            )


        retrieval_ms = (
            time.perf_counter()
            - retrieval_start
        ) * 1000


        retrieval_duration.record(
            retrieval_ms
        )


        documents_retrieved.record(
            document_count
        )


        logfire.info(
            "Retrieval and reranking completed",
            document_count=document_count,
            duration_ms=retrieval_ms,
        )


        # =================================================
        # BUILD CONTEXT
        # =================================================

        context_parts = []


        for document in retrieved_documents:

            source = document.metadata.get(
                "source",
                "Unknown source",
            )

            page_numbers = document.metadata.get(
                "page_numbers",
                [],
            )

            headings = document.metadata.get(
                "headings",
                [],
            )

            chunk_id = document.metadata.get(
                "chunk_id",
                "Unknown chunk",
            )


            context_part = f"""
            SOURCE: {source}
            PAGE(S): {page_numbers}
            SECTION: {headings}
            CHUNK ID: {chunk_id}
            CONTENT:
            {document.page_content}
            """


            context_parts.append(
                context_part
            )


        context = "\n\n".join(
            context_parts
        )


        # =================================================
        # ANSWER GENERATION
        # =================================================

        logging.info(
            "Generating final answer from retrieved context"
        )


        generation_start = time.perf_counter()


        with logfire.span(
            "rag.generate_answer",
            document_count=document_count,
            model="gpt-4.1-mini",
        ):


            response = app.state.answer_chain.invoke(
                {
                    "query": query,
                    "context": context,
                }
            )


        generation_ms = (
            time.perf_counter()
            - generation_start
        ) * 1000


        llm_duration.record(
            generation_ms
        )


        logging.info(
            "Final answer generated successfully"
        )


        logfire.info(
            "Final answer generated successfully",
            duration_ms=generation_ms,
            document_count=document_count,
        )


        return QueryResponse(
            query=query,
            answer=response.content,
        )


    except HTTPException:

        error_counter.add(1)

        raise


    except Exception as e:

        error_counter.add(1)


        logging.exception(
            "RAG query processing failed"
        )


        logfire.exception(
            "RAG query processing failed",
            error_type=type(e).__name__,
        )


        raise HTTPException(
            status_code=500,
            detail=(
                "RAG query processing failed: "
                f"{str(e)}"
            ),
        ) from e


    finally:

        # -------------------------------------------------
        # REQUEST FINISHED
        # -------------------------------------------------

        active_requests.add(-1)


        total_duration_ms = (
            time.perf_counter()
            - request_start
        ) * 1000


        request_duration.record(
            total_duration_ms
        )


        logfire.info(
            "RAG request processing finished",
            duration_ms=total_duration_ms,
        )