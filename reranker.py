from logger import logging
from typing import Any
from pydantic import BaseModel,Field,ConfigDict
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from exception import CustomException


QUERY_MODEL = "gpt-4.1-mini"
CROSS_ENCODER_MODEL = "BAAI/bge-reranker-base"
#Number of chunks Pinecone returns for EACH transformed query
BASE_K = 64
# Maximum number of documents sent to the cross encoder
MAX_CANDIDATES = 20
# Final number of chunks returned after cross-encoder reranking
FINAL_TOP_N = 4
# Reciprocal Rank Fusion constant
RRF_K = 60

class QueryTransformation(BaseModel):

    rewritten_query: str = Field(description=("One clear retrieval-optimized version of the user's original query."))
    expanded_queries: list[str] = Field(default_factory=list,
        description=(
            "Alternative search queries using synonyms, "
            "technical terminology, abbreviations, "
            "related expressions, and alternative "
            "formulations of the same information need.")
           )
    decomposed_queries: list[str] = Field(default_factory=list,
        description=(
            "Independent and self-contained subquestions "
            "created when the original query contains "
            "multiple distinct information needs.")
         )

class QueryTransformer:
    def __init__(self):
        logging.info( "Creating query transformation model")
        llm = ChatOpenAI(model=QUERY_MODEL, temperature=0)
        structured_llm = llm.with_structured_output(QueryTransformation)
        self.prompt = ChatPromptTemplate.from_messages(
                [(
                        "system",
                        """
                You are the query-transformation component of a
                Retrieval-Augmented Generation (RAG) system.
                Your responsibility is to transform a user's question
                into high-quality search queries that maximize the
                probability of retrieving relevant document chunks
                from a semantic vector database.

                You are NOT responsible for answering the user's
                question.
                You MUST NOT answer the question.
                You MUST NOT fabricate information.
                Your only responsibility is to improve document
                retrieval.

                For every user query, perform THREE separate operations:

                1. QUERY REWRITING
                Create exactly ONE improved version of the original
                query.The purpose of rewriting is to express the same
                information more clearly and in terminology that
                is suitable for semantic retrieval.

                The rewritten query must:
                - preserve the meaning of the original query;
                - preserve all important entities and concepts;
                - preserve conditions and constraints;
                - correct unclear grammar or spelling;
                - replace vague wording with precise terminology when
                the intended meaning is clear;
                - expand obvious abbreviations when useful;
                - convert conversational wording into a concise,
                retrieval-friendly formulation.

                Do NOT introduce unrelated information.
                Do NOT change the user's intent.
                Do NOT answer the question.

                Example:
                Original: "How does it read words from scanned pages?"
                Rewritten: "How does optical character recognition extract text from scanned document pages?"

                2. QUERY EXPANSION
                Generate approximately 2 to 4 alternative search
                queries when useful. The purpose of query expansion is to improve
                RETRIEVAL RECALL. A relevant document may discuss the same concept using
                different terminology from the terminology used by the
                user. Expanded queries should therefore represent the SAME
                underlying information need using alternative useful
                expressions.
                Query expansion may include:
                - synonyms;
                - technical terminology;
                - domain-specific terminology;
                - abbreviations;
                - expanded forms of abbreviations;
                - alternative names for the same concept;
                - closely related terminology;
                - alternative grammatical formulations;
                - broader terminology when directly relevant;
                - narrower terminology when directly relevant.

                Every expanded query must remain directly related to
                the original information need.
                Do NOT generate unrelated queries simply to increase
                the number of searches.
                Do NOT answer the user's question.

                Example:
                Original: "How does OCR extract text?"
                Expanded queries could include:
                - "optical character recognition text extraction"
                - "OCR text recognition from scanned documents"
                - "machine-readable text extraction from document images"
                - "character recognition in scanned PDF pages"

                3. QUERY DECOMPOSITION
                Determine whether the original query contains multiple
                distinct information needs. If it does, break it into smaller, independent,self-contained search questions.

                Each decomposed query should:
                - represent one information need;
                - make sense when searched independently;
                - preserve important context;
                - preserve important entities;
                - avoid unnecessary overlap with other subquestions.

                Do NOT decompose a simple single-topic question.
                If decomposition is unnecessary, return an empty list.

                Example:
                Original:"How does OCR recognize text and how are tables extracted from PDFs?"
                Decomposed queries:
                - "How does optical character recognition extract text
                from PDF documents?"
                - "How are tables detected and extracted from PDF documents?"


            
                IMPORTANT DIFFERENCE
        
                REWRITING:
                Creates ONE clearer formulation of the user's original
                query.

                EXPANSION:
                Creates ALTERNATIVE expressions of the same information
                need so that documents using different terminology may
                also be retrieved.

                DECOMPOSITION:
                Splits a MULTI-PART information need into independent
                questions.

                Do not treat these three operations as the same task.

                OUTPUT REQUIREMENTS
                Return only the structured output required by the
                schema.
                The output contains:
                rewritten_query:
                    One improved retrieval query.
                expanded_queries:
                    Alternative formulations representing the same
                    information need.
                decomposed_queries:
                    Independent subquestions when decomposition is
                    necessary.

                Do not answer the user's question.
                Do not provide explanations outside the structured
                output.
                Do not invent document-specific facts.

                Your goal is exclusively to maximize the probability
                of retrieving relevant document chunks.
                """
                                    ),

                                    (
                                        "human",
                                        """
                Transform the following user query for document
                retrieval:
                {query}
                """)])
                                  
        self.chain = self.prompt|structured_llm
        
    def transform(self, query: str)-> QueryTransformation:
        logging.info("Transforming user query")
        transformation = self.chain.invoke({"query": query})
        return transformation

def create_base_retriever(vector_store):
    logging.info("Creating Pinecone base retriever")
    retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": BASE_K })
    return retriever

class TransformedQueryRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    base_retriever: Any
    query_transformer: Any
    max_candidates: int = MAX_CANDIDATES
    rrf_k: int = RRF_K

    def _unique_queries(self,queries: list[str]) -> list[str]:
        unique_queries = []
        seen = set()
        for query in queries:
            normalized_query = query.strip().lower()
            if normalized_query not in seen:
                seen.add(normalized_query)
                unique_queries.append(query.strip())
        return unique_queries

    def _get_document_key(self,document: Document):
        chunk_id = document.metadata.get("chunk_id")
        if chunk_id:
            return ("chunk_id",chunk_id)
        return ("content",document.page_content)

    def _reciprocal_rank_fusion(self, retrieval_results:list[list[Document]]) -> list[Document]:
        logging.info("Applying Reciprocal Rank Fusion")
        rrf_scores = {}
        document_map = {}
        for documents in retrieval_results:
            for rank, document in enumerate(documents, start=1):
                document_key = self._get_document_key(document)
                document_map[document_key] = document
                score = (1/ (self.rrf_k + rank))
                rrf_scores[document_key] = rrf_scores.get(document_key, 0.0,) + score
                
        ranked_document_keys = sorted(rrf_scores,key=lambda key:rrf_scores[key],reverse=True)
        ranked_documents = []
        for document_key in ranked_document_keys:
            document = document_map[document_key]
            document.metadata["rrf_score"] = rrf_scores[document_key]
            ranked_documents.append(document)
        logging.info("Documents after RRF ranking: %d",len(ranked_documents))
        return ranked_documents

    def _get_relevant_documents(self,query: str,*,run_manager=None) -> list[Document]:
        logging.info("Running transformed-query retrieval")
        transformation = self.query_transformer.transform(query)
        logging.info("Original query: %s", query)
        logging.info("Rewritten query: %s",transformation.rewritten_query)
        logging.info("Expanded queries: %s",transformation.expanded_queries)
        logging.info("Decomposed queries: %s",transformation.decomposed_queries)
        search_queries = [
            # Original query
            query,
            # Rewritten query
            transformation.rewritten_query,
            # Expanded queries
            *transformation.expanded_queries,
            # Decomposed queries
            *transformation.decomposed_queries,
        ]
        search_queries = self._unique_queries(search_queries)
        logging.info("Total unique retrieval queries: %d",len(search_queries))
        retrieval_results = []
        for search_query in search_queries:
            logging.info("Searching Pinecone with query: %s",search_query)
            documents = self.base_retriever.invoke(search_query)
            retrieval_results.append(documents)
        ranked_documents = self._reciprocal_rank_fusion(retrieval_results)
        candidate_documents = ranked_documents[:self.max_candidates]
        logging.info("Documents sent to cross encoder: %d",len(candidate_documents))
        return candidate_documents

def create_cross_encoder():
    logging.info("Loading cross-encoder model: %s",CROSS_ENCODER_MODEL)
    cross_encoder = HuggingFaceCrossEncoder(
            model_name=CROSS_ENCODER_MODEL,
            model_kwargs={
                "device": "cpu"   
            })
    return cross_encoder

def create_cross_encoder_reranker():
    logging.info("Creating cross-encoder reranker")
    cross_encoder = create_cross_encoder()
    reranker = CrossEncoderReranker(model=cross_encoder,top_n=FINAL_TOP_N)
    return reranker

def create_advanced_retriever(vector_store):
    logging.info("Creating advanced retrieval pipeline")
    base_retriever = create_base_retriever(vector_store)
    query_transformer = QueryTransformer()
    transformed_retriever = TransformedQueryRetriever(
            base_retriever = base_retriever,
            query_transformer = query_transformer,
            max_candidates=MAX_CANDIDATES,
            rrf_k=RRF_K
        )
    reranker = create_cross_encoder_reranker()
    final_retriever = ContextualCompressionRetriever(
            base_retriever=transformed_retriever,
            base_compressor= reranker
            )
    logging.info("Advanced retriever created successfully")
    return final_retriever