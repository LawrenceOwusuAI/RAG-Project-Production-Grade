from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from portkey_ai import (
    PORTKEY_GATEWAY_URL,
    createHeaders,
)

from logger import logging


load_dotenv(override=True)


def create_portkey_llm(
    model: str,
    component: str = "rag",
) -> ChatOpenAI:

    """
    Create a LangChain ChatOpenAI client that routes
    LLM requests through the Portkey AI Gateway.

    The actual Portkey configuration is stored
    centrally in Portkey and referenced using its
    pc-... Config slug.

    Portkey config handles:
    - fallback
    - load balancing
    - retries
    - exponential backoff
    - caching
    - input guardrails
    - output guardrails
    - provider routing

    Portkey also provides:
    - token tracking
    - cost tracking
    - latency monitoring
    - request logs
    - observability
    """


    # =====================================================
    # 1. READ PORTKEY API KEY
    # =====================================================

    portkey_api_key = os.getenv(
        "PORTKEY_API_KEY"
    )


    if not portkey_api_key:

        raise ValueError(
            "PORTKEY_API_KEY was not found. "
            "Add it to your .env file."
        )


    # =====================================================
    # 2. READ SAVED PORTKEY CONFIG SLUG
    # =====================================================

    portkey_config_id = os.getenv(
        "PORTKEY_CONFIG_ID"
    )


    if not portkey_config_id:

        raise ValueError(
            "PORTKEY_CONFIG_ID was not found. "
            "Add the saved pc-... Portkey Config "
            "slug to your .env file."
        )


    # =====================================================
    # 3. VALIDATE CONFIG SLUG
    # =====================================================

    if not portkey_config_id.startswith("pc-"):

        raise ValueError(
            "PORTKEY_CONFIG_ID must be a saved "
            "Portkey Config slug beginning with 'pc-'. "
            f"Received: {portkey_config_id}"
        )


    # =====================================================
    # 4. CREATE PORTKEY HEADERS
    # =====================================================

    portkey_headers = createHeaders(

        api_key=portkey_api_key,


        # IMPORTANT:
        #
        # We are NOT passing an inline dictionary.
        #
        # We are passing the saved Portkey Config slug:
        #
        # pc-xxxxxxxx
        #
        config=portkey_config_id,


        # Metadata helps distinguish different
        # parts of the RAG application in Portkey.
        metadata={

            "application": "rag-project",

            "component": component,

        },

    )


    # =====================================================
    # 5. CREATE LANGCHAIN LLM
    # =====================================================

    llm = ChatOpenAI(

        model=model,

        


        # LangChain sends requests to Portkey
        # instead of directly to OpenAI.
        base_url=PORTKEY_GATEWAY_URL,


        # Portkey authentication + saved config
        # are sent through these headers.
        default_headers=portkey_headers,


        # ChatOpenAI requires an API-key-shaped value.
        # Provider credentials are handled by Portkey.
        api_key="PORTKEY",


        # IMPORTANT:
        #
        # Disable ChatOpenAI retries.
        #
        # Your saved Portkey config already contains:
        #
        # retry.attempts = 3
        #
        # We don't want:
        #
        # LangChain retry
        #       +
        # Portkey retry
        #
        max_retries=0,

    )


    # =====================================================
    # 6. LOG INITIALIZATION
    # =====================================================

    logging.info(
        "Portkey LLM Gateway initialized. "
        "model=%s component=%s config=%s",
        model,
        component,
        portkey_config_id,
    )


    # =====================================================
    # 7. RETURN LANGCHAIN MODEL
    # =====================================================

    return llm