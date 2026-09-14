import requests
import streamlit as st
import requests
import streamlit as st


# ============================================================
# FASTAPI BACKEND
# ============================================================

API_URL = "http://127.0.0.1:8000/query"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>


.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(91, 124, 250, 0.13),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 15%,
            rgba(151, 117, 250, 0.10),
            transparent 28%
        ),
        linear-gradient(
            135deg,
            #edf4fb 0%,
            #e6eef8 45%,
            #edf1f8 100%
        );

    color: #172033;
}

.block-container {
    max-width: 1050px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.main-title {
    text-align: center;
    font-size: 3.2rem;
    font-weight: 800;
    margin-bottom: 0.4rem;

    background: linear-gradient(
        90deg,
        #304ffe,
        #5f3dc4,
        #7048e8
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    color: #526375;
    font-size: 1.05rem;
    margin-bottom: 1rem;
}

.status-box {
    text-align: center;
    margin-bottom: 2rem;
    color: #526375;
    font-size: 0.92rem;
    font-weight: 500;
}


.status-dot {
    display: inline-block;

    width: 9px;
    height: 9px;

    border-radius: 50%;

    background: #2f9e44;

    margin-right: 7px;

    box-shadow:
        0 0 8px
        rgba(47, 158, 68, 0.50);
}


.info-card {

    background: rgba(
        255,
        255,
        255,
        0.88
    );

    border: 1px solid #d9e2ec;

    border-radius: 20px;

    padding: 24px 28px;

    margin-bottom: 2rem;

    box-shadow:
        0 10px 30px
        rgba(30, 50, 80, 0.08);
}


.info-title {

    font-size: 1.15rem;

    font-weight: 700;

    color: #172033;

    margin-bottom: 8px;
}


.info-text {

    color: #536273;

    line-height: 1.7;
}


.question-title {

    font-size: 1.25rem;

    font-weight: 700;

    color: #172033;

    margin-top: 1rem;

    margin-bottom: 0.4rem;
}


.stMarkdown,
.stCaption {

    color: #26364a;
}


.stTextArea textarea {

    background-color: #ffffff !important;

    color: #172033 !important;

    border-radius: 15px !important;

    border: 1px solid #cbd6e2 !important;

    font-size: 1rem !important;

    min-height: 125px !important;

    padding: 16px !important;

    box-shadow:
        0 4px 15px
        rgba(30, 50, 80, 0.05) !important;
}


/* Input text when selected */
.stTextArea textarea:focus {

    border: 1px solid #5c7cfa !important;

    box-shadow:
        0 0 0 2px
        rgba(92, 124, 250, 0.15) !important;
}


/* Placeholder */
.stTextArea textarea::placeholder {

    color: #8291a3 !important;

    opacity: 1 !important;
}


div[data-testid="stFormSubmitButton"] button {

    width: 100%;

    height: 3rem;

    border: none !important;

    border-radius: 12px !important;

    font-size: 1rem !important;

    font-weight: 700 !important;

    color: white !important;

    background:
        linear-gradient(
            90deg,
            #4263eb,
            #6741d9
        ) !important;

    box-shadow:
        0 6px 18px
        rgba(66, 99, 235, 0.20);

    transition: all 0.2s ease;
}


div[data-testid="stFormSubmitButton"] button:hover {

    transform: translateY(-1px);

    box-shadow:
        0 10px 24px
        rgba(66, 99, 235, 0.28);
}


/* Button text */
div[data-testid="stFormSubmitButton"] button p {

    color: white !important;
}


[data-testid="stChatMessage"] {

    background: #ffffff !important;

    border: 1px solid #d7e0ea !important;

    border-radius: 18px !important;

    padding: 16px !important;

    margin-top: 12px !important;

    margin-bottom: 14px !important;

    box-shadow:
        0 6px 20px
        rgba(30, 50, 80, 0.07) !important;
}


[data-testid="stChatMessage"] p {

    color: #1c2735 !important;

    font-size: 1rem !important;

    line-height: 1.75 !important;
}


[data-testid="stChatMessage"] span {

    color: #1c2735 !important;
}


[data-testid="stChatMessage"] li {

    color: #1c2735 !important;

    line-height: 1.7 !important;
}


/* Markdown container */
[data-testid="stChatMessage"]
[data-testid="stMarkdownContainer"] {

    color: #1c2735 !important;
}

[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3,
[data-testid="stChatMessage"] h4 {

    color: #172033 !important;
}

[data-testid="stChatMessage"] strong {

    color: #111827 !important;

    font-weight: 700 !important;
}

[data-testid="stChatMessage"] a {

    color: #364fc7 !important;

    font-weight: 600;
}

[data-testid="stChatMessage"] code {

    color: #b42318 !important;

    background: #f2f4f7 !important;

    padding: 2px 5px;

    border-radius: 5px;
}

[data-testid="stCaptionContainer"] p {

    color: #617285 !important;
}

[data-testid="stSpinner"] {

    color: #253858 !important;
}

[data-testid="stForm"] {

    background: rgba(
        255,
        255,
        255,
        0.30
    );

    border: 1px solid
        rgba(
            190,
            205,
            220,
            0.55
        );

    border-radius: 18px;

    padding: 14px;
}

footer {
    visibility: hidden;
}

</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">AI Research Assistant</div>',unsafe_allow_html=True,)
st.markdown(
    """
<div class="subtitle">
Ask questions about your documents and receive answers grounded in retrieved evidence.
</div>
""",
    unsafe_allow_html=True,
)


st.markdown(
    """
<div class="status-box">
<span class="status-dot"></span>
RAG Assistant Online
</div>
""",
    unsafe_allow_html=True,
)

# QUERY FORM

with st.form("query_form", clear_on_submit=True):
    query = st.text_area("Your question",label_visibility="collapsed")
    submit_button = st.form_submit_button("✨ Ask Question")

if submit_button:
    query = query.strip()
    if not query:
        st.warning("Please enter a question before submitting.")
    else:
        with st.spinner(
                "Generating an answer..."):
                try:
                    response = requests.post(API_URL,json={"query": query},timeout=120)
                    response.raise_for_status()
                    data = response.json()
                    answer = data["answer"]
                    st.markdown(answer)

                except requests.exceptions.ConnectionError:

                    st.error(
                        "Unable to connect to the FastAPI backend. "
                        "Make sure FastAPI is running on "
                        "http://127.0.0.1:8000."
                    )

                except requests.exceptions.Timeout:
                    st.error("The request took too long. ""Please try again.")
                    
                except requests.exceptions.HTTPError as e:
                    st.error(f"FastAPI returned an error: {str(e)}")
                except Exception as e:
                    st.error(f"Something went wrong: {str(e)}")
                    

        
         
