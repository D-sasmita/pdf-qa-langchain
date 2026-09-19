from dotenv import load_dotenv
load_dotenv()

import os
import time
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

PROMPT_TEXT = """You are a helpful assistant that answers questions using only the provided context, which may come from one or more PDF documents.

Instructions:
- Answer using only the information in the context below.
- Do not use outside knowledge or make up information.
- If context is drawn from multiple documents, you may synthesize across them, but do not invent connections that aren't supported by the text.
- If the answer is not present in the context, clearly say: "The information was not found in the document(s)."

Context:
{context}

Question:
{question}

Answer:"""

SUGGESTED_QUESTIONS = [
    "Summarise this document",
    "What are the key points?",
    "What is this document about?",
]

BROAD_QUERY_KEYWORDS = (
    "summar", "overview", "key point", "main point",
    "both document", "each document", "all document", "every document",
)


def is_broad_query(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in BROAD_QUERY_KEYWORDS)


# ---------------- Core pipeline functions ----------------

def load_pdf(uploaded_file):
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    try:
        loader = PyPDFLoader(temp_path)
        documents = loader.load()
        for doc in documents:
            doc.metadata["source"] = uploaded_file.name  # use real filename, not temp path
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_documents(documents)


def create_vector_store(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embedding_model)


def create_rag_chain(vector_store, num_files=1):
    # Scale retrieval breadth with document count so broad/cross-document
    # questions aren't dominated by whichever PDF contributed the most chunks.
    k = min(3 * max(num_files, 1), 12)
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEXT)

    answer_chain = prompt_template | llm | StrOutputParser()

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | answer_chain
    )
    return chain, retriever, answer_chain


def format_docs(docs):
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'document')}, page {doc.metadata.get('page')}]\n{doc.page_content}"
        for doc in docs
    )


def build_broad_context(all_chunks, chunks_per_doc=4):
    """Group chunks by source file, in original reading order, and take the
    first few from each — used for summarise/overview-style questions where
    similarity search doesn't reliably match any single chunk."""
    by_source = {}
    for doc in all_chunks:
        src = doc.metadata.get("source", "document")
        by_source.setdefault(src, []).append(doc)

    selected = []
    for src, docs in by_source.items():
        selected.extend(docs[:chunks_per_doc])
    return selected


def answer_question(chain, retriever, answer_chain, all_chunks, question):
    if is_broad_query(question):
        selected_docs = build_broad_context(all_chunks)
        context = format_docs(selected_docs)
        answer = answer_chain.invoke({"context": context, "question": question})
        return answer, selected_docs
    else:
        answer = chain.invoke(question)
        sources = retriever.invoke(question)
        return answer, sources


def classify_llm_error(e: Exception) -> str:
    msg = str(e).lower()
    if "401" in msg or "invalid" in msg or "authentication" in msg or "api key" in msg:
        return "Invalid or expired API key. Check your GROQ_API_KEY in .env and restart the app."
    if "429" in msg or "rate limit" in msg:
        return "Rate limit reached on the Groq API. Wait a moment and try again."
    if "timeout" in msg or "timed out" in msg:
        return "The request timed out. Check your connection and try again."
    return f"Something went wrong generating the answer: {e}"


def reset_session():
    for key in ("rag_chain", "retriever", "answer_chain", "all_chunks", "pdf_processed",
                "file_names", "num_pages", "num_chunks", "chat_history", "pending_question"):
        st.session_state.pop(key, None)


# ---------------- Page config + styling ----------------

st.set_page_config(page_title="PDF Q&A", page_icon="\U0001F4C4", layout="centered")

st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; }

    .app-header {
        text-align: center;
        padding: 1.2rem 0 0.4rem 0;
    }
    .app-header h1 {
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0.1rem;
        background: linear-gradient(90deg, #34d399, #22d3ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .app-header p {
        color: #9ca3af;
        font-size: 0.95rem;
        margin-top: 0;
    }

    .metric-card {
        background: #161b22;
        border: 1px solid #2d333b;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        text-align: center;
    }
    .metric-card .value { font-size: 1.4rem; font-weight: 700; color: #e5e7eb; }
    .metric-card .label { font-size: 0.75rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.04em; }

    .tech-badges { text-align: center; margin-top: 1.5rem; }
    .tech-badge {
        display: inline-block;
        background: #161b22;
        border: 1px solid #2d333b;
        color: #9ca3af;
        border-radius: 999px;
        padding: 0.2rem 0.7rem;
        font-size: 0.72rem;
        margin: 0.15rem;
    }

    div[data-testid="stChatInput"] textarea { border-radius: 10px; }

    .app-footer {
        text-align: center;
        margin-top: 2.5rem;
        padding-top: 1rem;
        border-top: 1px solid #2d333b;
        color: #6b7280;
        font-size: 0.8rem;
    }
    .app-footer span {
        background: linear-gradient(90deg, #34d399, #22d3ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ---------------- Sidebar ----------------

with st.sidebar:
    st.markdown("### \U0001F4C4 PDF Q&A")
    st.caption("Ask questions about any text-based PDF, grounded in its actual content via RAG.")

    with st.expander("\u2699\uFE0F How it works", expanded=False):
        st.markdown(
            "1. **Load** — extract text from your PDF\n"
            "2. **Split** — break it into overlapping chunks\n"
            "3. **Embed** — convert chunks into vectors\n"
            "4. **Retrieve** — find the chunks closest to your question\n"
            "5. **Answer** — an LLM responds using only those chunks"
        )

    if st.session_state.pdf_processed:
        st.divider()
        st.markdown(f"**{len(st.session_state.file_names)} file(s) loaded:**")
        for name in st.session_state.file_names:
            st.caption(f"\U0001F4C4 {name}")
        c1, c2 = st.columns(2)
        c1.metric("Pages", st.session_state.num_pages)
        c2.metric("Chunks", st.session_state.num_chunks)
        st.divider()
        if st.button("\U0001F5D1\uFE0F  Start over with new PDFs", use_container_width=True):
            reset_session()
            st.rerun()


# ---------------- Header ----------------

st.markdown(
    """
    <div class="app-header">
        <h1>PDF Question Answering</h1>
        <p>Upload one or more PDFs, ask anything, get answers grounded in your documents — not guesses.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not os.environ.get("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is not set. Add it to your .env file and restart the app.")
    st.stop()


# ---------------- Upload + processing ----------------

if not st.session_state.pdf_processed:
    uploaded_files = st.file_uploader(
        "Upload PDF(s)",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        status_box = st.status(f"Processing {len(uploaded_files)} PDF(s)...", expanded=True)
        try:
            with status_box:
                all_documents = []
                for uf in uploaded_files:
                    st.write(f"\U0001F4C2 Loading **{uf.name}**...")
                    docs = load_pdf(uf)
                    if len(docs) == 0 or all(d.page_content.strip() == "" for d in docs):
                        status_box.update(label="Failed", state="error")
                        st.error(f"**{uf.name}** has no extractable text (likely a scanned image). Remove it and try again.")
                        st.stop()
                    all_documents.extend(docs)
                time.sleep(0.2)

                st.write(f"\u2702\uFE0F Splitting {len(all_documents)} page(s) across {len(uploaded_files)} file(s) into chunks...")
                chunks = split_documents(all_documents)
                time.sleep(0.2)

                st.write(f"\U0001F9E0 Embedding {len(chunks)} chunk(s)...")
                vector_store = create_vector_store(chunks)
                time.sleep(0.2)

                st.write("\U0001F517 Building retriever and RAG chain...")
                chain, retriever, answer_chain = create_rag_chain(vector_store, num_files=len(uploaded_files))

                status_box.update(label="PDFs ready \u2705", state="complete", expanded=False)

            st.session_state.rag_chain = chain
            st.session_state.retriever = retriever
            st.session_state.answer_chain = answer_chain
            st.session_state.all_chunks = chunks
            st.session_state.pdf_processed = True
            st.session_state.file_names = [uf.name for uf in uploaded_files]
            st.session_state.num_pages = len(all_documents)
            st.session_state.num_chunks = len(chunks)
            st.rerun()

        except Exception as e:
            status_box.update(label="Failed", state="error")
            st.error(f"Failed to process these PDFs: {e}")
            st.stop()
    else:
        st.markdown(
            """
            <div class="tech-badges">
                <span class="tech-badge">LangChain</span>
                <span class="tech-badge">FAISS</span>
                <span class="tech-badge">HuggingFace Embeddings</span>
                <span class="tech-badge">Groq LLM</span>
                <span class="tech-badge">RAG</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------- Chat ----------------

else:
    m1, m2, m3, m4 = st.columns(4)
    for col, value, label in (
        (m1, len(st.session_state.file_names), "Files"),
        (m2, st.session_state.num_pages, "Pages"),
        (m3, st.session_state.num_chunks, "Chunks"),
        (m4, len(st.session_state.chat_history), "Questions asked"),
    ):
        col.markdown(
            f'<div class="metric-card"><div class="value">{value}</div>'
            f'<div class="label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")

    def handle_question(q: str):
        with st.chat_message("user", avatar="\U0001F9D1"):
            st.write(q)

        with st.chat_message("assistant", avatar="\U0001F4C4"):
            with st.spinner("Searching the document and generating an answer..."):
                try:
                    answer, sources = answer_question(
                        st.session_state.rag_chain, st.session_state.retriever,
                        st.session_state.answer_chain, st.session_state.all_chunks, q
                    )
                    st.write(answer)
                    with st.expander(f"\U0001F4CE Show {len(sources)} retrieved source chunk(s)"):
                        for i, doc in enumerate(sources):
                            src = doc.metadata.get("source", "document")
                            st.markdown(f"**Chunk {i + 1}** \u00b7 {src} \u00b7 page {doc.metadata.get('page')}")
                            st.caption(doc.page_content[:500])
                    st.session_state.chat_history.append(
                        {"question": q, "answer": answer, "sources": sources}
                    )
                except Exception as e:
                    error_msg = classify_llm_error(e)
                    st.error(error_msg)
                    st.session_state.chat_history.append(
                        {"question": q, "answer": f"\u26A0\uFE0F {error_msg}", "sources": None}
                    )

    if not st.session_state.chat_history:
        st.caption("Try asking:")
        cols = st.columns(len(SUGGESTED_QUESTIONS))
        for col, sq in zip(cols, SUGGESTED_QUESTIONS):
            if col.button(sq, use_container_width=True):
                st.session_state.pending_question = sq

    for turn in st.session_state.chat_history:
        with st.chat_message("user", avatar="\U0001F9D1"):
            st.write(turn["question"])
        with st.chat_message("assistant", avatar="\U0001F4C4"):
            st.write(turn["answer"])
            if turn.get("sources"):
                with st.expander(f"\U0001F4CE Show {len(turn['sources'])} retrieved source chunk(s)"):
                    for i, doc in enumerate(turn["sources"]):
                        src = doc.metadata.get("source", "document")
                        st.markdown(f"**Chunk {i + 1}** \u00b7 {src} \u00b7 page {doc.metadata.get('page')}")
                        st.caption(doc.page_content[:500])

    if st.session_state.pending_question:
        q = st.session_state.pending_question
        st.session_state.pending_question = None
        handle_question(q)

    typed_question = st.chat_input("Ask a question about the PDF...")
    if typed_question is not None:
        if not typed_question.strip():
            st.warning("Please enter a question before submitting.")
        else:
            handle_question(typed_question)


