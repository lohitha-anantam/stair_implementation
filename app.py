import os
import streamlit as st
from dotenv import load_dotenv

from stair_rag import (
    process_pdf,
    build_index,
    retrieve,
    generate_answer,
    get_openrouter_client,
    get_toc_preview,
)

load_dotenv()

st.set_page_config(
    page_title="STAIR-RAG | Structure-Aware PDF Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 STAIR-RAG: Structure-Aware PDF Assistant")
st.caption(
    "A STAIR-inspired RAG application: document hierarchy → section selection "
    "→ hybrid retrieval → grounded answers with PDF page references."
)

with st.sidebar:
    st.header("Settings")

    try:
        default_api_key = st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        default_api_key = os.getenv("OPENROUTER_API_KEY", "")

    try:
        default_model = st.secrets["OPENROUTER_MODEL"]
    except Exception:
        default_model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

    api_key = st.text_input(
        "OpenRouter API key",
        value=default_api_key,
        type="password",
        help="Use an OpenRouter API key. For deployment, configure Streamlit Secrets.",
    )
    model_name = st.text_input(
        "OpenRouter model ID",
        value=default_model,
        help="Example: openai/gpt-4o-mini",
    )
    top_k = st.slider("Evidence chunks", min_value=2, max_value=10, value=5)
    use_reranker = st.checkbox("Use cross-encoder reranking", value=False)
    strict_mode = st.checkbox("Strict evidence mode", value=True)
    st.divider()
    st.caption(
        "For deployment, add OPENROUTER_API_KEY and OPENROUTER_MODEL "
        "under App → Settings → Secrets."
    )

if "processed" not in st.session_state:
    st.session_state.processed = None
if "index" not in st.session_state:
    st.session_state.index = None
if "filename" not in st.session_state:
    st.session_state.filename = None
if "history" not in st.session_state:
    st.session_state.history = []

uploaded = st.file_uploader("Upload a PDF document", type=["pdf"])

col1, col2 = st.columns([1, 2])
with col1:
    process_clicked = st.button(
        "Process PDF and build index",
        type="primary",
        disabled=uploaded is None,
        use_container_width=True,
    )
with col2:
    if st.session_state.filename:
        st.success(f"Loaded document: {st.session_state.filename}")

if process_clicked and uploaded is not None:
    with st.spinner(
        "Extracting pages, identifying document structure, and building the retrieval index..."
    ):
        try:
            pdf_bytes = uploaded.getvalue()
            processed = process_pdf(pdf_bytes)

            # Attach the uploaded filename before building the index.
            # Keep this loop INSIDE the try block.
            for chunk in processed["chunks"]:
                chunk["document"] = uploaded.name

            index = build_index(processed["chunks"])

            st.session_state.processed = processed
            st.session_state.index = index
            st.session_state.filename = uploaded.name
            st.session_state.history = []
            st.success(
                f"Processed {processed['page_count']} pages and created "
                f"{len(processed['chunks'])} searchable chunks."
            )
        except Exception as exc:
            st.error(f"Could not process this PDF: {exc}")
            st.exception(exc)

if st.session_state.processed:
    processed = st.session_state.processed

    with st.expander("Document structure / extracted table of contents", expanded=False):
        st.write(f"**Document:** {st.session_state.filename}")
        st.write(f"**Pages:** {processed['page_count']}")
        st.write(f"**Detected sections:** {len(processed['sections'])}")
        toc_text = get_toc_preview(processed["sections"])
        st.code(
            toc_text
            or "No clear headings detected. Retrieval will use page-level chunks."
        )
        if processed.get("toc_source"):
            st.caption(f"Structure source: {processed['toc_source']}")

    st.subheader("Ask a question about your document")
    question = st.text_area(
        "Question",
        placeholder="Example: Explain how the proposed method reduces hallucinations.",
        height=90,
    )
    ask = st.button("Generate answer", type="primary", disabled=not question.strip())

    if ask:
        if not api_key.strip():
            st.warning(
                "Enter your OpenRouter API key in the sidebar, or configure "
                "OPENROUTER_API_KEY in Streamlit Secrets."
            )
        elif not model_name.strip():
            st.warning("Enter a valid OpenRouter model ID in the sidebar.")
        else:
            try:
                client = get_openrouter_client(api_key.strip())
                with st.spinner(
                    "Selecting relevant sections, retrieving evidence, and generating your answer..."
                ):
                    results, selected_sections, selection_method = retrieve(
                        question=question,
                        processed=processed,
                        index=st.session_state.index,
                        client=client,
                        model_name=model_name.strip(),
                        top_k=top_k,
                        use_reranker=use_reranker,
                    )
                    answer = generate_answer(
                        question=question,
                        results=results,
                        client=client,
                        model_name=model_name.strip(),
                        strict_mode=strict_mode,
                    )

                st.session_state.history.insert(
                    0,
                    {
                        "question": question,
                        "answer": answer,
                        "results": results,
                        "sections": selected_sections,
                        "selection_method": selection_method,
                    },
                )
            except Exception as exc:
                st.error(f"Question answering failed: {exc}")
                st.exception(exc)

    for item in st.session_state.history:
        with st.container(border=True):
            st.markdown(f"**Question:** {item['question']}")
            st.markdown("### Answer")
            st.markdown(item["answer"])

            # Show a concise, deduplicated source list. The retrieved chunk text
            # remains internal context and is not displayed to the user.
            references = []
            seen = set()
            for result in item["results"]:
                document = result.get("document", st.session_state.filename or "Uploaded PDF")
                page = result.get("page", "unknown")
                key = (document, page)
                if key not in seen:
                    seen.add(key)
                    references.append(key)

            if references:
                st.markdown("### Source pages")
                for document, page in references:
                    st.write(f"📄 **{document}** — PDF page {page}")

            with st.expander("Retrieval details", expanded=False):
                st.caption(
                    "Selected sections: "
                    + (" | ".join(item["sections"]) if item["sections"] else "None")
                    + f" · Selection method: {item['selection_method']}"
                )
                st.caption(
                    f"Retrieved {len(item['results'])} evidence chunks. "
                    "Raw chunk text is hidden; it was passed to the LLM as context."
                )

st.divider()
st.caption(
    "Note: This is a practical STAIR-inspired implementation, not an exact reproduction "
    "of the research paper's Mistral/LoRA training pipeline. Scanned PDFs may require OCR."
)
