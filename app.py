    st.subheader("Ask a question about your document")
    question = st.text_area(
        "Question",
        placeholder="Example: Explain how the proposed method reduces hallucinations.",
        height=90,
    )
    ask = st.button(
        "Generate answer", type="primary", disabled=not question.strip()
    )

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
                        "filename": st.session_state.filename,
                    },
                )
            except Exception as exc:
                st.error(f"Question answering failed: {exc}")
                st.exception(exc)

    # Show newest question and answer first, followed by references and chunks.
    for item in st.session_state.history:
        with st.container(border=True):
            st.markdown(f"**Question:** {item['question']}")

            st.markdown("### Answer")
            st.markdown(item["answer"])

            # Build a deduplicated list of source-document/page pairs.
            references = []
            seen_references = set()
            for result in item["results"]:
                document = result.get("document", item.get("filename", "Uploaded PDF"))
                page = result.get("page", "unknown")
                reference = (document, page)
                if reference not in seen_references:
                    seen_references.add(reference)
                    references.append(reference)

            if references:
                st.markdown("### Source pages")
                for document, page in references:
                    st.write(f"📄 **{document}** — PDF page {page}")

            with st.expander("Retrieval details", expanded=False):
                selected = item.get("sections") or []
                st.caption(
                    "Selected sections: "
                    + (" | ".join(selected) if selected else "None")
                    + f" · Selection method: {item.get('selection_method', 'N/A')}"
                )
                st.write(f"Retrieved {len(item['results'])} evidence chunks.")

            # Relevant retrieved chunks are shown at the bottom, on demand.
            with st.expander(
                f"View relevant retrieved chunks ({len(item['results'])})",
                expanded=False,
            ):
                if not item["results"]:
                    st.info("No relevant chunks were retrieved for this question.")
                else:
                    for i, result in enumerate(item["results"], start=1):
                        section = result.get("section", "Unknown section")
                        page = result.get("page", "Unknown")
                        document = result.get(
                            "document", item.get("filename", "Uploaded PDF")
                        )

                        st.markdown(f"#### Chunk {i}: {section}")
                        st.caption(f"Source: {document} · PDF page: {page}")
                        st.write(result.get("text", "No chunk text available."))

                        score = result.get("score")
                        if isinstance(score, (int, float)):
                            score_text = f"{score:.4f}"
                        else:
                            score_text = "N/A"
                        st.caption(
                            f"Hybrid score: {score_text} · Retrieval: "
                            f"{result.get('retrieval_sources', 'N/A')}"
                        )
                        if i < len(item["results"]):
                            st.divider()

st.divider()
st.caption(
    "Note: This is a practical STAIR-inspired implementation, not an exact reproduction "
    "of the research paper's Mistral/LoRA training pipeline. Scanned PDFs may require OCR."
)
