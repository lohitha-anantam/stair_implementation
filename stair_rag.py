import io
import re
import json
import math
from collections import Counter, defaultdict

import fitz
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import faiss
except Exception:
    faiss = None

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

_EMBEDDER = None


def get_embedder():
    """Load the embedding model once per Python process."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _EMBEDDER


def get_openrouter_client(api_key):
    """Create an OpenRouter client using its OpenAI-compatible API."""
    from openai import OpenAI
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={"X-OpenRouter-Title": "STAIR-RAG"}
    )


def call_llm(client, model_name, prompt, json_mode=False, temperature=0.0):
    """Call an OpenRouter chat model and return its text response."""
    kwargs = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def clean_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_paragraphs(text):
    parts = re.split(r"\n\s*\n", text)
    return [clean_text(p) for p in parts if len(clean_text(p)) > 35]


def extract_bookmarks(doc):
    """Return PDF outline/bookmark entries, if the PDF contains them."""
    try:
        toc = doc.get_toc(simple=True)
    except Exception:
        toc = []
    entries = []
    for row in toc:
        if len(row) >= 3:
            level, title, page = row[:3]
            title = clean_text(str(title))
            if title:
                entries.append({
                    "level": max(1, int(level)),
                    "title": title,
                    "page": max(1, int(page))
                })
    return entries


HEADING_PATTERNS = [
    re.compile(r"^\s*(chapter|unit|part)\s+[\wIVX.-]+[:.\- ]?.{0,120}$", re.I),
    re.compile(r"^\s*\d+(?:\.\d+){0,4}\s+[A-Z][^\n]{2,120}$"),
    re.compile(r"^\s*[A-Z][A-Z0-9 ,:&()'/-]{5,90}$")
]


def detect_headings(page_text, page_num):
    """Conservative fallback heading detector for PDFs without bookmarks."""
    found = []
    lines = [clean_text(x) for x in page_text.splitlines()]
    lines = [x for x in lines if x and len(x) <= 140]
    for line in lines:
        if any(p.match(line) for p in HEADING_PATTERNS):
            # Avoid treating ordinary all-caps short labels as headings.
            if len(line.split()) <= 16:
                found.append({
                    "level": 1 if re.match(r"^(chapter|unit|part)\b", line, re.I) else 2,
                    "title": line,
                    "page": page_num
                })
    return found[:8]


def normalize_section_title(title):
    return re.sub(r"\s+", " ", title).strip()


def make_sections(bookmarks, page_headings, page_count):
    """Create a flat section catalogue with page ranges and hierarchical paths."""
    source = "PDF bookmarks" if bookmarks else "heuristic heading detection"
    entries = bookmarks if bookmarks else page_headings

    # Remove duplicate title/page entries while preserving order.
    deduped = []
    seen = set()
    for item in sorted(entries, key=lambda x: (x["page"], x["level"])):
        key = (normalize_section_title(item["title"]).lower(), item["page"])
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    if not deduped:
        return [{
            "id": "S000",
            "title": "Entire document",
            "path": "Entire document",
            "level": 1,
            "start_page": 1,
            "end_page": page_count
        }], "No explicit ToC detected; whole-document fallback"

    stack = []
    sections = []
    for i, item in enumerate(deduped):
        level = max(1, min(6, int(item["level"])))
        while stack and stack[-1]["level"] >= level:
            stack.pop()
        stack.append({"level": level, "title": item["title"]})
        path = " > ".join(x["title"] for x in stack)
        next_page = page_count
        for nxt in deduped[i + 1:]:
            if nxt["page"] > item["page"]:
                next_page = nxt["page"] - 1
                break
        sections.append({
            "id": f"S{i:03d}",
            "title": item["title"],
            "path": path,
            "level": level,
            "start_page": max(1, item["page"]),
            "end_page": max(item["page"], next_page)
        })

    # If the outline has only a few entries, preserve them as broad sections.
    return sections, source


def section_for_page(sections, page):
    candidates = [
        s for s in sections
        if s["start_page"] <= page <= s["end_page"]
    ]
    if not candidates:
        return sections[0]
    # Prefer the deepest matching section.
    return max(candidates, key=lambda s: (s["level"], s["start_page"]))


def chunk_text(text, max_chars=1200, overlap_chars=180):
    """Simple paragraph-aware chunker; overlap retains nearby context."""
    paragraphs = split_paragraphs(text)
    if not paragraphs and text.strip():
        paragraphs = [clean_text(text)]

    chunks = []
    current = ""
    for para in paragraphs:
        if len(para) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            start = 0
            while start < len(para):
                piece = para[start:start + max_chars]
                if piece.strip():
                    chunks.append(piece.strip())
                start += max_chars - overlap_chars
            continue

        if not current:
            current = para
        elif len(current) + len(para) + 2 <= max_chars:
            current += "\n\n" + para
        else:
            chunks.append(current)
            tail = current[-overlap_chars:] if overlap_chars else ""
            current = (tail + "\n\n" + para).strip()

    if current:
        chunks.append(current)
    return [c for c in chunks if len(c.strip()) > 40]


def process_pdf(pdf_bytes):
    """Extract page text, detect document structure, and create metadata-rich chunks."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_count = len(doc)
    if page_count == 0:
        raise ValueError("The uploaded PDF has no pages.")

    bookmarks = extract_bookmarks(doc)
    page_headings = []
    page_texts = []

    for page_index, page in enumerate(doc):
        page_num = page_index + 1
        text = clean_text(page.get_text("text", sort=True))
        page_texts.append((page_num, text))
        if not bookmarks:
            page_headings.extend(detect_headings(text, page_num))

    sections, toc_source = make_sections(bookmarks, page_headings, page_count)

    chunks = []
    for page_num, text in page_texts:
        if not text:
            continue
        section = section_for_page(sections, page_num)
        page_chunks = chunk_text(text)
        for chunk_index, chunk in enumerate(page_chunks):
            chunks.append({
                "id": f"P{page_num:04d}_C{chunk_index:03d}",
                "text": chunk,
                "page": page_num,
                "section_id": section["id"],
                "section": section["title"],
                "section_path": section["path"],
                "section_start": section["start_page"],
                "section_end": section["end_page"]
            })

    if not chunks:
        raise ValueError(
            "No selectable text was found. This may be a scanned PDF; run OCR first."
        )

    return {
        "page_count": page_count,
        "sections": sections,
        "chunks": chunks,
        "toc_source": toc_source
    }


def get_toc_preview(sections, max_items=100):
    lines = []
    for s in sections[:max_items]:
        indent = "  " * max(0, min(s["level"] - 1, 5))
        lines.append(
            f"{indent}- {s['title']} "
            f"(pages {s['start_page']}-{s['end_page']})"
        )
    return "\n".join(lines)


def build_index(chunks):
    if not chunks:
        raise ValueError("There are no chunks to index.")
    embedder = get_embedder()
    texts = [c["text"] for c in chunks]
    vectors = embedder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    ).astype("float32")

    faiss_index = None
    if faiss is not None:
        faiss_index = faiss.IndexFlatIP(vectors.shape[1])
        faiss_index.add(vectors)

    tokenized = [re.findall(r"\w+", t.lower()) for t in texts]
    bm25 = BM25Okapi(tokenized) if BM25Okapi is not None else None

    return {
        "vectors": vectors,
        "faiss": faiss_index,
        "bm25": bm25,
        "tokenized": tokenized
    }


def section_candidates(question, processed, client=None, model_name="gemini-2.5-flash", limit=3):
    sections = processed["sections"]
    if len(sections) <= limit:
        return sections, "all sections (small document)"

    # Ask OpenRouter to choose only from known section IDs. Fallback to embedding similarity.
    if client is not None:
        catalogue = [
            {"id": s["id"], "path": s["path"], "pages": [s["start_page"], s["end_page"]]}
            for s in sections
        ]
        prompt = f"""
You are a document section router. Choose up to {limit} sections that are most likely
to contain the answer to the user's question. Use ONLY IDs in the supplied catalogue.
Return valid JSON only in this exact shape:
{{"section_ids": ["S001", "S002"]}}
If no section is relevant, return an empty list.

Question:
{question}

Section catalogue:
{json.dumps(catalogue, ensure_ascii=False)}
"""
        try:
            response_text = call_llm(
                client, model_name, prompt, json_mode=True, temperature=0
            )
            data = json.loads(response_text)
            valid = set(s["id"] for s in sections)
            ids = [x for x in data.get("section_ids", []) if x in valid]
            chosen = [s for s in sections if s["id"] in ids][:limit]
            if chosen:
                return chosen, "OpenRouter ToC section routing"
        except Exception:
            pass

    embedder = get_embedder()
    qvec = embedder.encode(
        [question], convert_to_numpy=True, normalize_embeddings=True
    )[0]
    section_texts = [s["path"] for s in sections]
    svecs = embedder.encode(
        section_texts, convert_to_numpy=True, normalize_embeddings=True
    )
    scores = np.dot(svecs, qvec)
    indices = np.argsort(scores)[::-1][:limit]
    return [sections[int(i)] for i in indices], "embedding-based section routing fallback"


def retrieve(question, processed, index, client=None, model_name="gemini-2.5-flash",
             top_k=5, use_reranker=False):
    chunks = processed["chunks"]
    chosen_sections, selection_method = section_candidates(
        question, processed, client, model_name
    )
    chosen_ids = {s["id"] for s in chosen_sections}
    allowed = [
        i for i, c in enumerate(chunks)
        if c["section_id"] in chosen_ids
    ]

    # If section routing is too restrictive or has no matching chunks, broaden safely.
    if len(allowed) < min(3, len(chunks)):
        allowed = list(range(len(chunks)))

    embedder = get_embedder()
    qvec = embedder.encode(
        [question], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")[0]

    semantic_scores = np.dot(index["vectors"][allowed], qvec)
    semantic_rank = {
        allowed[int(pos)]: rank
        for rank, pos in enumerate(np.argsort(semantic_scores)[::-1], start=1)
    }

    bm25_scores = None
    if index["bm25"] is not None:
        all_bm25 = index["bm25"].get_scores(re.findall(r"\w+", question.lower()))
        bm25_scores = np.asarray([all_bm25[i] for i in allowed], dtype="float32")
        bm25_rank = {
            allowed[int(pos)]: rank
            for rank, pos in enumerate(np.argsort(bm25_scores)[::-1], start=1)
        }
    else:
        bm25_rank = {}

    # Reciprocal Rank Fusion combines dense and lexical rankings.
    fused = []
    for idx in allowed:
        score = 1.0 / (60 + semantic_rank.get(idx, 9999))
        if bm25_rank:
            score += 1.0 / (60 + bm25_rank.get(idx, 9999))
        fused.append((idx, score))

    fused.sort(key=lambda x: x[1], reverse=True)
    selected = []
    for idx, score in fused[:max(top_k * 4, top_k)]:
        item = dict(chunks[idx])
        item["score"] = float(score)
        sources = []
        if idx in semantic_rank:
            sources.append("semantic")
        if idx in bm25_rank:
            sources.append("BM25")
        item["retrieval_sources"] = " + ".join(sources)
        selected.append(item)

    if use_reranker and selected:
        try:
            from sentence_transformers import CrossEncoder
            reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            pairs = [[question, item["text"]] for item in selected]
            scores = reranker.predict(pairs)
            for item, score in zip(selected, scores):
                item["rerank_score"] = float(score)
            selected.sort(key=lambda x: x["rerank_score"], reverse=True)
        except Exception as exc:
            # Keep the hybrid ranking if the optional reranker cannot load.
            selected = selected

    return selected[:top_k], [s["path"] for s in chosen_sections], selection_method


def generate_answer(
    question,
    results,
    client,
    model_name="openai/gpt-4o-mini",
    strict_mode=True
):
    """
    Generate one user-facing answer from retrieved evidence.
    Raw chunks are supplied privately as context; the caller should display
    only this answer and the source list, not the raw chunk text.
    """
    if not results:
        return (
            "I couldn't find relevant evidence in the uploaded documents. "
            "Try rephrasing the question or checking whether the information "
            "is present in the PDFs."
        )

    evidence = []
    for i, item in enumerate(results, start=1):
        document = item.get("document", "Uploaded PDF")
        page = item.get("page", "unknown")
        section = item.get("section_path", item.get("section", "Unspecified section"))
        evidence.append(
            f"[Evidence {i} | Document: {document} | PDF page: {page} | "
            f"Section: {section}]\\n{item['text']}"
        )

    context = "\\n\\n".join(evidence)

    if strict_mode:
        grounding_rule = (
            "Use only the evidence below. If it does not contain enough "
            "information to answer, explicitly say that the uploaded documents "
            "do not provide sufficient information. Do not fill gaps from "
            "general knowledge."
        )
    else:
        grounding_rule = (
            "Prioritize the evidence below. If you make an inference, label it "
            "clearly. Never invent facts or citations."
        )

    prompt = f"""
You are a precise document question-answering assistant.

USER QUESTION:
{question}

RETRIEVED EVIDENCE (internal context; do not reproduce it as raw chunks):
{context}

TASK:
Write a direct, complete, well-organized answer to the user's question.
Synthesize information across the relevant evidence passages. Do not tell the
user that you are showing or retrieving chunks. Do not output the evidence
block or a chunk-by-chunk summary.

GROUNDING RULE:
{grounding_rule}

CITATION RULES:
- Cite every important factual claim using [Document name, PDF page N].
- Use only document names and page numbers supplied in the evidence.
- Do not invent or guess page numbers.
- If multiple documents support a point, cite each relevant source.
- If evidence is contradictory, explain the conflict and cite both sources.
- If the answer is not supported, say so clearly.

OUTPUT:
Return only the answer in natural language, with citations included inline.
"""

    return call_llm(
        client,
        model_name,
        prompt,
        json_mode=False,
        temperature=0.2
    )
