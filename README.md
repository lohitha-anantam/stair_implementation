# STAIR-RAG PDF Assistant (OpenRouter)

A Streamlit application that extracts text from a PDF, identifies document sections, creates semantic and BM25 retrieval indexes, retrieves relevant evidence, and asks an OpenRouter chat model to generate a grounded answer with PDF page citations.

## 1. Project files

- `app.py` — Streamlit user interface.
- `stair_rag.py` — PDF extraction, section detection, indexing, hybrid retrieval, and answer generation.
- `requirements.txt` — Python dependencies.
- `.env.example` — local environment-variable template.
- `.streamlit/secrets.toml.example` — Streamlit Cloud secrets template.

## 2. Run locally

Use Python 3.10 or 3.11 if possible.

```bash
python -m venv .venv
```

Activate the environment:

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Create a `.env` file by copying `.env.example`, then add your OpenRouter API key:

```env
OPENROUTER_API_KEY=your_actual_key
OPENROUTER_MODEL=openai/gpt-4o-mini
```

Start the app:
```bash
streamlit run app.py
```

## 3. Deploy on Streamlit Community Cloud

1. Upload all project files to the same GitHub repository folder.
2. In Streamlit Community Cloud, create or open the app and set the main file path to `app.py`.
3. Open **App settings → Secrets** and add:

```toml
OPENROUTER_API_KEY = "your_actual_openrouter_key"
OPENROUTER_MODEL = "openai/gpt-4o-mini"
```

4. Save and allow the app to redeploy.

Do not commit your real API key to GitHub. Keep it in Streamlit Secrets or a local `.env` file that is excluded by `.gitignore`.

## 4. How to use

1. Open the app.
2. Upload one text-based PDF.
3. Click **Process PDF and build index**.
4. Enter a question and click **Generate answer**.
5. Read the generated answer and its source PDF page references.
6. Expand **Retrieval details** only if you want to inspect the selected sections and retrieval process.

Retrieved chunks are passed to the language model internally. The default interface does not display the raw chunk text.

## 5. Notes and limitations

- This version accepts one PDF at a time.
- Scanned/image-only PDFs need OCR before processing. The app reports an error if no selectable text is found.
- The first run downloads the `sentence-transformers/all-MiniLM-L6-v2` embedding model, so startup may take longer.
- The cross-encoder reranker is optional and may download an additional model when enabled.
- OpenRouter model availability, pricing, and rate limits depend on the selected provider/model and your OpenRouter account.
- This is a practical STAIR-inspired RAG implementation, not an exact reproduction of a research paper's training pipeline.
