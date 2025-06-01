# Databricks Certified Generative AI Engineer Associate – Study Notes

## 🧱 Section 1: Design Applications
*How to architect GenAI solutions from use-case to execution flow*

---

### 1️⃣ Prompt Design Rules

- **For structured output** → Use few-shot prompting with clearly labeled examples.  
  Example:  
  `"Respond in JSON with keys ‘summary’, ‘sentiment’, and ‘entities’."`

- **For classification tasks** → Use zero-shot prompts with explicit classes:  
  `"Classify as one of: positive, neutral, negative."`

- **To elicit formatted output** → Include formatting instructions **after** core context.  
  Models prioritize recent tokens → put format instructions last.

- **Use system prompts** to establish tone/persona:  
  `"You are a financial assistant who responds concisely in bullet points."`

---

### 2️⃣ Task-to-Model Mapping

| Business Need                          | Model Task                  | Preferred Approach                        |
|---------------------------------------|-----------------------------|-------------------------------------------|
| Text summarization                    | Summarization               | Prompt LLM with extractive/abstractive instructions |
| Structured data extraction (PDF/HTML) | Information Extraction      | Retrieval + field-specific prompt         |
| Classification                        | Zero-/Few-shot Classification | Prompt with example-labeled outputs     |
| Q&A over documents                    | Retrieval-Augmented Generation (RAG) | Embed + retrieve + prompt       |
| Multi-step reasoning or action        | Agent                        | Tool selection + LLM + memory             |

> 🔑 **Default**: Use **retrieval + prompting** before considering fine-tuning.

---

### 3️⃣ Choosing Chain Components

**Input-based decisions:**
- Raw text → Embed + index in Vector DB
- PDF/HTML → Chunk + tag with metadata
- SQL or structured input → Function calling

**Output-based decisions:**
- Summary → Summarization chain
- Grounded answer → RAG (Retriever + Generator)
- Action (e.g. booking) → Agent with tools

---

### 4️⃣ Translating Business Goals into Pipeline Design

Use this mental checklist:

- ✅ What is the **goal**?  
- ✅ What **input format** is expected?  
- ✅ What **output format** is required?  
- ✅ Is **factual accuracy** required?

**Example:**

> *"Generate ticket replies based on internal KB."*

- Input: ticket text  
- Output: concise email response  
- Pipeline:  
  - Chunk & embed KB  
  - Vector search relevant chunks  
  - Prompt LLM with user ticket + top docs  
  - Format output in helpdesk response style

---

### 5️⃣ Multi-Stage Reasoning and Tool Use

- **Tools** = APIs or functions (e.g. calculator, search, lookup)
- **Agents** = LLMs that plan tool use across steps
- **Chain-of-thought prompting** → Break complex reasoning into steps

**Example Tool Chain:**

1. Extract keywords from user question  
2. Use keywords to search KB  
3. Summarize retrieved docs  
4. Generate structured answer or invoke function

> 🔁 RAG = static retrieval → generate.  
> 🧠 Agents = dynamic reasoning → tool → loop.

---

### 🎯 Key Rules of Thumb

- Use RAG before fine-tuning → cheaper, safer, faster.
- Use system prompts to control tone and intent.
- Favor few-shot examples for structured output.
- Chain components based on **input/output**, not tool familiarity.
- Multi-stage reasoning? Consider **agents + tools**.

---

## 🧾 Section 2: Data Preparation
*How to structure, clean, chunk, and persist content for retrieval-augmented generation (RAG) pipelines*

---

### 1️⃣ Chunking Strategies

- **Default to semantic chunking** → Split on paragraph/topic boundaries, not fixed size.
  - Better semantic integrity → improves relevance of retrieved chunks.
  - Avoid mid-sentence or mid-thought cuts → lowers grounding quality.

- **Use overlapping sliding windows** for dense information (e.g. scientific docs).
  - Ensures context continuity even when chunks are small.

- **Adjust chunk size to model context window**:
  - Small model (4k tokens) → 256–512 token chunks
  - Larger model (32k) → 1k–2k token chunks

- **Chunk metadata**:
  - Always attach `source`, `page_number`, `section_title`, etc. for filtering and reranking.

---

### 2️⃣ Filtering and Cleaning Documents

- **Remove noise before chunking**:
  - Ads, headers, footers, table of contents, legal disclaimers, repeated boilerplate.

- **Normalize input text**:
  - Fix broken line breaks, remove HTML tags, decode special characters.

- **Avoid over-including irrelevant sections**:
  - E.g., don’t embed entire manuals or books — focus on FAQs, high-quality docs, curated knowledge bases.

- **Prioritize quality over quantity**:
  - More chunks ≠ better retrieval — irrelevant docs pollute results.

---

### 3️⃣ Python Packages for Document Parsing

| Format        | Recommended Package           |
|---------------|-------------------------------|
| PDF           | `PyMuPDF` (fitz), `pdfplumber` |
| Word (DOCX)   | `python-docx`                 |
| HTML          | `BeautifulSoup`               |
| Markdown      | `markdown` + `html2text`      |
| Scanned PDFs  | `pytesseract` (OCR required)  |

> 🔑 Use parsers that preserve layout and sectioning where possible.

---

### 4️⃣ Writing Chunks to Delta Lake in Unity Catalog

- Store output as **structured tables** for governance and traceability:
  - Columns: `chunk_text`, `embedding_vector`, `document_id`, `metadata`

- Sequence:
  1. Extract → Clean → Chunk → Embed
  2. Write to Delta table (`spark.write.format("delta")`)
  3. Register table in Unity Catalog (`CREATE TABLE ... USING DELTA`)

- Use `AUTOINCREMENT` or UUIDs for chunk ID column
- Maintain lineage by tracking source file paths and timestamps

> Unity Catalog ensures access control and metadata lineage for chunked data.

---

### 5️⃣ Source Document Selection for RAG

- Use **authoritative**, **concise**, and **updated** documents
  - Avoid:
    - Raw logs
    - Social media
    - Long unstructured books or legal docs without filtering

- Favor:
  - Internal knowledge bases
  - FAQs
  - Product documentation
  - Wikis with change control

> Choose sources that align with use-case scope and tone.

---

### 6️⃣ Prompt/Response Pair Selection

- Match prompt types to model capability:
  - Classification → Short input, clear output labels
  - Summarization → Long input, short output
  - Retrieval QA → Grounded context + question → answer

- Filter out:
  - Noisy or inconsistent labels
  - Overly long or ambiguous prompts
  - Hallucinated responses

- Use these pairs for:
  - Evaluation
  - Fine-tuning (if needed)
  - Retrieval-based simulation

---

### 7️⃣ Retrieval System Evaluation & Re-Ranking

- **Use metrics to evaluate retrieval**:
  - Precision@k → % of relevant chunks in top k
  - Recall@k → % of total relevant chunks retrieved
  - NDCG → Rewards good ranking order
  - MRR (Mean Reciprocal Rank) → Focus on getting top-1 right

- **Re-ranking boosts quality**:
  - First stage: vector similarity (fast, dense)
  - Second stage: reranker (cross-encoder LLM or heuristic scoring)

> Re-rankers compare **question + passage together** and rescore.

---

### 🎯 Key Rules of Thumb

- Use semantic chunking with overlaps for best retrieval quality.
- Clean and normalize before embedding — garbage in = garbage out.
- Store chunks in Delta Lake with rich metadata → enables smart filtering + governance.
- Retrieval systems work best with high-quality, targeted source docs.
- Evaluate your retriever using precision/recall and apply reranking if quality is low.

---

## 🛠️ Section 3: Application Development
*Building, evaluating, and securing GenAI apps using prompts, agents, tools, and frameworks*

---

### 1️⃣ Retrieval-Aware Development

- **Create tools to extract data**:
  - Write wrapper functions for search, lookup, or internal APIs
  - Integrate data extraction tools (e.g., OCR, SQL, REST APIs) into RAG pipelines

- **Select chunking strategy based on evaluation**:
  - Poor recall → try smaller chunks or increase overlap
  - Irrelevant results → better metadata filtering or re-ranking

- **Embed with context length in mind**:
  - Long source docs → split into manageable chunks
  - Align chunk size with embedding model’s max token size

---

### 2️⃣ Prompt Engineering in Practice

- **Augment prompts with context**:
  - Dynamically insert fields like `customer_name`, `product_type`, or recent actions
  - Use string templating: `f"Given that the user purchased {product_type}, recommend..."`

- **Adjust output via prompt rewrites**:
  - Original: “Summarize the following.”  
    Improved: “Summarize the following in 2 bullet points for a 5th grader.”

- **Use system prompts for baseline behavior**:
  - `"You are a friendly assistant."` or `"Always respond concisely and include source links."`

- **Write metaprompts to reduce hallucination**:
  - `"Only respond using information from the provided context. If unsure, say 'I don’t know'."`

---

### 3️⃣ Model & Embedding Selection

- **Select LLMs based on attributes**:
  - Open-weight: Llama 2, MPT → good for cost control, internal deployment
  - Proprietary: GPT-4, Claude → strong reasoning, broader context window
  - MosaicML → for custom fine-tuning, hosted models

- **Key model metadata to check**:
  - Context window size  
  - Training dataset transparency  
  - Eval metrics (MMLU, HELM, TruthfulQA)

- **Use cases by type**:
  - Text classification → smaller model or distilled variant
  - Chat/agentic behavior → instruction-tuned, high-context model
  - Tool use / multi-function → model with ReAct or function-calling support

- **Choose embedding model** based on:
  - Query complexity and domain
  - Document length
  - Cost/performance tradeoff

---

### 4️⃣ Frameworks and Chains

- **LangChain or similar tools**:
  - Use for chaining steps: retrieval → transform → generation
  - Supports agents, tools, chat memory, and retrievers

- **Agent Frameworks (LangChain, Semantic Kernel, etc.)**:
  - Enable tool selection at runtime
  - Useful for multi-step tasks: search → analyze → generate → act

- **Agent prompt template** example:

## 🚀 Section 4: Assembling and Deploying Applications
*How to build, register, serve, and secure GenAI applications on the Databricks platform*

---

### 1️⃣ Building and Coding Chains

- **LangChain chains** (common exam focus):
  - Chain = pipeline of components: retriever → prompt → LLM → output parser
  - `SimpleSequentialChain`, `LLMChain`, and `RetrievalQA` are key constructs

- **Pyfunc model chaining (MLflow)**:
  - Wrap pre-processing → LLM call → post-processing as a single `pyfunc` model
  - Register using `mlflow.pyfunc.log_model()` with a custom `PythonModel` class

- **Example use case**:
  - Pre: clean input, lookup metadata  
  - Core: invoke LLM or retriever  
  - Post: validate format, redact PII

> 🔧 Know how to encode a basic retrieval→generate flow either in LangChain or custom pyfunc model.

---

### 2️⃣ Registering and Serving Models

- **Register models in Unity Catalog**:
  - Use `mlflow.register_model()` or via UI
  - Model versioning is automatic
  - Attach signature (`input_schema`, `output_schema`) to ensure compatibility

- **Deployment steps (for RAG app)**:
  1. Ingest and chunk source data
  2. Generate embeddings (e.g., HuggingFace, OpenAI)
  3. Create and persist a vector index
  4. Wrap prompt + retriever into a LangChain or pyfunc chain
  5. Log the model with MLflow
  6. Register model in Unity Catalog
  7. Deploy to Model Serving

- **Control access** using Unity Catalog permissions or API gateway tokens

---

### 3️⃣ Serving Vector Search and LLM APIs

- **Databricks Vector Search**:
  - Fully managed vector index service (dense and hybrid search supported)
  - Backed by Delta tables
  - Use `CREATE VECTOR INDEX` SQL or Python SDK
  - Query via `.query()` or SQL `ai_query()`

- **Vector Index Components**:
  - Source Delta Table with embedded column
  - Metadata for filtering (e.g., tags, doc_id, user_role)
  - Index type (dense or hybrid)

- **Mosaic AI Vector Search**:
  - Similar to Databricks Vector Search
  - Focused on large-scale retrieval + tight LLM integration
  - May appear on exam in name only (know it exists and does what Vector Search does, but scaled)

- **ai_query() for batch inference**:
  - SQL function to embed and search in batch (ideal for offline scoring)
  - `SELECT ai_query('vector_index_name', question_column) AS answer`

---

### 4️⃣ Foundation Model Serving

- **Serving hosted models via Databricks**:
  - GPT-4, Claude, LLaMA 2 via MosaicML or third-party APIs
  - Use `dbx.ChatCompletion.create()` or compatible SDKs
  - Use Databricks Model Serving (managed or serverless) for endpoint hosting

- **Resources required to serve RAG app**:
  - Model artifacts (LLM + retriever)
  - Vector search index
  - Serving cluster (serverless or managed)
  - Unity Catalog for access control
  - Prompt templates or chain config (JSON, LangChain YAML, etc.)

> 🧠 Exam focuses on sequencing these components correctly for deployment.

---

### 🎯 Key Rules of Thumb

- Prefer **LangChain for chaining** unless pyfunc custom logic is required.
- Register all production models in **Unity Catalog** for governance.
- Use **Vector Search + ai_query()** for low-latency embedding-based lookups.
- Sequence = ingest → chunk → embed → index → chain → model registry → endpoint
- Secure model serving with **ACLs or token-based API access**.
- For batch workloads → use `ai_query()` inside SQL or scheduled notebooks.

---

## 🛡️ Section 5: Governance
*Enforcing safety, compliance, and content integrity in Generative AI applications*

---

### 1️⃣ Masking Techniques for Guardrails

- **Why use masking**:
  - Protect sensitive data (e.g., PII) from leaking into prompts or model outputs.
  - Prevent performance degradation from noisy/unstructured inputs.

- **Common masking patterns**:
  - Regex-based scrubbing (e.g., emails, phone numbers)
  - Named Entity Recognition (NER) → auto-detect entities to mask
  - Pre-processing step before chunking or embedding

- **Masking use cases on exam**:
  - Improve retrieval quality by removing user-specific tokens.
  - Prevent prompts from overfitting to irrelevant patterns.

> 🧠 Example: Mask `user_id` before storing in embedding index → improves generalization + avoids leakage.

---

### 2️⃣ Guardrails Against Malicious Input

- **Input filtering**:
  - Reject or flag queries with prompt injection patterns.
  - Block known jailbreak phrases (e.g., “ignore the previous instruction”).

- **System-level prompts**:
  - `"You are an assistant that only responds with factual, safe content."`

- **Rate limiting / abuse detection**:
  - Apply usage quotas or anomaly detection on user inputs

- **Output validation**:
  - Use regex or classifiers to reject dangerous content (e.g., hate speech, threats)

> 🔐 Combine **input filtering**, **system prompts**, and **output sanitization** for full coverage.

---

### 3️⃣ Problematic Text Mitigation in Source Data

- **Common issues**:
  - Toxic language, hallucinated data, misinformation, PII, offensive terms

- **Mitigation strategies**:
  - Text classification: detect & flag at ingestion
  - Data redaction: remove or mask terms before chunking
  - Embedding filtering: exclude high-risk chunks from index

> ✅ Prefer *removal or replacement at ingest time*, not after serving.

---

### 4️⃣ Licensing and Legal Considerations

- **Know your source**:
  - Don’t embed or serve outputs from copyrighted content without permission
  - Prefer open license datasets (e.g., Creative Commons, Apache 2.0)

- **Regulatory awareness**:
  - GDPR: avoid storing or exposing user PII without legal basis
  - Copyright: don’t fine-tune or RAG on content under closed license

- **Recommended practices**:
  - Maintain metadata on source licenses
  - Use attribution if required (via prompt injection or citations)

> 📚 If in doubt, don’t index it — ensure data is legally safe *before* embedding or serving.

---

### 🎯 Key Rules of Thumb

- Mask sensitive info early → improves performance and security.
- Use input/output guardrails to block prompt injection and toxic responses.
- Pre-filter problematic documents — don’t assume the LLM will "handle it."
- Always verify data source licenses → avoid legal exposure in RAG chains.
- Use Unity Catalog to track data lineage and enforce access policies where applicable.

---

## 📈 Section 6: Evaluation and Monitoring  
*Measuring, improving, and maintaining LLM performance in production RAG systems*

---

### 1️⃣ LLM Selection Based on Metrics

- **Quantitative metrics to compare models**:
  | Metric           | Use Case                                      |
  |------------------|-----------------------------------------------|
  | MMLU             | Academic/general knowledge benchmarking       |
  | TruthfulQA       | Hallucination resistance                      |
  | GSM8K            | Math and reasoning                            |
  | ARC              | Multi-step question answering                 |
  | HELM             | Overall evaluation framework (accuracy + bias)|

- **Rule of thumb**:  
  - Choose **smaller models** (e.g., MPT, LLaMA 7B) when:
    - Cost is key, and latency is critical  
    - Task is simple or repetitive
  - Choose **larger models** (e.g., GPT-4, Claude 2) when:
    - High accuracy or nuance is required  
    - Multi-step reasoning is needed

---

### 2️⃣ Key Monitoring Metrics

| Scenario                        | Metric to Monitor                        |
|--------------------------------|------------------------------------------|
| General LLM health             | Token usage, latency, error rate         |
| RAG performance                | Retrieval hit rate, context overlap, hallucination %
| Chatbot/agent quality          | User thumbs up/down, task success rate   |
| Compliance/safety              | Blocked output %, PII leakage detections |

- Use **custom metrics** in MLflow or inference tables for:
  - Prompt length, response length
  - Response time buckets
  - API error codes

---

### 3️⃣ Using MLflow for Evaluation

- Track:
  - Prompt templates
  - Retrieved context
  - Model outputs
  - Eval scores

- Log with:
  ```python
  mlflow.log_metrics({'bleu': 0.89, 'rouge': 0.76})
  mlflow.log_params({'model_name': 'llama2', 'chunk_size': 512})

## 🧠 Additional Study Notes

---

### 🔍 RAG Pipelines & Optimization

- **End-to-End Flow**: Ingest → preprocess → chunk → embed → index → retrieve → prompt → generate.
- **Chunking & Evaluation**:
  - Choose chunk strategy (semantic, sliding window) based on Recall@k or NDCG.
  - Use `LLM-as-a-judge` or human evals to tune retrieval performance.
- **Indexing**: Use Delta tables + Databricks Vector Search to manage embeddings.
- **Retrieval Tip**: Use metadata filters (e.g. `book="1"`) to narrow scope in dense indexes.

---

### 🧠 Databricks Vector Search & Index Management

- **Databricks Vector Search**:
  - Indexes Delta tables with embedded chunks.
  - Query with `.search()` or SQL `ai_query()`.
- **Supports hybrid search**: keyword + embedding for better relevance.
- **Similarity Metric**: Default is L2 via HNSW. Normalize embeddings for cosine similarity.
- **Filtering**: Add `section`, `doc_type`, or `user_role` as metadata filters.

---

### ⚡ Real-Time Data with Feature Store

- Use **Databricks Feature Store + Feature Serving** to supply live, structured inputs.
- Example: Serve up-to-date delivery times, account balances, or session state.
- Complements RAG by grounding with real-time facts.
- Secure via Unity Catalog + auto-scaling endpoints.

---

### 🚀 Model Serving & AI SQL Functions

- **Model Deployment Options**:
  - MLflow PyFunc → wrap chains or preprocessors.
  - Hosted foundation models via MosaicML → use `dbx.ChatCompletion()`.
- **AI Functions**:
  - SQL-native: `ai_query()` or `llm_query()` for inline LLM inference.
  - Useful for dashboards and reporting.

---

### 🧰 MLflow PyFunc + Secrets Best Practices

- Use **PyFunc** for chaining LLM calls with custom logic.
- Avoid `spark.conf.set()` for secrets — use environment variables or Databricks Secrets.
- Track prompt templates, metrics, input/output examples in MLflow runs.

---

### 🔐 Unity Catalog Governance

- Store all data (chunks, features, logs) in Unity Catalog-backed tables.
- Use UC to manage ACLs, model promotion, and audit trails.
- Enable table and model lineage for traceability.

---

### 🤖 Agents, Tools, & Multi-Step Reasoning

- **Agent Use Case**: When an LLM needs to decide dynamically between tools (e.g., RAG vs SQL vs API call).
- Tool list is passed as part of system prompt.
- Use LangChain or Databricks Agents to build agents with memory and tool use.

---

### ✏️ Prompt Engineering Enhancements

- **Input Augmentation**: Dynamically insert user-specific context (e.g. `plan_type`, `locale`, `account_status`).
- **Output Formatting**: Guide model with structure (e.g. `"Respond in JSON with fields X, Y, Z"`).
- **Pre-processing**: Use PyFunc to sanitize inputs, enforce format, or append extra context before prompt hits LLM.

---

### 📊 Model Selection & Trade-offs

| Model Type            | Use Case                                      |
|-----------------------|-----------------------------------------------|
| Small Open Models     | Low-latency tasks (e.g. classification, tagging) |
| Large Open Models     | Reasoning-heavy RAG with self-hosted options  |
| Proprietary Models    | Complex summarization, conversational agents  |

- Check context length, MMLU scores, architecture type when comparing.
- Use proprietary APIs only if compliance risk is acceptable.

---

### 🧪 Monitoring & Evaluation Metrics

- **Before deployment (eval)**:
  - BLEU/ROUGE for summarization
  - MMLU for general model strength
  - Retrieval: Precision@k, Recall@k, NDCG

- **Post-deployment (monitoring)**:
  - Token count, latency, success/failure rate
  - Retrieval relevance score
  - Output rejection rate (e.g. via safety guardrails)

---

### ⚖️ Guardrails: Types & Use Cases

| Type         | Purpose                                            |
|--------------|----------------------------------------------------|
| Safety       | Block offensive/harmful inputs                     |
| Compliance   | Enforce business or legal constraints (e.g. no politics) |
| Contextual   | Ensure outputs align with user scope/intended use |
| Evaluation   | Human or model-in-the-loop scoring & audits       |

- Use **input filters + output format checks + system prompt constraints**.
- Guardrails can wrap the entire app or live inside a chain.

---

### 🧠 DatabricksIQ, LakehouseIQ, & Assistant Features

- **DatabricksIQ**: LLM assistant for code, SQL, and UI help. Not exposed via API.
- **LakehouseIQ**: Natural language interface over structured data. Supports business-specific querying (e.g. “show me sales by channel for Q2”).
- **SQL AI Functions**:
  - Use `ai_query()` in dashboards or notebooks for lightweight inference.
  - Ideal for summarization, classification, or content tagging in batch.

---

## 🧠 Additional GenAI Study Notes (Cleaned & Expanded)

---

### ✅ Prompt Design & Output Control

- **Neutralizing Tone**: Instruct the LLM to rephrase emotionally charged input into neutral, professional language.
  - Example: `"Rewrite this message in a neutral, factual tone."`

- **Summarization Task**: Condensing paragraphs into 1–2 sentences = summarization. Evaluate with ROUGE (coverage) or BLEU (precision).

- **Output Structuring**: Use clear format instructions (e.g., “Respond in JSON with fields X, Y, Z”) and few-shot examples for reliable formatting.

---

### 🔁 LLM Workflows & Agents

- **Multi-step LLM Workflow**: Used when tasks require chaining multiple steps (e.g., retrieve → reason → act).
  - Use LangChain or Databricks Agent Framework to orchestrate.
  
- **ReAct Framework**: Combines reasoning (thoughts) with tool actions — ideal for agentic LLMs making decisions and using tools.

- **Agent Frameworks**: Let LLMs select from registered tools (e.g., API lookup, SQL query, RAG retriever). Enables dynamic, goal-directed workflows.

---

### 💲 Cost Management & Serving Patterns

- **Pay-per-token**: Ideal for low-volume use cases — no idle costs, scales with usage, no provisioning needed.

- **Low-Cost RAG Setup**: Efficient stack = Prompt + Retriever + LLM (no fine-tuning or agent logic unless required).

---

### 📚 Feature Store & Real-Time Context

- **Databricks Feature Serving**: Serves structured, per-query data (e.g., user balance, delivery time) in real-time.
  - Use when data can't be embedded in advance.
  - Complements unstructured RAG by grounding LLMs in current state.

---

### 🧱 Chunking & Semantic Context

- **Section Headers in Chunks**: Boost semantic clarity in RAG. Helps retriever/embedding models infer context.

---

### 🧰 Tooling Overview

#### Orchestration & Reasoning
- `LangChain` – Build chains, agents, tool use
- `ReAct` – Reasoning + acting loop for tool-using agents

#### Evaluation & Monitoring
- `MLflow` – Track experiments, register models, deploy with PyFunc
  - PyFunc lets you wrap preprocessing + postprocessing logic
- **Evaluation Metrics**:
  - `BLEU` – Translation accuracy
  - `ROUGE` – Summarization recall
  - `MMLU` – General LLM benchmark (academic + reasoning)
  - `NDCG` – Normalized Discounted Cumulative Gain, ranks relevance in retrieval
  - `LLMs-as-a-judge` – LLMs evaluate outputs for quality/consistency

---

### 🧠 Databricks-Specific Features

- **LakehouseIQ** – Natural language interface to structured data; understands metadata and lineage.
- **MosaicML** – Databricks-hosted open models; scalable + customizable foundation models.
- **Unity Catalog Volume** – Managed volume-based storage with access control + lineage (e.g., for models, training data).
- **Inference Tables** – Auto-log requests/responses from model serving endpoints — enables live debugging and monitoring.

---

### 🧪 Embedding & Model Ecosystem

#### Key Python Libraries
- `unstructured` – Extract text from PDFs, DOCX, HTML
- `pytesseract` – OCR for image-to-text
- `langchain` – Chain components, agents, tool use
- `mlflow` – Model tracking, serving, versioning
- `sentence-transformers` – Generate embeddings for retrieval
- `transformers` – Load/tokenize Hugging Face models
- `pandas` – Manipulate structured data
- `openai` – Interface with OpenAI models (e.g., GPT-4)
- `faiss` – Self-hosted vector search engine
- `scrapy` – Web scraping 
- `PyMuPDF` – Extract structured text from PDFs (preserves layout)
- `pdfplumber` – Extract text from PDFs with table support
- `doctr` – Deep learning OCR for scanned PDFs/images (DocTR = Document Text Recognition)
- `datasets` (Hugging Face) – Load benchmark datasets for evaluation/fine-tuning

---

### 🧠 Databricks LLMs – Simplified Reference

| Model Name         | Notes                                                                 |
|--------------------|-----------------------------------------------------------------------|
| **DBRX**           | Databricks' flagship LLM; strong at summarization and general reasoning |
| **MPT (7B / 30B)** | MosaicML models; efficient for chat/instruction-tuned tasks           |
| **LLaMA (2, 3.1, 3.3, 4)** | Meta’s open models; high performance for reasoning & chat   |
| **CodeLLaMA**      | Optimized for code generation and understanding                       |
| **Claude (3.7, 4, Opus)** | Anthropic models; excellent for deep reasoning, summarization |
| **Whisper-large-v3** | OpenAI’s speech-to-text model; used for audio transcription        |
| **BGE / GTE**      | Embedding models for RAG; fast and effective in dense retrieval       |
| **Dolly (1.0 / 2.0)** | Early Databricks models; not production-ready but good for learning |
| **DistilBERT**     | Lightweight transformer for classification or embedding               |

---

