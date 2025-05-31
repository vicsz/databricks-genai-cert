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
