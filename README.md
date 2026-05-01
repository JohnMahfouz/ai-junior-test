# Multi-Agent RAG System – Smart Restaurant Assistant

---

# 📌 Scenario

You are building an AI system for a restaurant chain called:

# **NovaBite Restaurants**

NovaBite operates multiple branches and wants an AI assistant that can:

- Answer customer questions from internal knowledge base (RAG)
- Check live table availability
- Provide menu recommendations
- Handle follow-up questions using memory
- Call backend tools (MCP-style or simulated server logic)
- Route tasks intelligently using sub-agents

The system **must** be architected using:

- LangChain
- Sub-agents
- RAG
- Tool-based execution
- Memory
- Proper orchestration

> ⚠️ This is NOT a chatbot demo.  
> This is a real system design + implementation evaluation.

---

# 🎯 Goal

Build a production-style **multi-agent architecture** that:

- Uses RAG for restaurant knowledge
- Uses tool-calling for operational tasks
- Delegates properly through sub-agents
- Demonstrates memory continuity
- Minimizes hallucinations
- Can be evaluated for retrieval accuracy

---

# 🏗️ Required Architecture

You must implement the following components:

---

## 1️⃣ Main Orchestrator Agent

### Responsibilities

- Classify user intent
- Route requests to appropriate sub-agent
- Maintain conversation memory
- Merge and validate sub-agent responses
- Decide when to call tools
- Handle ambiguity and clarification
- Prevent hallucinated outputs

⚠️ The orchestrator must NOT contain business logic directly.  
It must delegate to sub-agents.

---

## 2️⃣ Required Sub-Agents

---

### 🍽️ A. Restaurant Knowledge RAG Agent

Responsible for answering questions from internal documents such as (You don't have to implement all domains below just pick 2):

- Menu descriptions  
- Allergen information  
- Opening hours  
- Branch policies  
- Loyalty program rules  
- Refund policy  
- Event hosting information  

---

### 🔍 RAG Implementation Requirements

You must implement:

- Document ingestion pipeline
- Chunking strategy (**justify your choice**)
- Embedding model (**justify your choice**)
- Vector database (FAISS / Chroma / etc.)
- Retrieval strategy (top-k, optional hybrid)
- Context filtering
- Hallucination prevention
- Grounded answer generation

Your system must NOT hallucinate nonexistent menu items.

---

### Example RAG Queries

- “Do you have vegan pasta?”
- “Is the chicken grilled or fried?”
- “What are your opening hours on weekends?”
- “Do you host birthday events?”
- “What’s included in the premium catering package?”

---

### 🛠️ B. Operations Agent (Tool-Based / MCP-Style)

This agent handles live operational queries.

You may:

- Connect to a real MCP server  
**OR**
- Implement functions that simulate server logic

The system must behave like it is calling real external tools.

---

### Required Tools (Implement At Least Two)


check_table_availability(date, time, branch)
book_table(name, date, time, branch)
get_today_special(branch)
check_loyalty_points(user_id)


---

# ⏳ Time Limit

You have **2 days (48 hours)** from the moment you receive this test to complete and submit your solution.

---

# 📬 Submission Instructions

1. Fork the provided repository.
2. Implement your solution in your fork.
3. Ensure your repository includes:
   - Complete source code
   - Updated README with:
     - Architecture explanation
     - RAG design decisions
     - Tool simulation or MCP integration explanation
     - Memory design explanation
     - Example queries and outputs
     - Assumptions made
4. Push your final implementation to your forked repository.
5. Send the repository link via email to:

📩 **careers@fekracorp.com**

---

# Implementation Notes / Updated README Details

This section documents the completed implementation for the submission requirements listed above.

## Architecture Explanation

The project is implemented as a multi-agent LangChain system:

```text
User
  -> Main Orchestrator Agent
      -> Restaurant Knowledge RAG Agent
      -> Operations Agent
          -> LangChain tools
          -> MCP-style FastAPI tool server
  -> Session memory
```

Main files:

```text
agents/orchestrator.py       classifies intent, routes requests, stores memory
agents/rag_agent.py          answers knowledge questions using retrieved context
agents/operations_agent.py   handles booking, availability, specials, and loyalty flows
rag/ingest.py                loads documents, chunks them, embeds them, and builds FAISS
rag/retriever.py             loads the local FAISS retriever
prompts/rag_prompt.py        grounded answer prompt for RAG
tools/restaurant_tools.py    LangChain tool wrappers
tools/mcp_server.py          simulated external operations service
memory/session_memory.py     per-session conversation memory
api/main.py                  FastAPI chat endpoint
ui/app.py                    Streamlit chat UI
eval/rag_eval.py             basic retrieval evaluation
```

The orchestrator routes requests to sub-agents and does not keep restaurant business logic inside the router.

## RAG Design Decisions

The RAG agent answers from:

- `data/menu.md`
- `data/policies.md`

Implemented RAG pipeline:

- Document ingestion: `TextLoader` loads Markdown files from `data/`
- Chunking: `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)`
- Embeddings: `all-MiniLM-L6-v2`
- Vector database: local FAISS index in `faiss_index/`
- Retrieval strategy: top-k retrieval with `k=4`
- Source tracking: document source filename is stored in metadata
- Grounded generation: answers are generated only from retrieved context

The chunk size keeps short menu and policy sections together without mixing too many unrelated items in one chunk.

If the answer is not available in the retrieved context, the assistant returns:

```text
I don't have that information in our knowledge base.
```

## Tool Simulation / MCP-Style Integration Explanation

The operations tools are exposed through a local FastAPI service:

```text
tools/mcp_server.py
```

The LangChain tools in `tools/restaurant_tools.py` call that service over HTTP. This keeps tool execution separate from the agent and simulates an external backend/MCP-style service.

Implemented required tools:

```text
check_table_availability(date, time, branch)
book_table(name, date, time, branch)
get_today_special(branch)
check_loyalty_points(user_id)
```

The implementation also includes an availability slot helper for demo usability.

## Memory Design Explanation

Memory is stored per `session_id` in:

```text
memory/session_memory.py
```

The orchestrator loads previous turns before routing a request and saves the user/assistant turn after the response is produced.

Memory is used by:

- The RAG agent for follow-up knowledge questions
- The operations agent for follow-up booking or availability requests

Memory is in-process and resets when the server restarts.

## Example Queries and Outputs

### RAG: Vegan Pasta

Input:

```text
Do you have vegan pasta?
```

Expected output:

```text
Yes. Pasta Primavera is vegan. It is penne pasta with seasonal vegetables in a garlic-olive oil sauce, with no cream or cheese.
```

### RAG: Missing Knowledge

Input:

```text
Do you serve sushi?
```

Expected output:

```text
I don't have that information in our knowledge base.
```

### Operations: Check Availability

Input:

```text
Is there a table available at Downtown on 2026-05-03 at 19:00?
```

Expected output:

```text
Good news! A table is available at the Downtown branch on 2026-05-03 at 19:00.
```

### Operations: Book Table

Input:

```text
Book a table for John Smith at Downtown on 2026-05-03 at 19:00
```

Expected output:

```text
Booking confirmed! Booking ID: NB-1001 | Name: John Smith | Branch: Downtown | 2026-05-03 at 19:00. We hold reservations for 15 minutes past the booking time. See you soon!
```

### Operations: Loyalty Points

Input:

```text
Check loyalty points for user_002
```

Expected output:

```text
Loyalty balance for user_002: 3,800 points. You can redeem for a $25 discount.
```

### Memory Follow-Up

Turn 1:

```text
Do you have vegan pasta?
```

Turn 2:

```text
Does it contain gluten?
```

Expected output:

```text
Yes. Pasta Primavera contains gluten because it uses penne pasta. It can be made with gluten-free pasta on request for an extra $3. [Source: menu.md]
```

## Assumptions Made

- NovaBite is a fictional restaurant chain.
- The operations backend is simulated with FastAPI.
- Availability, bookings, specials, and loyalty points are stored in memory for the demo.
- The FAISS index is local and can be rebuilt from `data/menu.md` and `data/policies.md`.
- Conversation memory is session-based and not persistent across server restarts.
- Groq is used for chat model calls through `langchain-groq`.
- The API is the main entry point; Streamlit is included for demo convenience.

## Setup and Run

```bash
pip install -r requirements.txt
cp .env.example .env
python tools/mcp_server.py
uvicorn api.main:app --reload
streamlit run ui/app.py
python eval/rag_eval.py
```
