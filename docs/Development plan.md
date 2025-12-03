# Dev plan 

# Phase 0: Infrastructure & Environment Specification

**Project:** The Headless Governor System **Version:** 1.0.1 **Status:** ACTIONABLE

## **1\. Executive Summary**

This phase establishes the "factory floor" for the application. Because the Governor System relies on a complex interaction between a deterministic State Machine (Kernel), multiple databases (Memory), and external APIs (Tools), the environment must be rigorously typed, containerized, and automated.

**Objective:** Create a reproducible local development environment and a strict CI/CD pipeline.

## **2\. Tech Stack Constraints**

* **Language:** Python 3.11+ (Required for latest async/await patterns).  
* **Package Manager:** Poetry (Required for dependency resolution).  
* **IDE:** PyCharm Professional (Recommended) or Community.  
* **Web Framework:** FastAPI (Async default).  
* **Linting/Formatting:** `ruff` (Replaces Black/Isort/Flake8).  
* **Type Checking:** `mypy` (Strict mode enabled).  
* **Configuration:** `pydantic-settings`.  
* **Container Runtime:** Docker & Docker Compose.

## **3\. Directory Structure (Monorepo)**

The project follows a "Modular Monolith" structure. The AI Agent must generate the following file tree.

headless-governor/  
├── .github/  
│   └── workflows/  
│       ├── quality\_gate.yml    \# Linting, Types, Tests  
│       └── build\_deploy.yml    \# Docker Build & Push  
├── .idea/                      \# PyCharm Shared Configs  
│   └── runConfigurations/      \# Shared Run/Debug configs (xml)  
├── config/                     \# Infrastructure Configs  
│   ├── prometheus/  
│   └── grafana/  
├── src/  
│   ├── \_\_init\_\_.py  
│   ├── main.py                 \# Application Entry Point  
│   ├── core/                   \# Shared types & utilities  
│   │   ├── config.py           \# Pydantic Settings  
│   │   └── domain/             \# The "Event Object" definitions  
│   ├── governor/               \# The "Kernel" (Control Plane)  
│   │   ├── state\_machine/      \# LangGraph definitions  
│   │   └── context/            \# Context Manager logic  
│   ├── action\_plane/           \# The "Hands"  
│   │   ├── tool\_registry/      \# Atomic tool definitions  
│   │   └── policy\_engine/      \# Security logic (Level 0-2)  
│   ├── memory/                 \# Database Adapters (Data Plane)  
│   │   ├── redis\_client.py  
│   │   ├── vector\_store.py  
│   │   └── graph\_store.py  
│   └── interfaces/             \# I/O Routers  
│       └── webhooks/           \# Twilio/Telegram handlers  
├── tests/  
│   ├── unit/  
│   └── integration/  
├── .dockerignore  
├── .env.example  
├── .gitignore  
├── .pre-commit-config.yaml  
├── docker-compose.yml  
├── Dockerfile  
├── poetry.lock  
├── pyproject.toml  
└── README.md

## **4\. Configuration Specifications**

### **4.1 Dependency Management (`pyproject.toml`)**

Initialize Poetry with these core dependencies:

\[tool.poetry.dependencies\]  
python \= "^3.11"  
fastapi \= "^0.109.0"  
uvicorn \= "^0.27.0"  
pydantic \= "^2.6.0"  
pydantic-settings \= "^2.2.0"  
redis \= "^5.0.0"  
\# Core logic placeholders  
langchain \= "^0.1.0"  
langgraph \= "^0.0.10" \# Critical for State Machine

\[tool.poetry.group.dev.dependencies\]  
pytest \= "^8.0.0"  
ruff \= "^0.2.0"  
mypy \= "^1.8.0"  
pre-commit \= "^3.6.0"  
httpx \= "^0.26.0" \# For async testing

### **4.2 Linting Rules (`ruff.toml` embedded in pyproject)**

Strict enforcement is required to prevent "output parsing" errors later in the Action Plane.

\[tool.ruff\]  
line-length \= 88  
select \= \["E", "F", "I", "B", "UP"\] \# pycodestyle, pyflakes, isort, bugbear, pyupgrade  
target-version \= "py311"

\[tool.mypy\]  
strict \= true  
ignore\_missing\_imports \= true  
plugins \= \["pydantic.mypy"\]

### **4.3 PyCharm Specific Setup**

* **Interpreter:** Configure PyCharm to use the Poetry virtual environment (`poetry env info --path`).  
* **Plugins:** Install the **Ruff** plugin for real-time linting and the **Pydantic** plugin for better autocomplete.  
* **Run Configurations:** Ensure `.idea/runConfigurations` is added to git so team members share the "Run Server" and "Run Tests" configurations.

## **5\. Containerization Strategy**

### **5.1 `Dockerfile` (Multi-stage)**

* **Base:** `python:3.11-slim`  
* **Builder Stage:** Install Poetry, export requirements.txt (for lighter runtime), build wheel.  
* **Runner Stage:** Install dependencies, copy source code.  
* **User:** Run as non-root user `governor`.

### **5.2 `docker-compose.yml` (The Brain)**

This defines the local "Headless OS".

version: '3.8'

services:  
  \# The Kernel  
  governor\_engine:  
    build: .  
    command: uvicorn src.main:app \--host 0.0.0.0 \--port 8000 \--reload  
    volumes:  
      \- ./src:/app/src  
    env\_file: .env  
    depends\_on:  
      \- redis  
      \- vector\_db  
      \- graph\_db  
    ports:  
      \- "8000:8000"

  \# Short-Term Memory (RAM)  
  redis:  
    image: redis:alpine  
    ports:  
      \- "6379:6379"

  \# Episodic Memory \+ Procedural Rules (Postgres \+ pgvector)  
  vector\_db:  
    image: pgvector/pgvector:pg16  
    environment:  
      POSTGRES\_USER: governor  
      POSTGRES\_PASSWORD: dev\_password  
      POSTGRES\_DB: governor\_memory  
    ports:  
      \- "5432:5432"  
    volumes:  
      \- postgres\_data:/var/lib/postgresql/data

  \# Semantic Memory (Graph)  
  graph\_db:  
    image: neo4j:community  
    environment:  
      NEO4J\_AUTH: neo4j/dev\_password  
    ports:  
      \- "7474:7474" \# HTTP  
      \- "7687:7687" \# Bolt  
    volumes:  
      \- neo4j\_data:/data

volumes:  
  postgres\_data:  
  neo4j\_data:

## **6\. CI/CD Pipeline (`.github/workflows`)**

### **6.1 `quality_gate.yml`**

Must run on every Pull Request.

1. **Checkout Code**  
2. **Install Poetry**  
3. **Lint:** `poetry run ruff check .`  
4. **Type Check:** `poetry run mypy src/`  
5. **Test:** `poetry run pytest tests/unit`

### **6.2 `build_deploy.yml`**

Runs on push to `main`.

1. **Login to GHCR**  
2. **Build Docker Image**: Tag with `sha` and `latest`.  
3. **Scan**: Run `trivy` for vulnerability scanning.  
4. **Push**: Push to registry.

# **Phase 1: The Kernel & Interface Specification**

**Project:** The Headless Governor System **Version:** 1.0.0 **Status:** ACTIONABLE **Dependencies:** Phase 0 (Infrastructure)

## **1\. Executive Summary**

Phase 1 focuses on building the **Control Plane** (The Governor) and the **Action Plane** (The Tools). We will implement the deterministic State Machine that manages the conversation flow, ensuring the AI doesn't just "chat" but actually "executes" tasks safely.

**Objective:** Deploy a working "Governor" that can receive a message via webhook, decide to run a tool (simulated), check permissions, and reply to the user.

## **2\. The Control Plane (Governor Engine)**

### **2.1 The State Machine (`src/governor/state_machine/`)**

We will use **LangGraph** to define a cyclic graph. The LLM is not the controller; the Graph is the controller.

**States (Nodes):**

1. **`IDLE`**: System sleeping. Triggered by incoming Event.  
2. **`ANALYZE`**: The "Context Manager" builds the prompt. LLM decides: "Talk" or "Act"?  
3. **`TOOL_DECISION`**: If "Act", identify the tool and parameters.  
4. **`POLICY_CHECK`**: **(Critical)** Intercepts the tool call. Checks Risk Score.  
   * *Pass:* \-\> `EXECUTE`.  
   * *Fail/High Risk:* \-\> `AWAIT_CONFIRMATION`.  
5. **`EXECUTE`**: Runs the Python function from the Tool Registry.  
6. **`AWAIT_CONFIRMATION`**: Pauses execution. Sends "Do you want to proceed?" to user.  
7. **`RESPOND`**: Formulates final response to user.

**Graph Flow:**

graph TD  
    Start\[Webhook Event\] \--\> ANALYZE  
    ANALYZE \--\>|Text Reply| RESPOND  
    ANALYZE \--\>|Tool Call| TOOL\_DECISION  
    TOOL\_DECISION \--\> POLICY\_CHECK  
    POLICY\_CHECK \--\>|Approved| EXECUTE  
    POLICY\_CHECK \--\>|Risk Level 2| AWAIT\_CONFIRMATION  
    AWAIT\_CONFIRMATION \--\>|User Says Yes| EXECUTE  
    AWAIT\_CONFIRMATION \--\>|User Says No| RESPOND  
    EXECUTE \--\> ANALYZE  
    RESPOND \--\> End\[End Turn\]

### **2.2 The Context Manager (`src/governor/context/`)**

This module dynamically assembles the System Prompt.

* **Input:** `User_ID`, `Current_Input`, `State`.  
* **Logic:** `Prompt = Persona + Environment + Memory + Tasks`.  
* **Implementation:** Create a `ContextAssembler` class. For Phase 1, "Environment", "Memory", and "Tasks" will return mock/static data.

## **3\. The Action Plane (Tool Registry)**

### **3.1 The Decorator Pattern (`src/action_plane/tool_registry/`)**

We must not manually register tools. We need a decorator that handles registration and risk scoring.

**Requirement:** Define a custom decorator `@governor_tool` that accepts:

* `name`: string  
* `description`: string (for LLM)  
* `risk_level`: Enum (`SAFE`, `SENSITIVE`, `DANGEROUS`)  
* `args_schema`: Pydantic Model

**Example Signature:**

class SendEmailSchema(BaseModel):  
    recipient: EmailStr  
    body: str

@governor\_tool(  
    name="send\_email",  
    risk\_level=RiskLevel.DANGEROUS, \# Level 2  
    args\_schema=SendEmailSchema  
)  
def send\_email(recipient: str, body: str):  
    ...

### **3.2 The Policy Engine (`src/action_plane/policy_engine/`)**

This is the "Firewall".

* **Logic:**  
  * IF `risk_level == SAFE`: Return `ALLOWED`.  
  * IF `risk_level == DANGEROUS`: Return `REQUIRES_CONFIRMATION`.  
* **Storage:** It doesn't store state yet; it just evaluates the *current* request against the *static* definition.

## **4\. Interfaces (Webhooks)**

### **4.1 Unified Ingress (`src/interfaces/webhooks/`)**

* Create a generic FastAPI route `POST /webhook/ingress`.  
* **Payload Adapter:** It should accept a raw JSON payload and normalize it into the standard `GovernorEvent` object defined in `src/core/domain/`.  
* **Async Processing:** The route must return `200 OK` immediately to the provider (Twilio/Telegram) and spawn the Governor Graph as a background task (`BackgroundTasks` in FastAPI).

# **Phase 2: RAG++ Memory Hierarchy Specification**

**Project:** The Headless Governor System **Version:** 1.5.0 **Status:** ACTIONABLE **Dependencies:** Phase 1 (Kernel)

## **1\. Executive Summary**

Phase 2 replaces the mock memory in the `ContextAssembler` with actual database integrations. We will implement a multi-tiered memory architecture that persists conversation history (Redis), logs immutable episodes (Vector DB), and maps structured relationships (Graph DB).

**Objective:** Enable the Governor to recall past conversations, understand user relationships, and maintain context across sessions.

## **2\. Architecture Overview**

To ensure low-latency responses (\~500ms overhead), we utilize a **Parallel Context Grid** strategy. We do not chain queries sequentially; instead, the `ContextAssembler` fires asynchronous requests to all four memory tiers simultaneously.

### **2.1 The Retrieval Grid**

1. **Tier 1 (Short-Term):** Fetches the **Rolling Summary** \+ Recent Buffer from Redis (Token-optimized).  
2. **Tier 2 (Episodic):** Uses the **Embedding Strategy** to find top-k past events in PGVector.  
3. **Tier 3 (Semantic):** Uses Entity Extraction to query the **Dynamic Ontology** in Neo4j.  
4. **Tier 4 (Procedural):** Uses the **Embedding Strategy** to find relevant **Behavioral Instructions** in PGVector.

### **2.2 Data Flow Diagram**

flowchart TD  
    UserInput\[User Input\]  
      
    subgraph Pre\_Processing \[Pre-Processing\]  
        Embed\[Embedding Strategy\]  
        Extract\[Entity Extractor\]  
    end  
      
    subgraph Storage\_Layer \[The Data Plane\]  
        Redis\[(Redis: Summary \+ Buffer)\]  
        VectorDB\[(PGVector: Episodic Logs)\]  
        GraphDB\[(Neo4j: Dynamic Graph)\]  
        RuleDB\[(PGVector: Instructions)\]  
    end  
      
    subgraph Logic\_Layer \[The Control Plane\]  
        Assembler\[Context Assembler\]  
        LLM\[LLM Inference\]  
    end

    %% Flow  
    UserInput \--\> Embed  
    UserInput \--\> Extract  
    UserInput \--\> Redis

    %% Parallel Fetch  
    Redis \-- "1. Rolling Summary" \--\> Assembler  
    Embed \-- "Vector Search" \--\> VectorDB  
    VectorDB \-- "2. Relevant Episodes" \--\> Assembler  
    Extract \-- "Cypher Query" \--\> GraphDB  
    GraphDB \-- "3. Semantic Facts" \--\> Assembler  
    Embed \-- "Vector Search" \--\> RuleDB  
    RuleDB \-- "4. Operational Rules" \--\> Assembler

    %% Execution  
    Assembler \-- "System Prompt" \--\> LLM

## **3\. Tier 1: Short-Term Memory (Working Context)**

**Location:** `src/memory/short_term/`

### **3.1 The Strategy: Token-Managed Rolling Summary**

We rely on **Redis** but move beyond a simple list. We use a "Summary Hinge" approach.

* **The Buffer:** Holds raw messages up to a strict token limit (e.g., 2,000 tokens).  
* **The Hinge:** When the buffer overflows, the oldest messages are not deleted; they are fed to a cheap LLM (GPT-4o-mini) to update a persistent `running_summary`.

### **3.2 Schema (Redis Hash)**

Key: `session:{user_id}:v1`

* `buffer`: List\[JSON\] (The raw interaction log).  
* `summary`: String (The compressed context of everything *prior* to the buffer).  
* `token_count`: Integer (Cached count to verify budget without re-tokenizing).

### **3.3 Interface `RedisClient` (With Atomic Locking)**

* `add_turn(user_id, user_msg, ai_msg)`:  
  * **Acquire Lock:** Use Redis distributed lock to prevent race conditions.  
  * **Push:** Append messages to `buffer`.  
  * **Manage Budget:**  
    * Calculate total tokens (using `tiktoken`).  
    * IF `total > MAX_TOKENS`:  
      * Pop oldest $K$ messages.  
      * **Async Trigger:** `summarize_and_merge(old_summary, popped_messages)`.  
  * **Release Lock.**  
* `get_context(user_id) -> ContextObject`:  
  * Returns: `{ "summary": "User previously discussed X...", "recent_messages": [...] }`  
  * *Note:* The `ContextAssembler` injects the `summary` into the System Prompt and `recent_messages` as Chat History.

## **4\. Tier 2: Episodic Memory (Vector DB)**

**Location:** `src/memory/episodic/`

### **4.1 The Embedding Abstraction (Strategy Pattern)**

To prevent vendor lock-in and improve robustness, we do not call OpenAI directly in the store. We implement the **Strategy Pattern**.

* **Interface:** `src/core/embeddings/base.py` \-\> `EmbeddingProvider`  
  * `embed_text(text: str) -> List[float]`  
  * `embed_batch(texts: List[str]) -> List[List[float]]`  
* **Implementations:**  
  * `OpenAIEmbedding`: Uses `text-embedding-3-small` (cheaper/better than Ada).  
  * `LocalEmbedding`: Uses `sentence-transformers` (e.g., `all-MiniLM-L6-v2`) for offline/privacy.  
* **Resilience:** All API calls must be wrapped with `tenacity` decorators for **Exponential Backoff** retries (e.g., wait 1s, 2s, 4s, then fail).

### **4.2 Schema (PGVector/Weaviate)**

This is an **Append-Only** log.

* **Table:** `episodic_logs`  
* **Fields:**  
  * `id`: UUID  
  * `user_id`: String (Indexed)  
  * `timestamp`: DateTime  
  * `content`: Text (User input \+ Assistant response summary)  
  * `embedding`: Vector (Dimensions dynamic based on provider, e.g., 1536 or 384\)  
  * `metadata`: JSON (e.g., location, tool\_used)

### **4.3 Interface `VectorStore`**

* `log_interaction(user_id, text, metadata)`:  
  1. Call `EmbeddingProvider.embed_text(text)` (Handles retries).  
  2. Insert row to Postgres.  
* `recall(user_id, query_text, limit=5)`:  
  1. Embed query.  
  2. Perform Cosine Similarity Search.

## **5\. Tier 3: Semantic Memory (Graph DB)**

**Location:** `src/memory/semantic/`

### **5.1 Schema Strategy: Dynamic Ontology**

We do not enforce a rigid schema (e.g., only "Person" or "Location"). Instead, we use a **Schema-on-Write** approach governed by the O-E-R loop.

#### **Core Nodes (Fixed)**

These exist for every user to ensure system stability.

* `User`  
* `SystemAgent` (The AI itself)

#### **Dynamic Nodes (Learned)**

The extraction layer can propose any label.

* *Examples:* `Project`, `Pet`, `Investment`, `Cryptocurrency`, `Medication`.

### **5.2 The O-E-R Schema Evolution Loop**

To prevent schema fragmentation (e.g., having both `Dog` and `Canine` nodes), the **Reflector Agent** (Phase 4\) performs nightly ontology maintenance.

1. **Observe (Runtime):** The extractor tags entities with the most specific label possible.  
   * *Input:* "I bought Bitcoin." \-\> `(:Asset {name: "Bitcoin", type: "Crypto"})`  
2. **Reflect (Nightly):** The Reflector scans for disjointed labels.  
   * *Reasoning:* "User has `Bitcoin` labeled `Asset` and `Ethereum` labeled `Crypto`. These are semantic siblings."  
3. **Consolidate (Write):** The Reflector executes a graph refactor.  
   * *Action:* Rename label `Asset` to `Crypto` for consistency.

### **5.3 Interface `GraphStore`**

* `upsert_entity(user_id, entity_type, entity_name, properties)`: Uses `MERGE` to avoid duplicates.  
* `get_related_facts(user_id, text)`:  
  * **Step 1:** LLM extracts keywords (e.g., "Bitcoin").  
  * **Step 2:** Cypher Fuzzy Search on all label types.  
  * **Step 3:** Returns the subgraph (neighbors) of the matched node.

## **6\. Tier 4: Procedural Memory (Semantic Rules)**

**Location:** `src/memory/procedural/`

We split Procedural Memory into **Static Profiles** (Hard Facts) and **Dynamic Instructions** (Soft Rules).

### **6.1 The Profile Store (Key-Value)**

Used for rigid configuration that must *always* be present.

* **Table:** `user_profile`  
* **Fields:** `user_id`, `key` (e.g., "timezone", "name"), `value` (text).  
* **Usage:** Loaded into System Prompt $P\_{Base}$ on *every* turn.

### **6.2 The Instruction Store (Vectorized Rules)**

Used for behavioral preferences that are context-dependent.

* **Table:** `agent_instructions`  
* **Fields:**  
  * `instruction`: Text (e.g., "Never book Spirit Airlines unless cost \< $100")  
  * `category`: Text (e.g., "travel\_booking")  
  * `embedding`: Vector (Embeds the *intent* of the rule)  
  * `confidence`: Float (0.0 \- 1.0, updated by O-E-R loop)  
* **Usage:** We embed the User's Query and retrieve the top-k relevant instructions. This allows the agent to find "Travel Rules" when the user discusses a trip, without polluting the context with "Dietary Rules".

## **7\. Integration: The Context Assembler (Update)**

Refactor `src/governor/context/assembler.py` to fetch from these services in parallel using `asyncio.gather`.

**Prompt Construction Logic:**

\# 1\. Parallel Fetch  
profile, history\_ctx, memories, facts, rules \= await asyncio.gather(  
    procedural.get\_profile(uid),  
    redis.get\_context(uid), \# Returns {summary, messages}  
    vector.recall(uid, query),  
    graph.get\_facts(uid, query),  
    procedural.get\_relevant\_rules(uid, query)  
)

\# 2\. Assembly  
system\_prompt \= f"""  
You are the Governor.  
CORE PROFILE: {profile}  
OPERATIONAL INSTRUCTIONS: {rules}  
PREVIOUS CONTEXT SUMMARY: {history\_ctx.summary}  
RELEVANT MEMORIES: {memories}  
KNOWN FACTS: {facts}  
"""

\# 3\. Chat History  
\# Inject history\_ctx.messages as the actual chat turns

# **Phase 2.5: Interface Wiring (Telegram)**

**Project:** The Headless Governor System **Version:** 1.0.0 **Status:** IMMEDIATE ACTION **Dependencies:** Phase 1 (Kernel), Phase 2 (Memory)

## **1\. Executive Summary**

The Core Logic (Kernel) and Memory (RAG++) are defined. To test the system "in the wild," we must implement the specific I/O adapters for Telegram. This allows you to chat with your local Governor using **ngrok**.

**Objective:** Enable bidirectional communication: `User (Telegram) -> Webhook -> Governor -> API Call -> User (Telegram)`.

## **2\. Ingress: The Telegram Adapter**

**Location:** `src/interfaces/webhooks/telegram.py`

We need a specific Pydantic model to validate incoming Telegram updates and a mapper to convert them into the system's canonical `GovernorEvent`.

### **2.1 Telegram Payload Schema**

We only care about text messages for now.

class TelegramChat(BaseModel):  
    id: int  
    type: str

class TelegramMessage(BaseModel):  
    message\_id: int  
    from\_user: dict \= Field(alias="from")  
    chat: TelegramChat  
    text: str | None \= None

class TelegramUpdate(BaseModel):  
    update\_id: int  
    message: TelegramMessage | None \= None

### **2.2 The Mapper Logic**

The generic `GovernorEvent` (from Phase 1\) expects a `user_id` and `session_id`.

* **Mapping Strategy:**  
  * `GovernorEvent.user_id` \= `TelegramUpdate.message.from_user.id` (Immutable ID).  
  * `GovernorEvent.session_id` \= `telegram:{user_id}`.  
  * `GovernorEvent.text` \= `TelegramUpdate.message.text`.

## **3\. Egress: The Outbound Transport**

**Location:** `src/interfaces/transports/telegram.py`

The Governor's State Machine ends at the `RESPOND` node. We need a function that actually delivers that response.

### **3.1 The Client Class**

Use `httpx` (Async) to call the Telegram Bot API.

* **Function:** `send_message(chat_id: int, text: str)`  
* **Retry Logic:** Use `tenacity` to handle network blips.  
* **Formatting:** Enable `parse_mode="Markdown"` to support the AI's formatted output.

### **3.2 Wiring to the Graph**

We must modify the `RESPOND` node in the LangGraph (Phase 1\) to call this transport function as a "side effect" before finishing the turn.

## **4\. Local Testing Setup (Ngrok)**

Since Telegram cannot send webhooks to `localhost`, we use a tunnel.

1. **Install Ngrok:** `brew install ngrok/cask/ngrok` (or similar).  
2. **Start Tunnel:** `ngrok http 8000`

**Set Webhook:**  
curl \-F "url=https://\<your-ngrok-id\>.ngrok-free.app/webhook/telegram" \\  
     \[https://api.telegram.org/bot\](https://api.telegram.org/bot)\<YOUR\_TOKEN\>/setWebhook

3. 

## **5\. Implementation Prompts for Agent**

**Prompt 1 (The Adapter):**

"Create `src/interfaces/webhooks/telegram.py`. Define the Pydantic models for a `TelegramUpdate`. Create a function `normalize_telegram_event(update: TelegramUpdate) -> GovernorEvent`. It should extract the user ID and text. If text is missing (e.g., image), raise a `ValueError` for now."

**Prompt 2 (The Transport):**

"Create `src/interfaces/transports/telegram_client.py`. Use `httpx` to create an async function `send_telegram_message(chat_id: str, text: str)`. It should use `settings.TELEGRAM_BOT_TOKEN`. Add error handling for 429 (Rate Limit)."

**Prompt 3 (The Router):**

"Update `src/interfaces/webhooks/router.py`. Add a specific route `POST /webhook/telegram`. It should: 1\. Parse the JSON into `TelegramUpdate`. 2\. Normalize it to `GovernorEvent`. 3\. Trigger the Governor Graph (background task). 4\. Return 200 OK immediately."

**Prompt 4 (Closing the Loop):**

"Update the `RESPOND` node in `src/governor/state_machine/nodes.py`. Inject the `send_telegram_message` function. Ensure that when the LLM generates a final answer, it is actually sent back to the user's `chat_id` (which must be passed through the graph state)."

