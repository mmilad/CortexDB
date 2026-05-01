# CortexDB Architecture Plan (AI/LLM/Agentic Brain Layer)

## 0) Critical Product Constraint (Must-Have)

CortexDB is a **service-layer memory database** and must **not run LLM inference or agent reasoning internally**.

- No internal model prompting/orchestration.
- No hidden AI decision logic.
- All embeddings, intent, and query intent come from caller-provided inputs.
- CortexDB provides deterministic storage, indexing, retrieval, filtering, scoring, and schema/tool discoverability.

Design principle: **"LLM outside, memory intelligence inside."**

## 1) Core Product Boundaries

CortexDB should act as a **memory and retrieval layer** for agentic systems, not as a full OLTP replacement.

Primary responsibilities:
- Store structured operational data (sessions, agents, tasks, artifacts).
- Store unstructured data (chat logs, notes, issue reports, documents).
- Store embeddings and support semantic/vector search.
- Provide scoped retrieval so each specialized agent sees only relevant memory.
- Provide hybrid retrieval (metadata + keyword + vector).
- Publish machine-readable schema and capability metadata for consumer apps.

Non-goals (initially):
- Complex analytics warehouse workflows.
- Heavy ETL orchestration.
- Full text indexing beyond what is needed for retrieval quality.
- Embedded LLM/AI orchestration logic inside CortexDB.

## 2) Data Domains to Model Early

Define these as first-class entities before API work:

1. **Tenant / Workspace**
   - Root ownership boundary.
   - Useful for SaaS and enterprise isolation.

2. **Agent**
   - Individual specialized agent identity.
   - Configuration for visibility scope and retention profile.

3. **Memory Namespace**
   - A logical partition inside a tenant.
   - Examples: `global_knowledge`, `support_agent_1`, `incident_agent`.

4. **Conversation Session**
   - Tracks runtime chats/process threads.
   - Attach participants, model metadata, tool calls.

5. **Memory Item**
   - Generic document unit: chat turn, issue, note, KB chunk, process event.
   - Includes metadata + optional embedding references.

6. **Knowledge Source + Chunk**
   - Source file/page + chunked fragments.
   - Chunk granularity should be explicit and reproducible.

7. **Known Issue / Incident Record**
   - Structured schema for known failures, symptoms, workaround, fix status.
   - Often high-value for retrieval in agent workflows.

8. **Dataset Registry**
   - Registry table for dynamic datasets (e.g., `tech_knowledge`).
   - Stores dataset contract, usage notes, ownership, and lifecycle.

9. **Tool Registry**
   - Tool catalog table with metadata and embeddings for tool discovery.
   - Supports vector search for tool selection by external applications.

## 3) Dynamic Schema + Capability Discoverability

Because datasets and tools evolve, CortexDB should expose explicit discoverability metadata.

### 3.1 Dataset Registry Contract

For each dataset entry (example: `tech_knowledge`) store:
- `dataset_key` (stable id)
- `display_name`
- `table_refs` (physical table/view references)
- `schema_version`
- `semantic_description` (what this dataset is for)
- `usage_guidance` (when to query / not query)
- `relationship_hints` (how it joins/relates to other datasets)
- `filterable_fields`
- `retrieval_profiles` (recommended scoring presets)
- `status` (active/deprecated)

### 3.2 Tool Registry Contract

For each tool entry store:
- `tool_key` (stable id)
- `name`
- `description`
- `input_schema_ref` (JSON schema pointer)
- `output_schema_ref`
- `capability_tags`
- `safety_scope` / constraints
- `relationship_hints` (which datasets/sessions it applies to)
- `embedding_vector` + `embedding_model_version`
- `status`

### 3.3 Why This Matters

Human developers can infer usage informally; consumer apps cannot.
CortexDB must publish this metadata so consumers can deterministically know:
- that datasets/tools exist,
- when to use them,
- and how they relate to the rest of the graph.

## 4) MCP Exposure Strategy (Dynamic/Generic MCP)

CortexDB should expose a dynamic MCP-compatible capability layer generated from registries.

- Capability discovery should be data-driven from Dataset/Tool Registry rows.
- Adding a new dataset/tool should update discoverable MCP resources without code branching per dataset.
- MCP exposure must include schema references and usage guidance metadata.
- Versioning: capability descriptors must include revision/version for client compatibility.

Practical outcome: external agent frameworks discover and use CortexDB capabilities without embedding assumptions.

## 5) Isolation and Access Strategy

You need two isolation layers:

- **Hard isolation**: tenant/workspace boundary (security critical).
- **Soft isolation**: per-agent namespace scoping (relevance critical).

Recommended retrieval filter pipeline:
1. Resolve requester identity (tenant + agent/service role).
2. Expand allowed namespaces/policies.
3. Apply metadata filters (time, source type, status, tags).
4. Apply vector/keyword search over the filtered candidate set.
5. Rerank and return with provenance.

This avoids irrelevant recall and reduces accidental cross-agent leakage.

## 6) Storage Architecture (Pragmatic v1)

A practical v1 architecture:

- **Relational store (Postgres)**
  - System of record for entities, metadata, permissions, sessions, registries.
- **Vector index (pgvector in Postgres or dedicated vector DB later)**
  - Start with pgvector for operational simplicity.
  - Migrate high-scale workloads to dedicated vector service when needed.
- **Object storage (optional but likely soon)**
  - Raw document blobs / large artifacts.

Why this works: one operational plane early, easier consistency, fewer moving parts.


## 6.1 Lightweight DB Options (Researched)

If you prefer lightweight deployments, prioritize these options:

1. **SQLite + vec1/sqlite-vector extension**
   - Best for single-node/embedded deployments and very small operational overhead.
   - SQLite now documents ANN vector search via the `vec1` extension.
   - Tradeoff: fewer distributed/HA capabilities.

2. **DuckDB + VSS extension**
   - Embedded analytical DB with vector similarity search support through `vss`.
   - Good for local-first analytics + vector workflows.
   - Tradeoff: extension maturity/operational behavior should be validated for production write-heavy paths.

3. **LanceDB (embedded)**
   - Embedded open-source vector DB style deployment for local workflows.
   - Good fit when vector retrieval is primary and you want local library-style usage.

4. **Qdrant (lightweight service mode)**
   - Still a service process, but commonly used for focused vector workloads with simple self-hosting.
   - Better when you outgrow purely embedded options and need stronger vector-only operations.

### Suggested default path

- **Phase A (fastest):** SQLite (+ vector extension) for proof-of-concept.
- **Phase B (balanced):** Postgres + pgvector for mixed relational + vector production baseline.
- **Phase C (scale):** Add dedicated vector engine (e.g., Qdrant) only when needed by scale/latency.

This keeps architecture simple while preserving an upgrade path.

## 7) Retrieval Patterns You Should Support

1. **Scoped semantic retrieval**
   - "Find similar memory in this agent namespace."

2. **Hybrid retrieval**
   - Combine keyword constraints + vector similarity.
   - Example: only `known_issue` records in `status=open` then semantic rank.

3. **Temporal retrieval**
   - Time windows for "current process" memory.

4. **Session-context retrieval**
   - Pull latest N interactions + semantically relevant older memory.

5. **Cross-namespace fallback (policy-gated)**
   - If no high-confidence local hit, search approved shared namespace.

6. **Tool discovery retrieval**
   - Vector + tag search over Tool Registry to identify relevant tools.

## 8) Filtering + Scoring Controls (DB Smartness Without LLM Logic)

To keep the service "smart" without embedded AI logic, provide configurable deterministic scoring.

### 8.1 Score Composition

- `final_score = a*vector_score + b*keyword_score + c*recency + d*source_trust + e*agent_affinity + f*schema_match`

### 8.2 Scoring Profiles

- Named profiles stored in DB (`default_memory`, `incident_response`, `tool_selection`, etc).
- Profiles define weights, thresholds, and hard filters.
- Caller selects profile explicitly; CortexDB applies it deterministically.

### 8.3 Provenance & Explainability

Keep provenance for every hit:
- source id
- chunk/tool id
- timestamp
- namespace
- score breakdown per component
- profile used

This is critical for debugging and auditability.

## 9) Ingestion Pipeline Design

Standardize ingestion stages:
1. Normalize payload.
2. Classify data type (`chat`, `knowledge`, `issue`, `event`, `tool`).
3. Chunk (if needed).
4. Embed using caller-selected/configured model/version.
5. Persist metadata + vectors atomically (or idempotently).
6. Record ingestion audit/log.

Version every embedding model used. You will need re-embedding jobs later.

## 10) Lifecycles and Retention

Define retention per memory class:
- Ephemeral process state (short TTL).
- Session chat history (medium).
- Knowledge base and known issues (long-term).
- Tool/Dataset registry records (long-term, versioned).

Support:
- soft delete
- hard delete
- legal hold flags (if enterprise use)

## 11) Suggested First API Capability Map (Pre-Endpoint)

Before endpoint naming, define capability groups:

- Identity & scope resolution.
- Write memory item (+ optional vector).
- Batch ingest documents/tools.
- Query memory (filter + vector + hybrid).
- Session history append/read.
- Known issue CRUD + resolution timeline.
- Namespace/agent policy management.
- Dataset registry CRUD + discoverability endpoints.
- Tool registry CRUD + tool retrieval endpoints.
- Re-embed and index maintenance jobs.
- MCP capability descriptor/read endpoints.

## 12) Minimal v1 Milestones

**Milestone 1: Foundation**
- Tenant, agent, namespace models.
- Memory item schema + metadata filters.
- Dataset/Tool registries.

**Milestone 2: Semantic retrieval**
- Embeddings + vector index + scoped similarity search.
- Tool retrieval and schema-aware filtering.

**Milestone 3: Hybrid + quality**
- Keyword + vector blend, reranking, provenance output.
- Deterministic scoring profiles.

**Milestone 4: Dynamic MCP exposure**
- Registry-backed capability descriptors and versioned discovery.

**Milestone 5: Operations**
- Retention jobs, re-embedding jobs, observability dashboards.

## 13) Risks to Address Early

- Over-sharing between agents because of weak namespace policy.
- Poor chunking that degrades retrieval quality.
- Missing provenance makes debugging impossible.
- No embedding versioning leads to silent relevance regressions.
- Dynamic dataset/tool metadata drift from actual table schemas.
- Accidental reintroduction of hidden AI logic into service layer.

## 14) Recommended Next Step

Create a **domain model spec** with:
- ERD-level entities and relationships.
- Mandatory metadata fields per memory/tool/dataset type.
- Scoring profile schema and filter DSL.
- Capability descriptor schema for MCP publication.

Once this is stable, API design becomes significantly easier.
