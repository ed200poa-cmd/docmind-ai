# DocMind MCP Server

Exposes DocMind's RAG retrieval layer to any MCP client. The existing FastAPI app answers
questions itself; this server instead hands retrieval to the client's model, which is the
division of labor MCP assumes.

## Tools

| Tool | Purpose |
|---|---|
| `search_documents(question, doc_id?, top_k?)` | Semantic search over the FAISS index. Returns excerpts with `filename`, `page`, and `relevance_score` so the client can cite sources. |
| `list_documents()` | Every indexed document with its `doc_id` and chunk count. |
| `get_document_info(doc_id)` | Metadata for one document. |

Resource: `docmind://documents` — a readable summary of the collection.

## Design decisions

**Read-only.** `rag_engine.remove_document()` is deliberately not exposed. An MCP client
is a language model; document deletion is irreversible and belongs behind a human.

**Retrieval only, no synthesis.** `claude_qa.answer_question()` is not wrapped either.
Under MCP the client already is the model — the server's job ends at supplying excerpts
and citations. Wrapping the Claude call would mean paying for two models to answer one
question.

**stdout is the transport.** Under stdio, JSON-RPC frames travel on stdout, so anything
else written there corrupts the stream. `logging.basicConfig(stream=sys.stderr)` runs
before the project imports, and the fastembed/faiss calls are wrapped in a
`redirect_stdout(sys.stderr)` guard — fastembed prints model-download progress on first
use, which would otherwise break the session.

**Absolute database path.** `document_store.DB_PATH` resolves relative to the working
directory, but an MCP client spawns the server from an arbitrary cwd. The server pins the
path off `__file__`.

**Lazy index build.** The FAISS index is rebuilt on first tool call, not at import, so
client startup stays fast.

**`page` is null for plain text.** `rag_engine.parse_txt()` labels every chunk as page 1,
because a `.txt` file has no pagination. Passing that through would invite the client to
cite "page 1" as though it were a real location in a paginated document — a fabricated
citation in a tool built to prevent exactly that. The server returns `page: null` unless
the source is a PDF. Found by watching a real client cite the fake page number.

## Run

All commands below assume you have cloned the repository and are standing in its root.

```bash
cd docmind-ai
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python mcp_server.py               # stdio; a client normally spawns this for you
```

Running it directly is mainly a smoke test — it will sit waiting for JSON-RPC frames on
stdin. In normal use the MCP client starts the process for you.

## Register with Claude Code

An MCP client spawns the server from its own working directory, so the registration needs
absolute paths. Generate them from the repo root rather than hardcoding them:

```bash
cd docmind-ai
claude mcp add docmind -- "$(pwd)/.venv/bin/python" "$(pwd)/mcp_server.py"
```

Then in Claude Code: *"Search my documents for the vacation policy."* The model calls
`search_documents`, gets cited excerpts back, and answers from them.

## Dependencies

`mcp==1.28.1`, listed in `requirements.txt` alongside the application dependencies. The
server reuses the app's own retrieval modules, so no separate environment is needed.
