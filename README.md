# Robin

**Round-robin your LLM subscriptions.** One local OpenAI-compatible endpoint in
front of every plan you hold — so the 5-hour and weekly windows you already pay
for all get used, instead of one plan burning out while the others expire idle.

Instead of hunting for one plan with enough capacity, hold several whose
capacity **adds up** — and let it add up.

```
your client  ──►  Robin  ──┬──►  ark/deepseek-v4-flash         (plan A: own window)
(Claude Code, opencode,    │     opencodego/deepseek-v4-flash  (plan B: own window)
 dsh — anything that       │     …add as many plans as you hold
 speaks /v1/chat/          └──►  deepseek/deepseek-v4-flash     (pay-as-you-go, last)
 completions)
```

```bash
pip install robin-router
robin --init      # writes starter config into the current directory
robin --check     # says which endpoints are usable, and why the rest are not
robin             # http://127.0.0.1:8080/v1
```

## How it routes

**The unit is the conversation, not the request.** The same conversation keeps
hitting the same plan; a new conversation starts on the next one. Quota
spreads, and the provider-side prefix cache survives — an agent replays its
whole transcript every turn, so a per-*call* decision turns a cache hit into
full-price prefill and costs more than a second plan saves.

Robin recognises a conversation from its own content — the system prompt plus
the first user message — so an unmodified client gets this with no cooperation
and no session id to pass.

**A spent window is parked until it reopens.** Robin reads the provider's own
`Retry-After` or stated reset instant and skips that endpoint until then,
rather than re-paying the same doomed call on every request. Burst throttling
and an exhausted plan are both HTTP 429, and the distinction is inferred from
the response — so `GET /stats` shows what is parked and `POST /unpark`
releases it if the guess was wrong.

**Pay-as-you-go is the tail, never a peer.** Endpoints listed under `fallback`
are tried only when every plan in the pool has failed or is parked. They are
what turns "everything stops until the window resets" into "the next request
goes elsewhere".

## Configure

Two files. `robin --init` writes both.

```jsonc
// llm_providers.json — a provider is (base_url, key NAME), NOT a vendor.
// Two plans with the same vendor? Register it twice under different names;
// their windows are then tracked, and parked, independently.
{
  "ark":        {"base_url": "https://ark.example/api/v3",     "api_key_env": "ARK_API_KEY"},
  "opencodego": {"base_url": "https://opencode.ai/zen/go/v1",  "api_key_env": "GO_API_KEY"},
  "deepseek":   {"base_url": "https://api.deepseek.com/v1",    "api_key_env": "DEEPSEEK_API_KEY"}
}
```

```jsonc
// model_routes.json — a MODEL is what clients ask for; it names ENDPOINTS.
{
  // rotate: one plan per new conversation.  fallback: only when all else fails.
  "flash": {"rotate":   ["ark/deepseek-v4-flash", "opencodego/deepseek-v4-flash"],
            "fallback": ["deepseek/deepseek-v4-flash"]},

  // A plain list is ordered failover with no rotation.
  "local": ["localvllm/qwen3.8-27b"]
}
```

Keys live in files, one per key name:

```bash
mkdir -p ~/.robin-secrets && chmod 700 ~/.robin-secrets
( umask 077; printf '%s' "<your-key>" > ~/.robin-secrets/ARK_API_KEY )
```

`api_key_env` names the file. Nothing reads a key from the config or the
environment, so a config is safe to share and a subprocess that inherits the
environment never receives your keys. **A key name with no file means "I do not
hold this plan"** — Robin skips that endpoint instead of burning a call that
cannot succeed, so adding or dropping a plan is one file, no restart.

## Use

Point any client that speaks `/v1/chat/completions` at
`http://127.0.0.1:8080/v1` and ask for a **route name** (`flash`) as the model.
Streaming works.

| endpoint | |
|---|---|
| `GET /health` | routes and providers loaded |
| `GET /stats` | rotation cursors, live conversations, and what is parked |
| `GET /v1/models` | the route names, for clients with a model picker |
| `POST /reload` | re-read both tables; conversations and parks survive, and a broken edit is refused with the old config still serving |
| `POST /unpark` | release a park — one `?endpoint=…`, or all |
| `x-robin-endpoint` header · `robin.served_by` in the body | which plan served this turn |

Settings are environment variables: `ROBIN_HOST`, `ROBIN_PORT`,
`ROBIN_SECRETS_DIR`, `ROBIN_PROVIDERS`, `ROBIN_ROUTES`, `ROBIN_API_KEY_FILE`
(see `.env.example`). Robin does not read a `.env` file — export them, or set
them in whatever starts it.

Robin listens on loopback and holds every key you own. Put auth in front of it
before binding it anywhere else.

## What it is not

v0.1, and one job: spend the windows you already bought. No cost tracking, no
request logging, no persistence across restarts, one process. For an
observability plane or a team gateway, LiteLLM and Portkey are the grown-ups —
and they load-balance proactively and can pin traffic too, by API key, session
id or header. What they measure is tokens, requests and dollars; Robin
measures the subscription window.

**OpenAI protocol only**, in and out. Anthropic-format (`/v1/messages`)
requests are not accepted and endpoints that only serve that shape cannot be
routed to. For pooling **Claude** subscriptions specifically,
[claude-relay-service](https://github.com/Wei-Shaw/claude-relay-service) and
[ccflare](https://github.com/snipeship/ccflare) already do that job well.

Nothing here is DeepSeek-specific — any OpenAI-compatible base URL works,
including a local vLLM or Ollama box.

## Develop

```bash
pip install -e ".[dev]" && pytest -q
```

## Licence

MIT.
