# Robin

*v0.1 — early, and honest about it: see [What it is not](#what-it-is-not).*

**Round-robin your LLM subscriptions.** One local OpenAI-compatible endpoint in
front of every plan you hold — so the 5-hour and weekly windows you already pay
for all get used, instead of one plan burning out while the others expire idle.

Born for DeepSeek refugees: instead of hunting for one plan with enough
capacity, find several whose capacity **adds up** — and let them add up.

```
your client  ──►  Robin  ──┬──►  ark/deepseek-v4-flash         (plan A: own window)
(Claude Code, opencode,    │     opencodego/deepseek-v4-flash  (plan B: own window)
 dsh — anything that       │     …add as many plans as you hold
 speaks /v1/chat/          └──►  deepseek/deepseek-v4-flash     (pay-as-you-go, last)
 completions)
```

## What's actually different

Sticky routing is not new and neither is load balancing. LiteLLM's router
spreads proactively (weighted shuffle, lowest-TPM, latency) and can pin by API
key or `session_id`; Portkey has sticky load balancing on a configured
`hash_fields`; Kong does consistent hashing on a header. The key-pool rotators
(`dsh-api-key-pool` and friends) are the ones that move only on failure.

Two things are still unserved:

- **Those gateways spread by tokens, requests or dollars — never by a
  subscription *window*.** A 5-hour bucket that refills when the provider says
  so, and is worth nothing if it expires unspent, is a different unit from
  TPM. If you hold three plans, that unit is the whole reason you hold them.
- **Every sticky implementation above needs the caller to hand over an
  identity** — an API key, a `session_id`, a header. Robin derives one from
  the request itself, so an unmodified OpenAI client gets stickiness with no
  cooperation at all.

(Closest prior art, and worth your time if it fits: `claude-relay-service` and
`ccflare` do window-aware account pooling for Claude subscriptions
specifically — `claude-relay-service` derives its session hash much the way
Robin does. Robin's difference is that it is local, single-process, and works
for any OpenAI-compatible plan.)

Robin moves on **conversation boundaries**, and gets two things right:

- **Prefix stickiness.** Provider prefix caches are per-provider, and agent
  workloads replay the whole transcript every turn (measured on one real
  workload — AItelier's, where this routing layer came from: 26:1 prefill:decode at an 89.4% cache hit rate). Rotating per
  *call* converts cached input into full-price input and costs more than a
  second plan saves. Robin keys on the conversation prefix: the same
  conversation stays on the same endpoint, a NEW one starts on the next plan.
  Quota spreads; caches survive.
- **Spent-window parking.** A plan whose window is spent is parked until the
  provider's own stated reset instant, keyed on `provider/model` — so later
  requests skip it instead of re-paying the same doomed call. Burst throttling
  and an exhausted plan are both HTTP 429; only the prose says which.

Pay-as-you-go endpoints are declared as `fallback` and never rotate forward:
they are what turns "everything stops until the window resets" into "the next
request goes elsewhere", and money should be the last resort, not a peer.

## What it is not

No cost tracking, no request logging, no persistence across restarts, one
process. If you want an observability plane or a team gateway, LiteLLM and
Portkey are the grown-ups. Robin is one job: spend the windows you already
bought.

**OpenAI protocol only, in and out.** Robin speaks `/v1/chat/completions` and
forwards to `/v1/chat/completions`. It does not accept Anthropic-format
(`/v1/messages`) requests, and it cannot route to an endpoint that only serves
that shape — which is a real gap inside plans people hold: OpenCode Go, for
one, serves DeepSeek/GLM/Kimi over the OpenAI shape but Qwen-Max and MiniMax
over `/v1/messages`. Adding it is additive (a provider gains an `api:` field),
not a rewrite; it is on the list if people want it. For pooling **Claude**
subscriptions specifically, `claude-relay-service` and `ccflare` already do
that job well.

Nothing here is DeepSeek-specific — any OpenAI-compatible base URL works; the
shipped examples just happen to be what the author holds.

## Configuration

Robin reads the same two tables AItelier uses, so an existing deployment can
point a client at Robin with no migration:

```jsonc
// llm_providers.json — a provider is (base_url, key NAME), NOT a vendor.
// Hold two plans with the same vendor? Register it twice under different names.
{
  "ark":        {"base_url": "https://ark.example/api/v3", "api_key_env": "ARK_API_KEY"},
  "opencodego": {"base_url": "https://opencode.ai/zen/go/v1", "api_key_env": "GO_API_KEY"}
}
```

```jsonc
// model_routes.json — a MODEL is an ordered list of ENDPOINTS.
{
  "flash": {"rotate":   ["ark/deepseek-v4-flash", "opencodego/deepseek-v4-flash"],
            "fallback": ["deepseek/deepseek-v4-flash"]}
}
```

Keys are read from files (`~/.robin-secrets/<NAME>`, or `$ROBIN_SECRETS_DIR`),
never from the config and never from the environment — so a config is safe to
share, and a subprocess that inherits the environment does not receive your
keys. See `.env.example`. A key name with **no file** means "I do not hold this
plan": Robin skips that endpoint rather than burning a call that cannot
succeed.

## Run it

```bash
pip install -e .

cp llm_providers.example.json llm_providers.json   # who you can call
cp model_routes.example.json  model_routes.json    # which plans serve what

mkdir -p ~/.robin-secrets && chmod 700 ~/.robin-secrets
( umask 077; printf '%s' "<your-key>" > ~/.robin-secrets/ARK_API_KEY )

robin --check      # says which endpoints are usable, and why the rest are not
robin              # http://127.0.0.1:8080/v1
```

```bash
pip install robin-router && robin --init && robin --check
```

Point any client that speaks `/v1/chat/completions` at
`http://127.0.0.1:8080/v1` and ask for a **route name** (`flash`, `pro`) as the
model. Streaming works. `/v1/completions` and `/v1/embeddings` do not exist yet.

| endpoint | what it's for |
|---|---|
| `GET /health` | routes and providers loaded |
| `GET /stats` | rotation cursors, live conversations, and what is parked — the answer to "why is everything landing on the expensive endpoint" |
| `GET /v1/models` | the route names, for clients that populate a picker |
| `POST /reload` | re-read both tables without dropping conversations or parks; a broken edit is refused and the old config keeps serving |
| `POST /unpark` | release a park (`?endpoint=…`, or all) — parking is inferred from provider prose and can be wrong |
| `x-robin-endpoint` header, `robin.served_by` in the body | which plan actually served this turn |

Key files need no reload: they are read per call, so writing a new plan's
key takes effect on the next request.

Settings are environment variables (`ROBIN_HOST`, `ROBIN_PORT`,
`ROBIN_SECRETS_DIR`, `ROBIN_PROVIDERS`, `ROBIN_ROUTES`, `ROBIN_API_KEY_FILE`) —
see `.env.example` for what each does. Robin does **not** read a `.env` file;
export them, or put them in whatever unit file starts it.

Robin listens on loopback by default and holds every key you own. Put auth in
front of it (or set `ROBIN_API_KEY_FILE`) before binding it to anything else.

## Licence

MIT.
