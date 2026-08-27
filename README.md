# Robin

**Round-robin your LLM subscriptions.** One local OpenAI-compatible endpoint in
front of every plan you hold — so the 5-hour and weekly windows you already pay
for all get used, instead of one plan burning out while the others expire idle.

Born for DeepSeek refugees: instead of hunting for one plan with enough
capacity, find several whose capacity **adds up** — and let them add up.

```
your client  ──►  Robin  ──►  ark/deepseek-v4-flash        (plan A: own window)
(dsh, opencode,        │      opencodego/deepseek-v4-flash (plan B: own window)
 Claude Code router,   │      …add as many as you hold
 any /v1/chat/    └────► deepseek/deepseek-v4-flash    (pay-as-you-go, last)
 compatible)
```

## What makes it different from a plain key rotator

The rotators that exist today move **on failure**: they burn one key to
exhaustion, take the 429, then try the next. That leaves the other plans'
windows expiring unused, and it makes the switch happen at the worst possible
moment — mid-conversation.

Robin moves on **conversation boundaries**, and it knows two things a plain
rotator does not:

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

## Status

Early. The routing model and the parking logic are lifted from
[AItelier](https://github.com/linxuhao/AItelier)'s in-process router, where they
have been driving real multi-day pipeline runs across three subscription plans.
Robin is that layer made client-agnostic.

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
printf '%s' "<your-key>" > ~/.robin-secrets/ARK_API_KEY && chmod 600 "$_"

robin --check      # says which endpoints are usable, and why the rest are not
robin              # http://127.0.0.1:8080/v1
```

Point any client that speaks `/v1/chat/completions` at
`http://127.0.0.1:8080/v1` and ask for a **route name** (`flash`, `pro`) as the
model. Streaming works. `/v1/completions` and `/v1/embeddings` do not exist yet.

| | |
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
