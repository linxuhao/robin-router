# Robin

**Round-robin your LLM subscriptions.** One local OpenAI-compatible endpoint in
front of every plan you hold — so the 5-hour and weekly windows you already pay
for all get used, instead of one plan burning out while the others expire idle.

Born for DeepSeek refugees: prices moved, everyone now holds two or three
subscriptions plus a pay-as-you-go key, and every client speaks to exactly one
of them at a time.

```
your client  ──►  Robin  ──►  ark/deepseek-v4-flash     (plan A: $12 / 5h)
(dsh, opencode,        │      qwen/qwen3.8-flash        (plan B: own window)
 Claude Code router,   │      opencodego/deepseek-v4-flash (plan C: $60/mo)
 anything OpenAI-      └────► deepseek/deepseek-v4-flash  (pay-as-you-go, last)
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
  workload: 26:1 prefill:decode at an 89.4% cache hit rate). Rotating per
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

Keys are read from files (`~/.aitelier-secrets/<NAME>`, or
`$ROBIN_SECRETS_DIR`), never from the config — so a config is safe to share and
a subprocess that inherits the environment does not receive your keys.

## Licence

MIT.
