"""`robin` — start the router, or check the config without starting it."""
from __future__ import annotations

import argparse
import logging
import os
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="robin",
        description="Round-robin your LLM subscriptions.")
    parser.add_argument("--host", default=os.getenv("ROBIN_HOST", "127.0.0.1"),
                        help="listen address (default 127.0.0.1 — Robin holds "
                             "every key you own; do not expose it without auth)")
    parser.add_argument("--port", type=int,
                        default=int(os.getenv("ROBIN_PORT", "8080")))
    parser.add_argument("--providers", default=None)
    parser.add_argument("--routes", default=None)
    parser.add_argument("--check", action="store_true",
                        help="validate the tables and key files, then exit")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from .config import ConfigError, Providers, Routes

    try:
        providers = Providers(args.providers)
        routes = Routes(args.routes)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2

    if args.check:
        return _check(providers, routes)

    from .server import create_app
    import uvicorn

    app = create_app(providers, routes)
    print(f"Robin on http://{args.host}:{args.port}/v1  "
          f"({len(routes.names())} models, {len(providers.names())} providers)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def _check(providers, routes) -> int:
    """Say what WILL happen, before a client depends on it.

    Deliberately verbose about key presence: "which of my plans is Robin
    actually able to use" is the question every misconfiguration reduces to,
    and answering it at startup is cheaper than discovering it mid-run.
    """
    from .config import split_endpoint

    bad = routes.validate_against(providers)
    usable_any = False
    print(f"providers: {', '.join(providers.names())}")
    for model in routes.names():
        cands = routes.candidates(model)
        n = routes.rotate_size(model)
        print(f"\n{model}:" + ("" if n else "   (ordered failover, no rotation)"))
        for i, endpoint in enumerate(cands):
            provider, _ = split_endpoint(endpoint)
            slot = "rotate  " if i < n else "fallback"
            if provider not in providers:
                mark, note = "!!", "provider NOT in providers table"
            elif not providers.key_env(provider):
                mark, note = "ok", "no key required"
                usable_any = True
            elif providers.key(provider):
                mark, note = "ok", f"key {providers.key_env(provider)} present"
                usable_any = True
            else:
                mark, note = "--", (f"no key file {providers.key_env(provider)}"
                                    f" — will be skipped")
            print(f"  {mark} {slot}  {endpoint:<44} {note}")
    if bad:
        print(f"\n{len(bad)} candidate(s) name an unregistered provider.",
              file=sys.stderr)
    if not usable_any:
        print("\nNo endpoint is usable: every candidate is missing its key "
              "file. See .env.example.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
