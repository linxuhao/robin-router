"""`robin` — start the router, or check the config without starting it."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from . import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="robin",
        description="Round-robin your LLM subscriptions.")
    parser.add_argument("--host", default=os.getenv("ROBIN_HOST", "127.0.0.1"),
                        help="listen address (default 127.0.0.1 — Robin holds "
                             "every key you own; do not expose it without auth)")
    parser.add_argument("--port", type=int,
                        default=int(os.getenv("ROBIN_PORT", "8080")))
    parser.add_argument("--providers", default=None,
                        help="path to llm_providers.json")
    parser.add_argument("--routes", default=None,
                        help="path to model_routes.json")
    parser.add_argument("--check", action="store_true",
                        help="validate the tables and key files, then exit")
    parser.add_argument("--init", action="store_true",
                        help="write starter config tables here, then exit")
    parser.add_argument("--version", action="version",
                        version=f"robin {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from .config import (ConfigError, Providers, Routes, secrets_dir,
                         write_starter_tables)

    if args.init:
        written = write_starter_tables(Path.cwd())
        for p in written:
            print(f"wrote {p.name}")
        if not written:
            print("llm_providers.json and model_routes.json already exist "
                  "— nothing written.")
        print(f"\nEdit them, then put your keys in {secrets_dir()}/ "
              f"(one file per key name) and run `robin --check`.")
        return 0

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
    from .config import secrets_dir, split_endpoint

    bad = routes.validate_against(providers)
    dead_models: list[str] = []
    print(f"providers: {', '.join(providers.names())}")
    for model in routes.names():
        cands = routes.candidates(model)
        n = routes.rotate_size(model)
        usable_here = False
        print(f"\n{model}:" + ("" if n else "   (ordered failover, no rotation)"))
        for i, endpoint in enumerate(cands):
            provider, _ = split_endpoint(endpoint)
            slot = "rotate  " if i < n else "fallback"
            if provider not in providers:
                mark, note = "!!", "provider NOT in providers table"
            elif not providers.key_env(provider):
                mark, note = "ok", "no key required"
                usable_here = True
            elif providers.key(provider):
                mark, note = "ok", f"key {providers.key_env(provider)} present"
                usable_here = True
            else:
                mark, note = "--", (f"no key file {providers.key_env(provider)}"
                                    f" — will be skipped")
            print(f"  {mark} {slot}  {endpoint:<44} {note}")
        if not usable_here:
            dead_models.append(model)
    if bad:
        print(f"\n{len(bad)} candidate(s) name an unregistered provider.",
              file=sys.stderr)
    if dead_models:
        # Per MODEL, not "did anything anywhere work": a config where `pro`
        # has no usable candidate but `flash` does is broken for every request
        # that asks for `pro`, and a green pre-flight check that says
        # otherwise is worse than no check.
        print(f"\nNo usable endpoint for: {', '.join(dead_models)}.\n"
              f"Write each key to {secrets_dir()}/<KEY_NAME> (chmod 600) — "
              f"the KEY_NAME is the one shown against each endpoint above.",
              file=sys.stderr)
        return 1
    if bad:
        return 1
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
