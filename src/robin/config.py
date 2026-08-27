"""The two tables, and the key files they point at.

Three levels, named so no two share a word:

  provider   one place to call        `ark`                (base_url + key NAME)
  endpoint   one model at one place   `ark/deepseek-v4-flash`
  model      an ordered list of endpoints, and what a CLIENT asks for

A provider is a (base_url, key name) pair, NOT a vendor. Two token plans with
the same vendor are two providers: register the vendor twice under different
names with different `api_key_env`, list both in a rotate pool, and the
spent-window cooldown (keyed on `provider/model`) parks them independently.
That is the whole reason this level exists.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


class ConfigError(RuntimeError):
    """A table is missing, malformed, or names something that isn't there."""


def _expand(p: str | os.PathLike) -> Path:
    return Path(os.path.expanduser(str(p)))


def config_or_example(name: str) -> Path:
    """`<name>` if it exists, else `<name>.example` — a clean checkout starts.

    Shipping only the real file makes a fresh clone fail on config rather than
    run; shipping only the example makes every user's first act a rename. Both
    exist, the real one wins.
    """
    p = _expand(name)
    if p.is_file():
        return p
    stem = p.with_suffix("")
    example = stem.with_name(stem.name + ".example" + p.suffix)
    if example.is_file():
        return example
    # Installed from a wheel there is no checkout to fall back to, and the
    # first command in the README then failed naming two files the user has
    # never seen. Ship the examples INSIDE the package as the last resort.
    packaged = Path(__file__).with_name(example.name)
    if packaged.is_file():
        return packaged
    raise ConfigError(
        f"neither {p} nor {example} exists. Run `robin --init` to write "
        f"starter tables into the current directory.")


def _load_json(path: Path) -> dict:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise ConfigError(f"{path}: {e}") from e
    except ValueError as e:
        # A malformed table must never degrade to "routing is off" — that looks
        # like a routing bug and hides the typo that caused it.
        raise ConfigError(f"{path}: could not be parsed: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected an object at the top level")
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def write_starter_tables(dest: Path) -> list[Path]:
    """Copy the packaged examples into `dest` as the real table names.

    Existing files are never overwritten: this is the command a user runs when
    they are not sure what they have, and clobbering a working config would be
    the worst possible answer to that question.
    """
    written = []
    for example in ("llm_providers.example.json", "model_routes.example.json"):
        src = Path(__file__).with_name(example)
        target = dest / example.replace(".example", "")
        if target.exists() or not src.is_file():
            continue
        target.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        written.append(target)
    return written


def secrets_dir() -> Path:
    return _expand(os.getenv("ROBIN_SECRETS_DIR") or "~/.robin-secrets")


def read_key(name: str) -> str | None:
    """A key's value, from its file. Empty or absent → None ("unused").

    Deliberately NOT the environment: an environment variable is inherited by
    every subprocess and shows up in `docker inspect`, `ps` and crash dumps. A
    file is read only when a call needs it.
    """
    if not name:
        return None
    try:
        val = (secrets_dir() / name).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return val or None


class Providers:
    """`name -> {base_url, api_key_env}`."""

    def __init__(self, path: str | os.PathLike | None = None):
        self.requested = str(path or os.getenv("ROBIN_PROVIDERS")
                             or "llm_providers.json")
        self.path = config_or_example(self.requested)
        self._p = _load_json(self.path)
        for name, cfg in self._p.items():
            if not isinstance(cfg, dict) or not cfg.get("base_url"):
                raise ConfigError(
                    f"{self.path}: provider '{name}' needs a base_url")

    def __contains__(self, name: str) -> bool:
        return name in self._p

    def names(self) -> list[str]:
        return sorted(self._p)

    def base_url(self, name: str) -> str:
        return self._p[name]["base_url"].rstrip("/")

    def key_env(self, name: str) -> str:
        return self._p[name].get("api_key_env") or ""

    def key(self, name: str) -> str | None:
        return read_key(self.key_env(name))


class Routes:
    """`model -> ordered candidates`, plus how many of them rotate.

    Two shapes:
      "flash": ["ark/m", "deepseek/m"]                  ordered failover only
      "flash": {"rotate": [...], "fallback": [...]}     rotate the head

    `rotate_size` is the boundary: candidates[:n] rotate, the rest are the
    ordered tail. Pay-as-you-go lives in the tail and never rotates forward —
    money is the last resort, not a peer.
    """

    def __init__(self, path: str | os.PathLike | None = None):
        self.requested = str(path or os.getenv("ROBIN_ROUTES")
                             or "model_routes.json")
        self.path = config_or_example(self.requested)
        self._r: dict[str, list[str]] = {}
        self._rot: dict[str, int] = {}
        for name, value in _load_json(self.path).items():
            cands, n = self._parse(name, value)
            self._r[name] = cands
            self._rot[name] = n

    def _parse(self, name: str, value) -> tuple[list[str], int]:
        if isinstance(value, str):
            value = [value]
        if isinstance(value, dict):
            unknown = set(value) - {"rotate", "fallback"}
            if unknown:
                raise ConfigError(
                    f"{self.path}: route '{name}' has unknown key(s) "
                    f"{sorted(unknown)} — only 'rotate' and 'fallback'")
            rotate = value.get("rotate")
            fallback = value.get("fallback", [])
            if (not isinstance(rotate, list) or not rotate
                    or not isinstance(fallback, list)):
                raise ConfigError(
                    f"{self.path}: route '{name}': 'rotate' must be a "
                    f"non-empty list and 'fallback' a list")
            cands, n = list(rotate) + list(fallback), len(rotate)
        elif isinstance(value, list) and value:
            cands, n = list(value), 0
        else:
            raise ConfigError(
                f"{self.path}: route '{name}' must be a non-empty list, a "
                f"string, or a rotate/fallback object")
        for c in cands:
            if not isinstance(c, str) or c.count("/") < 1 or c.startswith("/"):
                raise ConfigError(
                    f"{self.path}: route '{name}': candidate {c!r} must be "
                    f"'provider/model-id' (the model id may contain slashes)")
        return cands, n

    def __contains__(self, model: str) -> bool:
        return model in self._r

    def names(self) -> list[str]:
        return sorted(self._r)

    def candidates(self, model: str) -> list[str]:
        return list(self._r[model])

    def rotate_size(self, model: str) -> int:
        return self._rot.get(model, 0)

    def validate_against(self, providers: Providers) -> list[str]:
        """Endpoints naming a provider that is not registered.

        Returned rather than raised: an unusable candidate in a FALLBACK slot
        should be a loud warning at startup, not a refusal to start — but it
        must never be silent, because the failure it causes (a client-side bad
        request that no failover retries) points nowhere near the typo.
        """
        return [c for cands in self._r.values() for c in cands
                if c.split("/", 1)[0] not in providers]


def split_endpoint(endpoint: str) -> tuple[str, str]:
    """`provider/model-id` → (provider, model-id).

    Split ONCE: HF-style ids (`org/name`) are a normal shape for several
    OpenAI-compatible hosts, and a truncating split would hand the upstream
    half a model name.
    """
    provider, _, model = endpoint.partition("/")
    return provider, model
