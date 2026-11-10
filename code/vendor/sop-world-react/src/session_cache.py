from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class PromptSessionCache:
    prompt_ids: tuple[int, ...] | None = None
    past_key_values: Any = None
    last_logits: Any = None

    def reset(self) -> None:
        self.prompt_ids = None
        self.past_key_values = None
        self.last_logits = None

    def can_extend(self, prompt_ids: Sequence[int]) -> bool:
        if self.prompt_ids is None:
            return False
        current = tuple(prompt_ids)
        cached = self.prompt_ids
        return len(current) >= len(cached) and current[: len(cached)] == cached

    def delta_tokens(self, prompt_ids: Sequence[int]) -> tuple[int, ...]:
        if not self.can_extend(prompt_ids):
            raise ValueError("Prompt does not extend the cached session")
        current = tuple(prompt_ids)
        cached = self.prompt_ids or ()
        return current[len(cached) :]

    def store(
        self,
        *,
        prompt_ids: Sequence[int],
        past_key_values: Any,
        last_logits: Any,
    ) -> None:
        self.prompt_ids = tuple(prompt_ids)
        self.past_key_values = past_key_values
        self.last_logits = last_logits


@dataclass(frozen=True)
class PromptCacheKey:
    family: str
    prefix_hash: str


class SessionCacheRegistry:
    def __init__(self) -> None:
        self._caches: dict[PromptCacheKey, PromptSessionCache] = {}

    @staticmethod
    def prefix_hash(prefix_text: str) -> str:
        return hashlib.sha256(prefix_text.encode("utf-8")).hexdigest()

    def key(self, family: str, prefix_text: str) -> PromptCacheKey:
        normalized_family = family.strip() or "default"
        return PromptCacheKey(
            family=normalized_family,
            prefix_hash=self.prefix_hash(prefix_text),
        )

    def get(self, family: str, prefix_text: str) -> PromptSessionCache:
        cache_key = self.key(family, prefix_text)
        cache = self._caches.get(cache_key)
        if cache is None:
            cache = PromptSessionCache()
            self._caches[cache_key] = cache
        return cache

    def reset_family(self, family: str) -> None:
        normalized_family = family.strip() or "default"
        for cache_key in list(self._caches):
            if cache_key.family == normalized_family:
                self._caches.pop(cache_key, None)

    def reset_all(self) -> None:
        self._caches.clear()


@dataclass
class PrefixTokenCache:
    prefix_text: str | None = None
    input_ids: tuple[int, ...] | None = None
    attention_mask: tuple[int, ...] | None = None
    verified_splice: bool = False

    def reset(self) -> None:
        self.prefix_text = None
        self.input_ids = None
        self.attention_mask = None
        self.verified_splice = False

    def store(
        self,
        *,
        prefix_text: str,
        input_ids: Sequence[int],
        attention_mask: Sequence[int],
        verified_splice: bool,
    ) -> None:
        self.prefix_text = prefix_text
        self.input_ids = tuple(input_ids)
        self.attention_mask = tuple(attention_mask)
        self.verified_splice = verified_splice


class PrefixTokenCacheRegistry:
    def __init__(self) -> None:
        self._caches: dict[PromptCacheKey, PrefixTokenCache] = {}

    def get(self, family: str, prefix_text: str) -> PrefixTokenCache:
        cache_key = PromptCacheKey(
            family=family.strip() or "default",
            prefix_hash=SessionCacheRegistry.prefix_hash(prefix_text),
        )
        cache = self._caches.get(cache_key)
        if cache is None:
            cache = PrefixTokenCache()
            self._caches[cache_key] = cache
        return cache

    def reset_family(self, family: str) -> None:
        normalized_family = family.strip() or "default"
        for cache_key in list(self._caches):
            if cache_key.family == normalized_family:
                self._caches.pop(cache_key, None)

    def reset_all(self) -> None:
        self._caches.clear()
