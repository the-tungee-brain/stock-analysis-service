from __future__ import annotations

from collections.abc import Iterable


RAW_CONTEXT_MARKERS = (
    "Portfolio AI context",
    "portfolio_ai_context_v1",
    "AIContextBuilder",
    '"strategy_policy"',
    '"user_profile"',
    '"app_intelligence"',
    '"market_context"',
    '"relevant_symbol_intelligence"',
)


class ChatOutputGuard:
    """Best-effort protection against raw internal context leaking into chat."""

    def __init__(self, *, allow_structured_examples: bool = False) -> None:
        self._buffer = ""
        self._allow_structured_examples = allow_structured_examples
        self._inside_code_block = False

    def feed(self, chunk: str) -> list[str]:
        if not chunk:
            return []
        self._buffer += chunk
        lines = self._buffer.splitlines(keepends=True)
        if lines and not lines[-1].endswith(("\n", "\r")):
            self._buffer = lines.pop()
        else:
            self._buffer = ""
        return list(self._visible_lines(lines))

    def flush(self) -> list[str]:
        if not self._buffer:
            return []
        pending = self._buffer
        self._buffer = ""
        return list(self._visible_lines([pending]))

    @classmethod
    def sanitize_text(
        cls,
        text: str,
        *,
        allow_structured_examples: bool = False,
    ) -> str:
        guard = cls(allow_structured_examples=allow_structured_examples)
        return "".join([*guard.feed(text), *guard.flush()])

    def _visible_lines(self, lines: Iterable[str]) -> Iterable[str]:
        for line in lines:
            if line.strip().startswith("```") and "ai context" in line.lower():
                continue
            if self._allow_structured_examples and line.strip().startswith("```"):
                self._inside_code_block = not self._inside_code_block
                yield line
                continue
            if self._inside_code_block and self._allow_structured_examples:
                yield line
                continue
            if self._is_raw_context_line(line):
                continue
            yield line

    def _is_raw_context_line(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        if any(marker in stripped for marker in RAW_CONTEXT_MARKERS):
            return True
        if (
            not self._allow_structured_examples
            and stripped.startswith("{")
            and '"portfolio"' in stripped
        ):
            return True
        return False
