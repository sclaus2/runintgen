"""Helpers for packaged C++ support headers."""

from __future__ import annotations

from pathlib import Path


def runtime_abi_header_path() -> Path:
    """Return the canonical packaged runtime ABI header path."""
    return Path(__file__).with_name("cpp") / "runintgen_runtime_abi.h"


def runtime_abi_header_text() -> str:
    """Return the canonical runtime ABI header text."""
    return runtime_abi_header_path().read_text(encoding="utf-8")


def runtime_abi_cdef() -> str:
    """Return runtime ABI declarations suitable for ``cffi.FFI.cdef``.

    The C++ header is the source of truth. This strips only preprocessor and
    ``extern "C"`` wrapper lines that CFFI cannot parse.
    """
    lines = []
    for line in runtime_abi_header_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if stripped == 'extern "C" {':
            continue
        if stripped == "}":
            continue
        lines.append(line)
    return "\n".join(lines).strip() + "\n"
