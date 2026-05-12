"""Write generated runtime kernel code to inspectable C files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .runtime_abi import RUNTIME_ABI_SOURCE
from .runtime_api import RunintModule


@dataclass(frozen=True)
class RuntimeCodeFiles:
    """Paths written by :func:`write_runtime_code`."""

    header: Path
    source: Path
    abi_header: Path


def _sanitize_identifier(name: str) -> str:
    """Return a C identifier-ish token for file prefixes and header guards."""
    token = re.sub(r"\W+", "_", name).strip("_")
    if not token:
        token = "runintgen"
    if token[0].isdigit():
        token = f"_{token}"
    return token


def _header_guard(prefix: str) -> str:
    """Return a stable C header guard."""
    token = _sanitize_identifier(prefix).upper()
    return f"RUNINTGEN_GENERATED_{token}_H"


def _abi_header_guard() -> str:
    """Return the runtime ABI header guard."""
    return "RUNINTGEN_RUNTIME_ABI_H"


def _needs_complex_header(module: RunintModule) -> bool:
    """Return true if any generated kernel declaration uses C complex types."""
    return any(
        (kernel.scalar_type or "").startswith("complex") for kernel in module.kernels
    )


def format_runtime_header(module: RunintModule, prefix: str) -> str:
    """Format a compact C header for generated runtime kernels."""
    guard = _header_guard(prefix)
    declarations = "\n".join(kernel.c_declaration for kernel in module.kernels)
    complex_include = "#include <complex.h>\n" if _needs_complex_header(module) else ""
    return f"""#ifndef {guard}
#define {guard}

{complex_include}#include <stdint.h>
#include <ufcx.h>

#ifdef __cplusplus
extern "C" {{
#endif

{declarations}

#ifdef __cplusplus
}}
#endif

#endif  // {guard}
"""


def format_runtime_abi_header() -> str:
    """Format the shared runtime ABI header used by generated sources."""
    guard = _abi_header_guard()
    return f"""#ifndef {guard}
#define {guard}

#include <stdint.h>

#ifdef __cplusplus
extern "C" {{
#endif

{RUNTIME_ABI_SOURCE}

#ifdef __cplusplus
}}
#endif

#endif  // {guard}
"""


def format_runtime_source(module: RunintModule, prefix: str) -> str:
    """Format C source for runtime kernels."""
    safe_prefix = _sanitize_identifier(prefix)
    definitions = "\n\n".join(kernel.c_definition for kernel in module.kernels)
    return f"""#include <complex.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "runintgen_runtime_abi.h"
#include "{safe_prefix}.h"

{definitions}
"""


def write_runtime_code(
    module: RunintModule,
    prefix: str,
    output_dir: str | Path,
) -> RuntimeCodeFiles:
    """Write generated runtime kernel header, source, and ABI header files.

    Args:
        module: Result from :func:`runintgen.compile_runtime_integrals`.
        prefix: File prefix, for example ``"laplace_p2"``.
        output_dir: Directory where files should be written.

    Returns:
        Paths to the generated header, source, and ABI header files.
    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    safe_prefix = _sanitize_identifier(prefix)

    header = outdir / f"{safe_prefix}.h"
    source = outdir / f"{safe_prefix}.c"
    abi_header = outdir / "runintgen_runtime_abi.h"

    header.write_text(format_runtime_header(module, safe_prefix), encoding="utf-8")
    source.write_text(format_runtime_source(module, safe_prefix), encoding="utf-8")
    abi_header.write_text(format_runtime_abi_header(), encoding="utf-8")

    return RuntimeCodeFiles(
        header=header,
        source=source,
        abi_header=abi_header,
    )
