"""Write generated runtime kernel code to inspectable C files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cpp_headers import runtime_abi_header_path, runtime_abi_header_text
from .runtime_api import RunintModule


@dataclass(frozen=True)
class RuntimeCodeFiles:
    """Paths produced by :func:`write_runtime_code`."""

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


def _basix_hash(element: Any) -> int:
    """Return a Basix element hash, or zero when unavailable."""
    basix_element = element
    if hasattr(element, "_element"):
        basix_element = element._element
    elif hasattr(element, "basix_element"):
        basix_element = element.basix_element

    if hasattr(basix_element, "basix_hash"):
        return int(basix_element.basix_hash())
    if hasattr(basix_element, "hash"):
        return int(basix_element.hash())
    return 0


def _role_enum_name(role_name: str) -> str:
    """Return the C enum constant for a form element role."""
    return f"RUNINTGEN_ELEMENT_{role_name.upper()}"


def _format_value_shape(values: tuple[int, ...]) -> str:
    """Return a four-entry C initializer for an element value shape."""
    padded = list(values[:4])
    padded.extend([0] * (4 - len(padded)))
    return f"{{{padded[0]}, {padded[1]}, {padded[2]}, {padded[3]}}}"


def _format_form_descriptor_source(module: RunintModule, prefix: str) -> str:
    """Format C definitions for the form-level element descriptor."""
    safe_prefix = _sanitize_identifier(prefix)
    descriptor_name = f"runintgen_form_descriptor_{safe_prefix}"
    elements_name = f"runintgen_form_elements_{safe_prefix}"
    metadata = module.form_metadata

    if metadata is None or len(metadata.unique_elements) == 0:
        return f"""const runintgen_form_descriptor {descriptor_name} =
{{
  .num_elements = 0,
  .elements = NULL,
}};
"""

    lines = [
        f"static const runintgen_form_element_descriptor {elements_name}"
        f"[{len(metadata.unique_elements)}] =",
        "{",
    ]
    for element in metadata.unique_elements:
        key = element.element_key
        lines.extend(
            [
                "  {",
                f"    .form_element_index = {element.form_elem_index},",
                f"    .role = {_role_enum_name(element.role.name)},",
                f"    .role_index = {element.index},",
                f"    .basix_hash = UINT64_C({_basix_hash(element.element)}),",
                f"    .family = {key.family},",
                f"    .cell_type = {key.cell_type},",
                f"    .degree = {key.degree},",
                f"    .value_rank = {len(key.value_shape)},",
                f"    .value_shape = {_format_value_shape(key.value_shape)},",
                f"    .block_size = {element.ncomps},",
                f"    .discontinuous = {int(key.discontinuous)},",
                "  },",
            ]
        )
    lines.extend(
        [
            "};",
            "",
            f"const runintgen_form_descriptor {descriptor_name} =",
            "{",
            f"  .num_elements = {len(metadata.unique_elements)},",
            f"  .elements = {elements_name},",
            "};",
            "",
        ]
    )
    return "\n".join(lines)


def format_runtime_header(module: RunintModule, prefix: str) -> str:
    """Format a compact C header for generated runtime kernels."""
    guard = _header_guard(prefix)
    safe_prefix = _sanitize_identifier(prefix)
    declarations = "\n".join(kernel.c_declaration for kernel in module.kernels)
    complex_include = "#include <complex.h>\n" if _needs_complex_header(module) else ""
    descriptor_declaration = (
        f"extern const runintgen_form_descriptor "
        f"runintgen_form_descriptor_{safe_prefix};"
    )
    return f"""#ifndef {guard}
#define {guard}

{complex_include}#include <stdint.h>
#include <ufcx.h>
#include "runintgen_runtime_abi.h"

#ifdef __cplusplus
extern "C" {{
#endif

{descriptor_declaration}

{declarations}

#ifdef __cplusplus
}}
#endif

#endif  // {guard}
"""


def format_runtime_abi_header() -> str:
    """Return the shared runtime ABI header used by generated sources."""
    return runtime_abi_header_text()


def format_runtime_source(module: RunintModule, prefix: str) -> str:
    """Format C source for runtime kernels."""
    safe_prefix = _sanitize_identifier(prefix)
    descriptor = _format_form_descriptor_source(module, safe_prefix)
    definitions = "\n\n".join(kernel.c_definition for kernel in module.kernels)
    return f"""#include <complex.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "runintgen_runtime_abi.h"
#include "{safe_prefix}.h"

{descriptor}

{definitions}
"""


def write_runtime_code(
    module: RunintModule,
    prefix: str,
    output_dir: str | Path,
) -> RuntimeCodeFiles:
    """Write generated runtime kernel header and source files.

    The runtime ABI header is reusable support code shipped in
    ``runintgen/cpp``. Generated files include it by name, so downstream builds
    should add that directory to their include path.

    Args:
        module: Result from :func:`runintgen.compile_runtime_integrals`.
        prefix: File prefix, for example ``"laplace_p2"``.
        output_dir: Directory where files should be written.

    Returns:
        Paths to the generated header/source and the canonical ABI header.
    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    safe_prefix = _sanitize_identifier(prefix)

    header = outdir / f"{safe_prefix}.h"
    source = outdir / f"{safe_prefix}.c"
    abi_header = runtime_abi_header_path()

    header.write_text(format_runtime_header(module, safe_prefix), encoding="utf-8")
    source.write_text(format_runtime_source(module, safe_prefix), encoding="utf-8")

    return RuntimeCodeFiles(
        header=header,
        source=source,
        abi_header=abi_header,
    )
