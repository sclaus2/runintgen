# Third Party Notices

This document records third-party dependencies and provenance notes for
`runintgen`.

`runintgen` itself is licensed under the MIT license. See `LICENSE`.

## FEniCSx Components

`runintgen` is built on the FEniCSx code-generation stack. It imports and uses
FFCx, UFL, and Basix APIs, including FFCx IR and code-generation extension
points.

- Basix: MIT
- UFL: LGPL-3.0-or-later
- FFCx: LGPL-3.0-or-later
- UFCx header from FFCx: Unlicense, according to the FFCx license file

The runtime code generator is designed as runintgen-owned code that calls,
subclasses, and adapts behavior through FFCx APIs. The project does not vendor
FFCx, UFL, or Basix source files. Files that are intentionally coupled to FFCx
internals carry local provenance comments.

If future changes copy or modify FFCx, UFL, or DOLFINx source text into this
repository, the copied or modified files must preserve the original copyright
and SPDX headers, and the package must include the corresponding license texts.

## Python And Build Dependencies

The Python package depends on:

- NumPy: BSD-3-Clause and other permissive notices in NumPy distribution
  metadata
- cffi, for the optional JIT entry point: MIT
- scikit-build-core, for builds: Apache-2.0
- nanobind, for the Basix runtime extension: BSD-style license

These dependencies are not vendored in `runintgen`; they are resolved as normal
Python or CMake dependencies.
