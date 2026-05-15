"""Runtime C ABI helpers for runintgen kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np
import numpy.typing as npt

from .cpp_headers import runtime_abi_cdef

if TYPE_CHECKING:
    import cffi


CFFI_DEF = runtime_abi_cdef()


def _borrow_array(
    value: Any,
    *,
    dtype: np.dtype,
    name: str,
) -> np.ndarray:
    """Return a caller-owned contiguous NumPy array without copying."""
    if not isinstance(value, np.ndarray):
        raise TypeError(
            f"{name} must be a NumPy ndarray for zero-copy runtime quadrature."
        )
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {value.dtype}.")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous.")
    return value


@dataclass
class RuntimeQuadratureRule:
    """Reference quadrature carried by ``custom_data``.

    Weights are expected to be pre-scaled by the runtime integration provider,
    including the relevant cell/facet/cut measure scaling.
    """

    points: npt.NDArray[np.float64]
    weights: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        """Normalise quadrature arrays."""
        self.points = np.ascontiguousarray(self.points, dtype=np.float64)
        self.weights = np.ascontiguousarray(self.weights, dtype=np.float64)
        if self.points.ndim != 2:
            raise ValueError("Runtime quadrature points must have shape (nq, tdim).")
        if self.points.shape[0] != self.weights.shape[0]:
            raise ValueError("Runtime quadrature points and weights disagree on nq.")

    @property
    def nq(self) -> int:
        """Number of quadrature points."""
        return int(self.weights.shape[0])

    @property
    def tdim(self) -> int:
        """Reference point dimension."""
        return int(self.points.shape[1])


@dataclass
class QuadratureRules:
    """Borrowed flat quadrature-rule arrays.

    The constructor validates and stores references to caller-owned NumPy
    arrays. It does not coerce dtype or contiguity, so compatible arrays from
    external C++ providers can pass through Python and into ``custom_data``
    without copying quadrature storage.
    """

    tdim: int
    points: npt.NDArray[np.float64]
    weights: npt.NDArray[np.float64]
    offsets: npt.NDArray[np.int64] | None = None
    parent_map: npt.NDArray[np.int32] | None = None
    rule_id: str | None = None
    kind: str = "per_entity"
    gdim: int | None = None
    physical_points: npt.NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        """Validate borrowed quadrature buffers."""
        self.tdim = int(self.tdim)
        if self.tdim <= 0:
            raise ValueError("tdim must be positive.")
        if self.kind not in {"per_entity", "shared"}:
            raise ValueError("kind must be 'per_entity' or 'shared'.")
        if self.rule_id is None:
            self.rule_id = str(uuid4())

        self.points = _borrow_array(
            self.points, dtype=np.dtype(np.float64), name="points"
        )
        self.weights = _borrow_array(
            self.weights, dtype=np.dtype(np.float64), name="weights"
        )

        if self.points.ndim == 2:
            if self.points.shape[1] != self.tdim:
                raise ValueError("points second dimension must equal tdim.")
            if (
                self.kind == "per_entity"
                and self.points.shape[0] != self.weights.size
            ):
                raise ValueError("points and weights disagree on total nq.")
            if self.kind == "shared" and self.points.shape[0] != self.weights.size:
                raise ValueError("shared points and weights disagree on nq.")
        elif self.points.ndim == 1:
            if self.points.size != self.weights.size * self.tdim:
                raise ValueError("flat points size must equal weights.size * tdim.")
        else:
            raise ValueError("points must have shape (total_nq, tdim) or be flat.")

        if self.weights.ndim != 1:
            raise ValueError("weights must have shape (total_nq,).")

        if self.kind == "per_entity":
            if self.offsets is None:
                raise ValueError("per-entity quadrature rules require offsets.")
            self.offsets = _borrow_array(
                self.offsets, dtype=np.dtype(np.int64), name="offsets"
            )
            if self.offsets.ndim != 1 or self.offsets.size == 0:
                raise ValueError("offsets must be a non-empty 1D array.")
            if self.offsets[0] != 0:
                raise ValueError("offsets[0] must be zero.")
            if np.any(np.diff(self.offsets) < 0):
                raise ValueError("offsets must be nondecreasing.")
            if self.offsets[-1] != self.weights.size:
                raise ValueError("offsets[-1] must equal weights.size.")
        elif self.offsets is not None:
            raise ValueError("shared quadrature rules do not use offsets.")

        if self.parent_map is not None:
            self.parent_map = _borrow_array(
                self.parent_map, dtype=np.dtype(np.int32), name="parent_map"
            )
            if self.parent_map.ndim != 1:
                raise ValueError("parent_map must be 1D.")
            expected = (
                self.num_rules if self.kind == "per_entity" else self.num_entities
            )
            if self.parent_map.size != expected:
                raise ValueError("parent_map has the wrong number of entries.")

        if self.gdim is not None:
            self.gdim = int(self.gdim)
            if self.gdim <= 0:
                raise ValueError("gdim must be positive when supplied.")
        if self.physical_points is not None:
            if self.gdim is None:
                raise ValueError("gdim is required when physical_points are supplied.")
            self.physical_points = _borrow_array(
                self.physical_points,
                dtype=np.dtype(np.float64),
                name="physical_points",
            )
            if self.physical_points.ndim != 2:
                raise ValueError("physical_points must have shape (gdim, total_nq).")
            if self.physical_points.shape[0] != self.gdim:
                raise ValueError("physical_points first dimension must equal gdim.")
            if self.physical_points.shape[1] != self.total_points:
                raise ValueError(
                    "physical_points second dimension must equal total_nq."
                )

    @property
    def num_rules(self) -> int:
        """Number of flat quadrature-rule slices."""
        if self.kind != "per_entity" or self.offsets is None:
            return 1
        return int(self.offsets.size - 1)

    @property
    def total_points(self) -> int:
        """Total number of quadrature points across all rule slices."""
        if self.kind == "shared":
            if self.parent_map is None:
                return int(self.weights.size)
            return int(self.weights.size * self.parent_map.size)
        return int(self.weights.size)

    @property
    def num_entities(self) -> int:
        """Number of supplied entities represented by the rule set."""
        if self.kind == "per_entity":
            return self.num_rules
        if self.parent_map is None:
            return 1
        return int(self.parent_map.size)

    @classmethod
    def from_rules(
        cls,
        rules: Any,
        *,
        parent_map: npt.ArrayLike | None = None,
    ) -> "QuadratureRules":
        """Pack one or more rule-like objects into flat quadrature arrays.

        This convenience constructor may copy quadrature values. Use the main
        constructor with flat arrays when the provider already owns suitable
        storage and zero-copy behavior matters.
        """
        rule_list = rules if isinstance(rules, (list, tuple)) else [rules]
        if len(rule_list) == 0:
            raise ValueError("At least one quadrature rule is required.")

        point_blocks = []
        weight_blocks = []
        offsets = [0]
        tdim: int | None = None
        for rule in rule_list:
            if isinstance(rule, RuntimeQuadratureRule):
                points = rule.points
                weights = rule.weights
            elif isinstance(rule, dict):
                points = rule["points"]
                weights = rule["weights"]
            else:
                points = getattr(rule, "points")
                weights = getattr(rule, "weights")

            points_array = np.ascontiguousarray(points, dtype=np.float64)
            weights_array = np.ascontiguousarray(weights, dtype=np.float64)
            if points_array.ndim != 2:
                raise ValueError(
                    "Runtime quadrature points must have shape (nq, tdim)."
                )
            if weights_array.ndim != 1:
                weights_array = np.ravel(weights_array)
            if points_array.shape[0] != weights_array.shape[0]:
                raise ValueError(
                    "Runtime quadrature points and weights disagree on nq."
                )
            if tdim is None:
                tdim = int(points_array.shape[1])
            elif points_array.shape[1] != tdim:
                raise ValueError("All quadrature rules must have the same tdim.")

            point_blocks.append(points_array)
            weight_blocks.append(weights_array)
            offsets.append(offsets[-1] + int(weights_array.size))

        points = np.ascontiguousarray(np.vstack(point_blocks), dtype=np.float64)
        weights = np.ascontiguousarray(np.concatenate(weight_blocks), dtype=np.float64)
        offset_array = np.asarray(offsets, dtype=np.int64)
        parent_array = (
            None
            if parent_map is None
            else np.ascontiguousarray(parent_map, dtype=np.int32)
        )

        return cls(
            tdim=int(tdim),
            points=points,
            weights=weights,
            offsets=offset_array,
            parent_map=parent_array,
        )


@dataclass
class RuntimeEntityMap:
    """Integration-loop entity map derived from form subdomain data."""

    entity_indices: npt.NDArray[np.int32]
    is_cut: npt.NDArray[np.uint8]
    rule_indices: npt.NDArray[np.int32]

    def __post_init__(self) -> None:
        """Validate borrowed or constructed entity-map arrays."""
        self.entity_indices = _borrow_array(
            self.entity_indices, dtype=np.dtype(np.int32), name="entity_indices"
        )
        self.is_cut = _borrow_array(
            self.is_cut, dtype=np.dtype(np.uint8), name="is_cut"
        )
        self.rule_indices = _borrow_array(
            self.rule_indices, dtype=np.dtype(np.int32), name="rule_indices"
        )
        if (
            self.entity_indices.ndim != 1
            or self.is_cut.ndim != 1
            or self.rule_indices.ndim != 1
        ):
            raise ValueError("entity map arrays must be 1D.")
        if not (
            self.entity_indices.size
            == self.is_cut.size
            == self.rule_indices.size
        ):
            raise ValueError("entity map arrays must have equal length.")

    @property
    def num_entities(self) -> int:
        """Number of integration-loop entities."""
        return int(self.entity_indices.size)

    @classmethod
    def runtime_only(cls, rules: QuadratureRules) -> "RuntimeEntityMap":
        """Build a runtime-only entity map from quadrature parent ids."""
        entity_indices = (
            rules.parent_map
            if rules.parent_map is not None
            else np.arange(rules.num_rules, dtype=np.int32)
        )
        return cls(
            entity_indices=entity_indices,
            is_cut=np.ones(entity_indices.size, dtype=np.uint8),
            rule_indices=np.arange(rules.num_rules, dtype=np.int32),
        )

    @classmethod
    def mixed(
        cls,
        *,
        standard_entities: npt.ArrayLike,
        rules: QuadratureRules,
    ) -> "RuntimeEntityMap":
        """Build an entity map from standard entities and per-entity rules."""
        standard = np.ascontiguousarray(standard_entities, dtype=np.int32)
        if standard.ndim != 1:
            standard = np.ravel(standard)
        runtime_entities = (
            rules.parent_map
            if rules.parent_map is not None
            else np.arange(rules.num_rules, dtype=np.int32)
        )
        entity_indices = np.ascontiguousarray(
            np.concatenate([standard, runtime_entities]), dtype=np.int32
        )
        is_cut = np.concatenate(
            [
                np.zeros(standard.size, dtype=np.uint8),
                np.ones(runtime_entities.size, dtype=np.uint8),
            ]
        )
        rule_indices = np.concatenate(
            [
                np.full(standard.size, -1, dtype=np.int32),
                np.arange(rules.num_rules, dtype=np.int32),
            ]
        )
        return cls(
            entity_indices=entity_indices,
            is_cut=np.ascontiguousarray(is_cut, dtype=np.uint8),
            rule_indices=np.ascontiguousarray(rule_indices, dtype=np.int32),
        )


@dataclass
class RuntimeQuadraturePayload:
    """Borrowed quadrature plus derived integration-loop entity map."""

    rules: QuadratureRules
    entities: RuntimeEntityMap
    quadrature_functions: "QuadratureFunctionValueSet | None" = None

    def __post_init__(self) -> None:
        """Validate entity rule references against per-entity rule storage."""
        cut_entries = self.entities.is_cut != 0
        if np.any(self.entities.rule_indices[cut_entries] < 0):
            raise ValueError("cut entities must reference a quadrature rule.")
        if np.any(self.entities.rule_indices[cut_entries] >= self.rules.num_rules):
            raise ValueError("cut entity rule index is out of range.")
        if np.any(self.entities.rule_indices[~cut_entries] >= 0):
            raise ValueError("standard entities must use rule index -1.")

    @property
    def tdim(self) -> int:
        """Topological dimension for quadrature points."""
        return self.rules.tdim

    @property
    def points(self) -> npt.NDArray[np.float64]:
        """Flat or 2D quadrature points."""
        return self.rules.points

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """Flat quadrature weights."""
        return self.rules.weights

    @property
    def offsets(self) -> npt.NDArray[np.int64]:
        """Rule offsets in quadrature-point units."""
        return self.rules.offsets

    @property
    def parent_map(self) -> npt.NDArray[np.int32] | None:
        """Parent entity for each runtime quadrature rule."""
        return self.rules.parent_map

    @property
    def entity_indices(self) -> npt.NDArray[np.int32]:
        """Integration-loop entity ids."""
        return self.entities.entity_indices

    @property
    def is_cut(self) -> npt.NDArray[np.uint8]:
        """Branch flag for each integration-loop entity."""
        return self.entities.is_cut

    @property
    def rule_indices(self) -> npt.NDArray[np.int32]:
        """Runtime rule index for each integration-loop entity."""
        return self.entities.rule_indices

    @property
    def num_rules(self) -> int:
        """Number of runtime quadrature rules."""
        return self.rules.num_rules

    @property
    def num_entities(self) -> int:
        """Number of integration-loop entities."""
        return self.entities.num_entities


@dataclass
class QuadratureFunctionValue:
    """Provider-owned packed values for one quadrature function slot."""

    values: npt.NDArray[np.float64]
    value_size: int

    def __post_init__(self) -> None:
        """Validate a kernel-facing quadrature-function value array."""
        self.values = _borrow_array(
            self.values, dtype=np.dtype(np.float64), name="quadrature function values"
        )
        self.value_size = int(self.value_size)
        if self.value_size <= 0:
            raise ValueError("value_size must be positive.")
        if self.values.ndim == 1:
            if self.value_size != 1:
                raise ValueError("vector/tensor quadrature values must be 2D.")
        elif self.values.ndim == 2:
            if self.values.shape[1] != self.value_size:
                raise ValueError("values second dimension must equal value_size.")
        else:
            raise ValueError(
                "values must have shape (total_nq,) or (total_nq, value_size)."
            )

    @property
    def num_points(self) -> int:
        """Number of quadrature points represented by the value array."""
        return int(self.values.shape[0])


@dataclass
class QuadratureFunctionValueSet:
    """Packed quadrature-function values ordered by generated slot."""

    functions: list[QuadratureFunctionValue]

    @property
    def num_functions(self) -> int:
        """Number of quadrature-function value slots."""
        return len(self.functions)


def _normalise_quadrature_function_values(
    values: npt.ArrayLike,
    *,
    value_size: int,
    total_points: int,
    borrowed: bool,
) -> npt.NDArray[np.float64]:
    """Return values in point-major kernel-facing layout."""
    if borrowed:
        array = _borrow_array(
            values, dtype=np.dtype(np.float64), name="quadrature function values"
        )
    else:
        array = np.ascontiguousarray(values, dtype=np.float64)

    if value_size == 1:
        if array.ndim == 2 and array.shape[1] == 1:
            array = np.ascontiguousarray(array[:, 0], dtype=np.float64)
        if array.ndim != 1:
            raise ValueError("scalar quadrature-function values must be 1D.")
        if array.shape[0] != total_points:
            raise ValueError("scalar quadrature-function values have wrong length.")
        return array

    if array.ndim != 2:
        raise ValueError("vector/tensor quadrature-function values must be 2D.")
    if array.shape != (total_points, value_size):
        raise ValueError(
            "vector/tensor quadrature-function values must have shape "
            f"({total_points}, {value_size})."
        )
    return array


def build_quadrature_function_value_set(
    infos: list[Any],
    rules: QuadratureRules,
    *,
    fallback_evaluator: Any | None = None,
) -> QuadratureFunctionValueSet | None:
    """Resolve callable or explicit values for generated quadrature functions."""
    if not infos:
        return None

    from .quadrature_function import (
        quadrature_function_source,
        quadrature_function_values,
    )

    packed: list[QuadratureFunctionValue] = []
    for info in infos:
        explicit_values = quadrature_function_values(info.terminal)
        rule_values = explicit_values.get(str(rules.rule_id))
        if rule_values is not None:
            values = _normalise_quadrature_function_values(
                rule_values,
                value_size=info.value_size,
                total_points=rules.total_points,
                borrowed=True,
            )
        else:
            source = quadrature_function_source(info.terminal)
            if source is not None:
                if rules.physical_points is None:
                    raise ValueError(
                        f"QuadratureFunction {info.label!r} uses a callable source, "
                        "but the quadrature rules do not carry physical_points."
                    )
                raw_values = source(rules.physical_points)
            elif fallback_evaluator is not None:
                raw_values = fallback_evaluator(info, rules)
            else:
                raise ValueError(
                    f"QuadratureFunction {info.label!r} has no values for "
                    f"quadrature rule {rules.rule_id!r}."
                )
            values = _normalise_quadrature_function_values(
                raw_values,
                value_size=info.value_size,
                total_points=rules.total_points,
                borrowed=False,
            )

        packed.append(
            QuadratureFunctionValue(values=values, value_size=info.value_size)
        )

    return QuadratureFunctionValueSet(functions=packed)


def as_runtime_quadrature_rules(
    quadrature: Any,
    *,
    parent_map: npt.ArrayLike | None = None,
) -> QuadratureRules:
    """Return flat runtime quadrature rules from supported inputs."""
    if isinstance(quadrature, QuadratureRules):
        if parent_map is not None:
            raise ValueError("parent_map is part of QuadratureRules.")
        return quadrature
    return QuadratureRules.from_rules(quadrature, parent_map=parent_map)


def _is_form_subdomain_data(value: Any) -> bool:
    """Return whether value looks like UFL mixed subdomain data."""
    if not isinstance(value, (list, tuple)):
        return False
    values = [
        item[1] if isinstance(item, tuple) and len(item) == 2 else item
        for item in value
    ]
    return any(_is_rule_like(item) for item in values) and any(
        not _is_rule_like(item) for item in values
    )


def _is_rule_like(value: Any) -> bool:
    """Return whether value looks like runtime quadrature data."""
    return hasattr(value, "points") and hasattr(value, "weights")


def _entity_array(value: Any) -> npt.NDArray[np.int32]:
    """Return entity ids as a contiguous int32 array."""
    entities = np.ascontiguousarray(value, dtype=np.int32)
    if entities.ndim != 1:
        entities = np.ravel(entities)
    return entities


def as_runtime_quadrature_payload(
    quadrature: Any,
    *,
    is_cut: npt.ArrayLike | None = None,
) -> RuntimeQuadraturePayload:
    """Return quadrature plus entity-map payload for ``custom_data``."""
    if isinstance(quadrature, RuntimeQuadraturePayload):
        if is_cut is not None:
            raise ValueError("is_cut is already encoded in RuntimeQuadraturePayload.")
        return quadrature

    if _is_form_subdomain_data(quadrature):
        runtime_items = []
        standard_blocks = []
        for item in quadrature:
            value = item[1] if isinstance(item, tuple) and len(item) == 2 else item
            if _is_rule_like(value):
                runtime_items.append(value)
            else:
                standard_blocks.append(_entity_array(value))

        if len(runtime_items) != 1:
            raise ValueError("Mixed runtime subdomain data must contain one rule set.")
        rules = as_runtime_quadrature_rules(runtime_items[0])
        if rules.kind != "per_entity":
            raise ValueError(
                "Runtime quadrature payloads require "
                "QuadratureRules(kind='per_entity')."
            )
        standard_entities = (
            np.concatenate(standard_blocks)
            if standard_blocks
            else np.array([], dtype=np.int32)
        )
        return RuntimeQuadraturePayload(
            rules=rules,
            entities=RuntimeEntityMap.mixed(
                standard_entities=standard_entities,
                rules=rules,
            ),
        )

    rules = as_runtime_quadrature_rules(quadrature)
    if rules.kind != "per_entity":
        raise ValueError(
            "Runtime quadrature payloads require "
            "QuadratureRules(kind='per_entity')."
        )
    if is_cut is None:
        entities = RuntimeEntityMap.runtime_only(rules)
    else:
        flags = np.ascontiguousarray(is_cut, dtype=np.uint8)
        if flags.ndim != 1:
            flags = np.ravel(flags)
        if flags.size != rules.num_rules:
            raise ValueError("is_cut must have one entry per quadrature rule.")
        entities = RuntimeEntityMap(
            entity_indices=(
                rules.parent_map
                if rules.parent_map is not None
                else np.arange(rules.num_rules, dtype=np.int32)
            ),
            is_cut=flags,
            rule_indices=np.where(
                flags != 0,
                np.arange(rules.num_rules, dtype=np.int32),
                np.full(rules.num_rules, -1, dtype=np.int32),
            ).astype(np.int32),
        )
    return RuntimeQuadraturePayload(rules=rules, entities=entities)


@dataclass
class RuntimeBasixElement:
    """Opaque Basix element handle carried by ``custom_data``."""

    handle: int = 0
    tabulate: int = 0


@dataclass
class RuntimeTableRequest:
    """Generated request for one runtime Basix element tabulation."""

    slot: int
    derivative_order: int = 0
    is_permuted: bool = False


@dataclass
class RuntimeTableView:
    """One raw Basix tabulation view.

    Values are flattened in Basix tabulate order:
    ``[derivative][point][dof][component]``.
    """

    values: npt.NDArray[np.float64]
    num_derivatives: int
    num_points: int
    num_dofs: int
    num_components: int

    def __post_init__(self) -> None:
        """Normalise values to contiguous float64 storage."""
        self.values = np.ascontiguousarray(self.values, dtype=np.float64)
        expected = (
            self.num_derivatives
            * self.num_points
            * self.num_dofs
            * self.num_components
        )
        if self.values.size != expected:
            raise ValueError(
                f"Runtime table has {self.values.size} values, expected {expected}."
            )


class RuntimeContextBuilder:
    """Pack a runtime context into CFFI structures.

    This helper is intended for tests and lightweight JIT use. Production
    DOLFINx/CutFEMx integration should populate the same C ABI from C++ and
    implement each element's `tabulate` function pointer using real Basix
    elements and wrapper-owned scratch storage.
    """

    def __init__(self, ffi: "cffi.FFI") -> None:
        """Initialise the builder."""
        self.ffi = ffi
        self._refs: list[object] = []

    def build_context(
        self,
        quadrature: RuntimeQuadratureRule
        | QuadratureRules
        | RuntimeQuadraturePayload
        | list[Any],
        basix_elements: list[RuntimeBasixElement] | None = None,
        form_context: "cffi.CData | None" = None,
        is_cut: npt.ArrayLike | None = None,
        scratch: "cffi.CData | None" = None,
        quadrature_functions: QuadratureFunctionValueSet | None = None,
    ) -> "cffi.CData":
        """Build a ``runintgen_context*``."""
        ffi = self.ffi
        payload = as_runtime_quadrature_payload(quadrature, is_cut=is_cut)
        if quadrature_functions is not None:
            payload = RuntimeQuadraturePayload(
                rules=payload.rules,
                entities=payload.entities,
                quadrature_functions=quadrature_functions,
            )
        rules = payload.rules
        entities = payload.entities
        self._refs.append(payload)

        c_quadrature = ffi.new("runintgen_quadrature_rules*")
        self._refs.append(c_quadrature)
        c_quadrature.tdim = rules.tdim
        c_quadrature.num_rules = rules.num_rules
        c_quadrature.offsets = ffi.cast("const int64_t*", rules.offsets.ctypes.data)
        c_quadrature.points = ffi.cast("const double*", rules.points.ctypes.data)
        c_quadrature.weights = ffi.cast("const double*", rules.weights.ctypes.data)
        c_quadrature.parent_map = (
            ffi.cast("const int32_t*", rules.parent_map.ctypes.data)
            if rules.parent_map is not None
            else ffi.NULL
        )

        c_entities = ffi.new("runintgen_entity_map*")
        self._refs.append(c_entities)
        c_entities.num_entities = entities.num_entities
        c_entities.entity_indices = ffi.cast(
            "const int32_t*", entities.entity_indices.ctypes.data
        )
        c_entities.is_cut = ffi.cast("const uint8_t*", entities.is_cut.ctypes.data)
        c_entities.rule_indices = ffi.cast(
            "const int32_t*", entities.rule_indices.ctypes.data
        )

        c_q_functions = ffi.NULL
        if payload.quadrature_functions is not None:
            q_values = payload.quadrature_functions.functions
            c_q_array = ffi.new(f"runintgen_quadrature_function[{len(q_values)}]")
            self._refs.append(c_q_array)
            for i, values in enumerate(q_values):
                c_q_array[i].values = ffi.cast(
                    "const double*", values.values.ctypes.data
                )
                c_q_array[i].value_size = values.value_size
                c_q_array[i].num_points = values.num_points

            c_q_functions = ffi.new("runintgen_quadrature_functions*")
            self._refs.append(c_q_functions)
            c_q_functions.num_functions = len(q_values)
            c_q_functions.functions = c_q_array

        if form_context is None:
            form_context = self.build_form_context(
                basix_elements=basix_elements,
                scratch=scratch,
            )

        ctx = ffi.new("runintgen_context*")
        self._refs.append(ctx)
        ctx.quadrature = c_quadrature
        ctx.entities = c_entities
        ctx.quadrature_functions = c_q_functions
        ctx.form = form_context
        return ctx

    def build_form_context(
        self,
        basix_elements: list[RuntimeBasixElement] | None = None,
        descriptor: "cffi.CData | None" = None,
        scratch: "cffi.CData | None" = None,
    ) -> "cffi.CData":
        """Build and retain a ``runintgen_form_context*``."""
        ffi = self.ffi
        elements = basix_elements or []
        c_elements = ffi.new(f"runintgen_basix_element[{len(elements)}]")
        self._refs.append(c_elements)
        for i, element in enumerate(elements):
            c_elements[i].handle = ffi.cast("const void*", element.handle)
            c_elements[i].tabulate = (
                ffi.cast("runintgen_element_tabulate_fn", element.tabulate)
                if element.tabulate
                else ffi.NULL
            )

        form = ffi.new("runintgen_form_context*")
        self._refs.append(form)
        form.num_elements = len(elements)
        form.elements = c_elements
        form.descriptor = descriptor or ffi.NULL
        form.scratch = scratch or ffi.NULL
        return form

    def build_request(self, request: RuntimeTableRequest) -> "cffi.CData":
        """Build and retain a ``runintgen_table_request*``."""
        ffi = self.ffi
        c_request = ffi.new("runintgen_table_request*")
        self._refs.append(c_request)
        c_request.slot = request.slot
        c_request.derivative_order = request.derivative_order
        c_request.is_permuted = int(request.is_permuted)
        return c_request


def to_intptr(ffi: "cffi.FFI", data: "cffi.CData") -> int:
    """Return an ``intptr_t`` representation of a CFFI pointer."""
    return int(ffi.cast("intptr_t", data))
