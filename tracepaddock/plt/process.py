"""Turn a dataset and a PLT script into display signals.

Every signal is treated as a *step function of time*: a sample holds until
the next one. This defines how signals with different sample rates are combined.

* ``exp=`` with ``x`` and ``t`` only is evaluated sample by sample.
* ``exp(x1=..)=`` is evaluated on the union of the time stamps of all signals
  involved, each held at its last value.
* Sample based operations (``diff``, ``integ``, ``mavr``, ``shift``, ``dgf``,
  ``revert``, ``-mean`` ...) work on the signal's own samples by default.
  Set ``ops_on_common_time_base`` to use the recording's common time axis.

A formula can reference a recording signal without a dedicated view entry.
"""
from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.model import KIND_BIT, Dataset, DatasetError, SignalInfo
from ..display import DisplaySignal, c_exponent
from ..i18n import tr
from .expr import ExprError, compile_expr
from .model import BIT_DELETING_OPS, Op, PltEntry, PltScript
from .palette import DEFAULT_COLOR

__all__ = ["build_display_signals", "resample_hold", "signal_statistics", "statistics_text", "apply_op",
           "apply_value_map", "raster_option", "MissingSignal", "SAMPLE_OPS", "is_equidistant"]

#: Operations that work sample by sample on a signal (not element-wise formulas).
SAMPLE_OPS = frozenset({"dgf", "min", "max", "-mean", "-min", "-max", "diff", "integ", "mavr", "shift", "revert"})


class MissingSignal(ValueError):
    """A signal needed for an entry is not available in the recording."""


def resample_hold(t_src: np.ndarray, y_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """Values of a step signal at the times ``t_dst`` (before the first sample: its first value)."""
    if len(t_src) == 0:
        return np.zeros(len(t_dst), dtype=np.float64)
    idx = np.searchsorted(t_src, t_dst, side="right") - 1
    return np.asarray(y_src, dtype=np.float64)[np.clip(idx, 0, len(t_src) - 1)]


def is_equidistant(t: np.ndarray, tolerance: float = 0.25) -> bool:
    """True if all time steps are within ``tolerance`` (relative) of the median step.

    Recorder time stamps jitter by a few percent around their raster; that is
    still equidistant. Gaps, mixed rasters and steps of zero are not.
    """
    if len(t) < 3:
        return True
    dt = np.diff(np.asarray(t, dtype=np.float64))
    step = float(np.median(dt))
    return step > 0 and float(np.max(np.abs(dt - step))) <= tolerance * step


#: Names of the operations in the "non constant sample time" message.
_OP_NAME = {"diff": "DIFF", "integ": "INTEG", "mavr": "MAVR(<n>)", "dgf": "DGF=<type>:<fc>:<n>:<r>"}


def _require_equidistant(t: np.ndarray, op: Op) -> None:
    if not is_equidistant(t):
        name = _OP_NAME.get(op.kind, op.kind.upper())
        raise ValueError(f"{op.text} needs an equidistant sample time, but the time steps of this signal vary "
                         f"(OPERATION {name} ILLEGAL ON DATASET WITH NON CONSTANT TC; it can be allowed under "
                         f"Configuration > Settings > Processing)")


def _moving_average(y: np.ndarray, n: int) -> np.ndarray:
    """``mavr(n)``: n>0 mean of x[k..k+n-1], n<0 mean of x[k-(|n|-1)..k], shortened at the edges."""
    size = abs(n)
    if size <= 1 or len(y) == 0:
        return y.astype(np.float64)
    csum = np.concatenate(([0.0], np.cumsum(y, dtype=np.float64)))
    k = np.arange(len(y))
    if n > 0:
        lo, hi = k, np.minimum(len(y), k + size)
    else:
        lo, hi = np.maximum(0, k - size + 1), k + 1
    return (csum[hi] - csum[lo]) / (hi - lo)


def _digital_filter(t: np.ndarray, y: np.ndarray, op: Op) -> np.ndarray:
    from scipy import signal as sps

    args = op.args
    kind = str(args[0]).upper() if args else ""
    nums = [float(a) for a in args[1:]]
    if kind not in ("PDT1", "PDT2", "BTH", "CHB1", "CHB2"):
        raise ValueError(f"unknown filter type in '{op.text}' (BTH, CHB1, CHB2, PDT1 or PDT2)")
    if len(y) < 2:
        return y
    _require_equidistant(t, op)
    if kind == "PDT1":
        if len(nums) != 3:
            raise ValueError(f"'{op.text}': dgf=pdt1 needs three coefficients a0:a1:b1")
        b, a = [nums[0], nums[1]], [1.0, nums[2]]
    elif kind == "PDT2":
        if len(nums) != 5:
            raise ValueError(f"'{op.text}': dgf=pdt2 needs five coefficients a0:a1:a2:b1:b2")
        b, a = nums[:3], [1.0, nums[3], nums[4]]
    else:
        if len(nums) < 2:
            raise ValueError(f"'{op.text}': dgf={kind.lower()} needs <cut-off frequency>:<order>")
        step = float(np.median(np.diff(t)))
        wn = nums[0] * 2.0 * step                      # cut-off relative to the Nyquist frequency
        order = int(nums[1])
        if not 0 < wn < 1 or not 1 <= order <= 20:
            raise ValueError(f"'{op.text}': the cut-off frequency must be below half the sample rate "
                             f"({0.5 / step:g} Hz) and the order between 1 and 20")
        if kind == "BTH":
            b, a = sps.butter(order, wn)
        else:
            if len(nums) < 3:
                raise ValueError(f"'{op.text}': dgf={kind.lower()} needs a ripple value <fc>:<order>:<ripple dB>")
            b, a = (sps.cheby1 if kind == "CHB1" else sps.cheby2)(order, nums[2], wn)
    try:
        # Start settled on the first value, so the output does not swing in from zero.
        zi = sps.lfilter_zi(b, a) * y[0]
    except (np.linalg.LinAlgError, ValueError):
        zi = np.zeros(max(len(a), len(b)) - 1)          # no steady state (e.g. an integrator): start at 0
    out, _ = sps.lfilter(b, a, y, zi=zi)
    return out


_RASTER = re.compile(r"@raster\s*=\s*(\S+)", re.IGNORECASE)


def raster_option(entry: object) -> Optional[str]:
    """Read the calculation raster from a ``// @raster=...`` comment.

    Supported values are ``union``, ``fixed:<seconds>`` and ``signal:<x | x1..x9 | name>``.
    """
    comment = getattr(entry, "comment", None) or ""
    m = _RASTER.search(comment)
    return m.group(1) if m else None


def _raster_times(raster: Optional[str], t: np.ndarray, others: Dict[str, Tuple[np.ndarray, np.ndarray]],
                  lookup: Callable[[str], Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Time stamps a formula is evaluated at: the union of its inputs (default), a fixed step, or the
    time stamps of one input."""
    union = t
    for rt, _ in others.values():
        union = np.union1d(union, rt)
    if not raster or raster.lower() == "union":
        return union
    kind, _sep, arg = raster.partition(":")
    kind = kind.lower()
    if kind == "fixed":
        try:
            step = float(arg)
        except ValueError:
            raise ValueError(f"@raster=fixed needs a step in seconds, e.g. @raster=fixed:0.01 (got '{arg}')")
        if not step > 0 or not len(union):
            raise ValueError("@raster=fixed needs a step greater than 0")
        count = int(np.floor((union[-1] - union[0]) / step + 1e-9)) + 1
        if count > 20_000_000:
            raise ValueError("@raster=fixed: the step is too small for this recording")
        return union[0] + step * np.arange(count)
    if kind == "signal":
        if arg.lower() == "x" or not arg:
            return t
        if arg in others:
            return others[arg][0]
        return lookup(arg)[0]
    raise ValueError(f"unknown @raster '{raster}' (union, fixed:<s> or signal:<name>)")


def apply_op(op: Op, t: np.ndarray, y: np.ndarray,
             lookup: Callable[[str], Tuple[np.ndarray, np.ndarray]], *,
             calculus_on_nonequidistant: bool = True, raster: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Apply one value changing attribute. Raises ``ValueError`` with a message for the user.

    ``lookup(name)`` returns ``(t, y)`` of a referenced signal or raises
    ``ValueError`` (:class:`MissingSignal` when it does not exist). ``raster``
    sets the time stamps of a formula (see :func:`raster_option`).
    """
    kind = op.kind
    if kind == "exp":
        try:
            expr = compile_expr(op.expr)
        except ExprError as exc:
            raise ValueError(f"expression error: {exc}") from exc
        env: Dict[str, object] = {}
        tt, yy = t, y
        others = {sym: lookup(name) for sym, name in op.refs.items()} if op.refs else {}
        if others or raster:
            tt = _raster_times(raster, t, others, lookup)
            yy = resample_hold(t, y, tt)
            for sym, (rt, ry) in others.items():
                env[sym] = resample_hold(rt, ry, tt)
        env["x"], env["t"] = yy, tt
        try:
            out = expr.evaluate(env)
        except ExprError as exc:
            raise ValueError(f"expression error: {exc}") from exc
        out = np.broadcast_to(np.asarray(out, dtype=np.float64), np.shape(tt)).copy()
        return tt, out
    y = np.asarray(y, dtype=np.float64)
    if kind == "dgf":
        return t, _digital_filter(t, y, op)
    if kind in ("diff", "integ", "mavr") and not calculus_on_nonequidistant:
        _require_equidistant(t, op)
    if kind in ("min", "max"):
        value = (np.nanmin(y) if kind == "min" else np.nanmax(y)) if len(y) else 0.0
        return t, np.full(len(y), value)
    if kind == "-mean":
        return t, y - (np.nanmean(y) if len(y) else 0.0)
    if kind == "-min":
        return t, y - (np.nanmin(y) if len(y) else 0.0)
    if kind == "-max":
        return t, y - (np.nanmax(y) if len(y) else 0.0)
    if kind == "diff":
        out = np.zeros(len(y))
        if len(y) > 1:
            dt = np.diff(t)
            np.divide(np.diff(y), dt, out=out[1:], where=dt > 0)
        return t, out
    if kind == "integ":
        # Left rectangles: the value held from t[k-1] to t[k] is x[k-1], so this is the exact
        # integral of the step signal.
        out = np.zeros(len(y))
        if len(y) > 1:
            out[1:] = np.cumsum(y[:-1] * np.diff(t))
        return t, out
    if kind == "mavr":
        return t, _moving_average(y, int(op.args[0]) if op.args else 1)
    if kind == "shift":
        n = int(op.args[0]) if op.args else 0
        if n == 0 or len(y) == 0:
            return t, y
        out = np.empty_like(y)
        if abs(n) >= len(y):
            out[:] = y[0] if n > 0 else y[-1]
        elif n > 0:
            out[:n], out[n:] = y[0], y[:-n]
        else:
            out[n:], out[:n] = y[-1], y[-n:]
        return t, out
    if kind == "revert":
        return t, y[::-1].copy()
    raise ValueError(f"unknown attribute '{op.text}'")


def _auto_range(y: Optional[np.ndarray], widen: float = 1.0) -> Tuple[float, float]:
    if y is None or len(y) == 0 or not np.isfinite(y).any():
        return 0.0, 1.0
    lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
    if lo == hi:
        pad = abs(lo) * 0.5 or 0.5
        return lo - pad, hi + pad
    if widen != 1.0:
        mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * widen
        return mid - half, mid + half
    return lo, hi


def value_map_names(entry: PltEntry, info: Optional[SignalInfo]) -> List[str]:
    """Names under which a mapping rule for a view line is looked up, the first name winning: the display
    name (``key=``); for a line that shows recorded values also the token and the recorded names."""
    names = [entry.key]
    if not entry.ops:
        names.append(entry.token)                 # name, or name:pos[:width] for a bit field
        if entry.bit_pos is not None and not entry.bit_signed and (entry.bit_count or 1) == 1:
            names.append(f"{entry.name}:{entry.bit_pos}:1")
        if entry.bit_pos is None:
            names.append(entry.name)
            if info is not None:
                names += [info.name, info.source_name, info.source_name.split("\\")[0]]
    # "*" (a placeholder) or a key with * / ? would act as a wildcard for every signal
    return [name for name in names if name and "*" not in name and "?" not in name]


def apply_value_map(sig: DisplaySignal, valuemaps: Optional[object]) -> None:
    """Give ``sig`` the user's mapping table for its values, if one is assigned to it.

    A table assigned to the recorded name applies only while the line shows
    the recorded values; once attributes changed them, only a table assigned
    to the display name (``key=``) of that line applies.
    """
    sig.value_mode, sig.map_name, sig.map_unit, sig.lane, sig.value_map = None, "", "", False, None
    if valuemaps is None:
        return
    try:
        hit = valuemaps.find(*value_map_names(sig.entry, sig.info))
    except Exception:
        hit = None
    if hit is None:
        return
    table, assignment = hit
    if table is not None:
        if table.rows or table.default:
            sig.enum = table.enum()
        sig.map_name, sig.map_unit, sig.value_map = table.name, table.unit, table
    sig.value_mode = assignment.mode
    sig.lane = bool(assignment.lane)


def build_display_signals(dataset: Dataset, script: PltScript, settings: Optional[object] = None, *,
                          progress: Optional[Callable[[int, int], None]] = None,
                          valuemaps: Optional[object] = None) -> List[DisplaySignal]:
    """One :class:`DisplaySignal` per enabled entry of ``script``, in script order.

    An entry that cannot be built still gets a display signal, without data
    and with ``error`` saying why; one failing entry never stops the others.
    With the setting ``load_disabled_signals`` disabled entries are processed
    too (so later entries can reference them) but are not returned.
    ``valuemaps`` is the user's :class:`~tracepaddock.valuemap.ValueMapLibrary`.
    """
    def opt(key: str, default: object) -> object:
        try:
            return settings.get(key, default) if settings is not None else default
        except Exception:
            return default

    load_disabled = bool(opt("load_disabled_signals", False))
    skip_incomplete = bool(opt("ignore_incomplete_computed", False))
    calculus_free = bool(opt("calculus_on_nonequidistant", True))
    on_common_base = bool(opt("ops_on_common_time_base", False))
    entries = list(script.entries())
    total = len(entries)
    common: Optional[np.ndarray] = None

    def common_base() -> np.ndarray:
        nonlocal common
        if common is None:
            try:
                common = np.asarray(dataset.common_time_base(), dtype=np.float64)
            except Exception:
                common = np.empty(0)
        return common

    # Where each name is defined, to tell "defined further down" from "not in the script".
    defined_at: Dict[str, int] = {}
    for index, entry in enumerate(entries):
        for name in {entry.display_name, entry.token}:
            defined_at.setdefault(name.lower(), index)
    processed: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    result: List[DisplaySignal] = []

    for number, entry in enumerate(entries):
        if progress is not None:
            progress(number, total)
        if not entry.enabled and not load_disabled:
            continue

        def lookup(name: str, _number: int = number) -> Tuple[np.ndarray, np.ndarray]:
            hit = processed.get(name) or processed.get(name.lower())
            if hit is not None:
                return hit
            # A signal of the recording can be used before (or without) its own
            # line, as in the help's example where 'moving' is defined below.
            info = dataset.find(name)
            if info is not None:
                data = dataset.read(info)
                return data.t, data.y
            where = defined_at.get(name.lower())
            if where is not None and where >= _number:
                raise MissingSignal(f"references '{name}', which is not defined yet: it is defined further "
                                    "down in the script; move this line below that definition")
            raise MissingSignal(f"references '{name}', which is neither defined above nor contained "
                                "in the dataset")

        sig = DisplaySignal(entry=entry, name=entry.display_name, unit=entry.unit or "",
                            color=entry.color if entry.color is not None else DEFAULT_COLOR,
                            fmt=entry.fmt, emphasize=entry.emphasize, group=entry.group)
        info: Optional[SignalInfo] = None
        t: Optional[np.ndarray] = None
        y: Optional[np.ndarray] = None
        enum = None
        try:
            if entry.is_placeholder:
                # A placeholder has no samples of its own: it lives on the common
                # time axis of the recording.
                t = common_base()
                if len(t) == 0:
                    lo, hi = dataset.time_range()
                    t = np.arange(lo, hi + 0.005, 0.01)
                y = np.zeros(len(t))
            else:
                info = dataset.find(entry.name, entry.channel_origin)
                if info is None:
                    raise MissingSignal(tr("Saved channel source is unavailable; choose the source in the view editor")
                                        if entry.channel_origin else "not contained in the dataset")
                data = dataset.read(info)
                t, y, enum = data.t, data.y, data.enum
                if entry.unit is None:
                    sig.unit = info.unit
            if entry.bit_pos is not None:
                width = max(1, entry.bit_count or 1)
                bits = np.nan_to_num(np.asarray(y, dtype=np.float64)).astype(np.int64)
                field = (bits >> entry.bit_pos) & ((1 << width) - 1)
                if entry.bit_signed:
                    field = np.where(field >= (1 << (width - 1)), field - (1 << width), field)
                y, enum = field.astype(np.float64), None
            moved = False
            raster = raster_option(entry)
            for op in entry.ops:
                if on_common_base and not moved and op.kind in SAMPLE_OPS and len(common_base()):
                    y, t, moved = resample_hold(t, y, common_base()), common_base(), True
                t, y = apply_op(op, t, y, lookup, calculus_on_nonequidistant=calculus_free, raster=raster)
                enum = None
        except (ValueError, DatasetError) as exc:
            computed = any(op.refs for op in entry.ops)
            if skip_incomplete and computed and isinstance(exc, MissingSignal):
                continue
            sig.error, t, y = str(exc), None, None
        except Exception as exc:                       # never let one line stop the whole script
            sig.error, t, y = f"processing error: {type(exc).__name__}: {exc}", None, None
        sig.info, sig.t, sig.y, sig.enum = info, t, y, enum
        sig.modified = bool(entry.ops)
        if t is not None and len(t):
            name = entry.key or entry.token
            processed[name] = processed[name.lower()] = (t, y)
        # placement: the DIGITAL window for one-bit signals, unless a value
        # changing attribute that comes after the (implicit or written) BIT deletes it
        deleting = [i for i, op in enumerate(entry.ops) if op.kind in BIT_DELETING_OPS]
        one_bit = (info is not None and info.kind == KIND_BIT) or (
            entry.bit_pos is not None and (entry.bit_count or 1) == 1)
        if entry.view == "bit":
            sig.digital = not any(i >= (entry.view_pos or 0) for i in deleting)
        elif entry.view == "nobit":
            sig.digital = False
        else:
            sig.digital = one_bit and not deleting
        apply_value_map(sig, valuemaps)
        if sig.digital or not sig.has_data:
            sig.lane = False                          # one-bit signals keep their DIGITAL lane
        # view range
        lo = entry.scmin if entry.scmin is not None else (info.scmin if info is not None else None)
        hi = entry.scmax if entry.scmax is not None else (info.scmax if info is not None else None)
        if entry.scansc:
            lo, hi = _auto_range(y, 1.3)
        elif lo is None or hi is None or not lo < hi:
            lo, hi = (0.0, 1.0) if sig.digital else _auto_range(y)
        sig.scmin, sig.scmax = float(lo), float(hi)
        for index in sorted(entry.refs):
            value = entry.refs[index]
            if isinstance(value, str):
                func = {"MIN": np.nanmin, "MAX": np.nanmax, "MEAN": np.nanmean}.get(value.upper())
                if func is None or y is None or len(y) == 0:
                    continue                          # only a number, MIN, MAX or MEAN is a reference line
                value = func(y)
            sig.ref_lines.append(float(value))
        if entry.enabled:
            result.append(sig)
    if progress is not None:
        progress(total, total)
    return result


def signal_statistics(sig: DisplaySignal, t0: Optional[float] = None, t1: Optional[float] = None,
                      base: Optional[np.ndarray] = None) -> Optional[Dict[str, float]]:
    """Minimum, maximum, mean and standard deviation of a signal in ``[t0, t1]``.

    The signal is a step function, so the value held at ``t0`` counts even if
    it was sampled earlier. With ``base`` (the dataset's common time base) the
    signal is evaluated at every base point in the window. Standard deviation
    uses N-1 in that mode. Without ``base``, each sample is weighted by the
    duration it is held. NaN counts as zero. Returns ``None`` for an empty window.
    """
    if not sig.has_data:
        return None
    t = np.asarray(sig.t, dtype=np.float64)
    y = np.nan_to_num(np.asarray(sig.y, dtype=np.float64))
    lo = t[0] if t0 is None else float(t0)
    hi = t[-1] if t1 is None else float(t1)
    if base is not None and len(base):
        pts = base[(base >= lo) & (base <= hi)]
        if len(pts) == 0:
            return None
        idx = np.clip(np.searchsorted(t, pts, side="right") - 1, 0, len(t) - 1)
        w = np.bincount(idx, minlength=len(t)).astype(np.float64)
        n = len(pts)
        used = w > 0
    else:
        first = max(0, int(np.searchsorted(t, lo, side="right")) - 1)
        last = int(np.searchsorted(t, hi, side="right"))
        if last <= first:
            return None
        edges = np.clip(np.concatenate((t[first:last], [max(hi, t[last - 1])])), lo, hi)
        w = np.zeros(len(t))
        w[first:last] = np.diff(edges)
        if w.sum() <= 0:
            w[first:last] = 1.0
        n = last - first
        used = np.zeros(len(t), dtype=bool)
        used[first:last] = True                       # every sample in the window counts for min / max
    total = w.sum()
    mean = float(np.sum(w * y) / total)
    var = float(np.sum(w * (y - mean) ** 2) / total)
    std = float(np.sqrt(var * n / (n - 1))) if n > 1 else 0.0
    return {"n": float(n), "min": float(y[used].min()), "max": float(y[used].max()),
            "mean": mean, "std": std, "t0": lo, "t1": hi}


def statistics_text(dataset_name: str, signals: Sequence[DisplaySignal], full: Tuple[float, float],
                    view: Tuple[float, float], base: Optional[np.ndarray] = None) -> str:
    """Tab-separated statistics for the clipboard.

    Columns: dataset, signal, unit, then for the whole recording and for the
    current view each: start, end, time unit, min, max, mean, standard deviation.
    """
    def num(value: float) -> str:
        return c_exponent("%12g" % value)

    def block(stats: Optional[Dict[str, float]], rng: Tuple[float, float]) -> List[str]:
        if stats is None:
            return [num(rng[0]), num(rng[1]), "s"] + [num(0.0)] * 4
        return [num(rng[0]), num(rng[1]), "s"] + [num(stats[k]) for k in ("min", "max", "mean", "std")]

    lines = []
    for sig in signals:
        row = [dataset_name, sig.name, sig.unit]
        row += block(signal_statistics(sig, full[0], full[1], base), full)
        row += block(signal_statistics(sig, view[0], view[1], base), view)
        lines.append("\t".join(row))
    return "\r\n".join(lines) + ("\r\n" if lines else "")
