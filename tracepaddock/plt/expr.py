"""Formula engine for the ``exp=`` attribute of PLT scripts.

An expression is parsed by a small parser into a tree of Python closures over
numpy arrays; user text is never handed to ``eval``. Names are
case-insensitive. Variables: ``x`` (the signal itself), ``t`` (its time
axis) and ``x1`` .. ``x9`` (other signals). Constants: ``E``, ``PI``.

Operators: ``+ - * /`` (left associative), ``**`` (power, right associative,
binds tighter than a sign: ``-2**2 = -4``), unary ``+``/``-`` and the type
cast ``(DWORD~FLOAT)``, which applies to the operand right after it like a C
cast. Functions: see :data:`FUNCTIONS`. Every argument may be an expression,
so bit positions or break points can also change from sample to sample.

The bit functions follow the Fortran intrinsics of the same names and work on
the 32 bit pattern of the truncated value; their results are signed 32 bit
numbers (``NOT(0) = -1``) except for ``IBITS`` and ``BTEST``. A bit position
outside 0..31 sets or clears nothing and tests / extracts 0.
"""
from __future__ import annotations

import math
import re
from typing import Callable, Dict, List, Mapping, Set, Tuple

import numpy as np

__all__ = ["ExprError", "CompiledExpr", "compile_expr", "FUNCTIONS"]


class ExprError(ValueError):
    """The expression is not valid; the message says what is wrong."""


Node = Callable[[Mapping[str, object]], object]

_TOKEN = re.compile(r"""\s*(?:
    (?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)
  | (?P<cast>\(\s*DWORD\s*~\s*FLOAT\s*\))
  | (?P<name>[A-Za-z_][A-Za-z_0-9]*)
  | (?P<op>\*\*|[-+*/(),])
)""", re.X | re.I)

_MAX_LENGTH = 20000
_MAX_NESTING = 100
_MASK32 = np.uint64(0xFFFFFFFF)


# ---------------------------------------------------------------------- helpers
def _arr(v: object) -> np.ndarray:
    return np.asarray(v, dtype=np.float64)


def _out(a: np.ndarray, *inputs: object) -> object:
    """Return a Python float when every input was a scalar, else the array."""
    if all(np.ndim(v) == 0 for v in inputs):
        return float(np.asarray(a).reshape(-1)[0]) if np.size(a) else float("nan")
    return a


def _int(v: object) -> np.ndarray:
    """Truncated integer value(s) as int64; NaN and infinite values count as 0."""
    a = np.nan_to_num(_arr(v), nan=0.0, posinf=0.0, neginf=0.0)
    return np.trunc(np.clip(a, -2.0 ** 62, 2.0 ** 62)).astype(np.int64)


def _bits(v: object) -> np.ndarray:
    """32 bit pattern of the truncated value(s) (taken modulo 2**32) as uint64."""
    return (_int(v) & 0xFFFFFFFF).astype(np.uint64)


def _signed(u: np.ndarray) -> np.ndarray:
    """uint64 holding a 32 bit pattern -> float value of the signed 32 bit number."""
    s = u.astype(np.int64)
    return np.where(s >= 2 ** 31, s - 2 ** 32, s).astype(np.float64)


def _positions(n: object) -> Tuple[np.ndarray, np.ndarray]:
    """Bit position(s) as uint64 (invalid ones replaced by 0) and the validity mask."""
    p = _int(n)
    valid = (p >= 0) & (p <= 31)
    return np.where(valid, p, 0).astype(np.uint64), valid


def _not(x):
    return _out(_signed(~_bits(x) & _MASK32), x)


def _ibset(x, n):
    u = _bits(x)
    p, ok = _positions(n)
    return _out(_signed(np.where(ok, u | (np.uint64(1) << p), u)), x, n)


def _ibclr(x, n):
    u = _bits(x)
    p, ok = _positions(n)
    return _out(_signed(np.where(ok, u & ~(np.uint64(1) << p) & _MASK32, u)), x, n)


def _btest(x, n):
    p, ok = _positions(n)
    return _out(np.where(ok, (_bits(x) >> p) & np.uint64(1), 0).astype(np.float64), x, n)


def _field(x, pos, length, signed):
    p, ok = _positions(pos)
    w = _int(length)
    ok = ok & (w >= 1) & (w <= 32) & (p.astype(np.int64) + w <= 64)
    w_safe = np.where(ok, w, 1).astype(np.uint64)
    mask = (np.uint64(1) << w_safe) - np.uint64(1)          # w <= 32, no overflow
    value = ((_bits(x) >> p) & mask).astype(np.int64)
    if signed:
        half = (np.int64(1) << (w_safe.astype(np.int64) - 1))
        value = np.where(value >= half, value - (half << 1), value)
    return _out(np.where(ok, value, 0).astype(np.float64), x, pos, length)


def _ishft(x, n):
    u = _bits(x)
    k = _int(n)
    left = (u << np.clip(k, 0, 63).astype(np.uint64)) & _MASK32
    right = u >> np.clip(-k, 0, 63).astype(np.uint64)
    out = np.where(np.abs(k) >= 32, np.uint64(0), np.where(k >= 0, left, right))
    return _out(_signed(out.astype(np.uint64)), x, n)


def _ishftc(x, n, size=32.0):
    u = _bits(x)
    size_i = _int(size)
    ok = (size_i >= 1) & (size_i <= 32)
    size_s = np.where(ok, size_i, 32)
    k = np.mod(_int(n), size_s)                              # rotate left by k within the field
    mask = ((np.uint64(1) << size_s.astype(np.uint64)) - np.uint64(1)) & _MASK32
    mask = np.where(size_s == 32, _MASK32, mask)
    field = u & mask
    k_u = k.astype(np.uint64)
    back = (size_s - k).astype(np.uint64)
    rotated = np.where(k == 0, field, ((field << k_u) | (field >> back)) & mask)
    out = np.where(ok, (u & ~mask & _MASK32) | rotated, u)
    return _out(_signed(out.astype(np.uint64)), x, n, size)


def _logic(func):
    def run(a, b):
        return _out(_signed(func(_bits(a), _bits(b)) & _MASK32), a, b)
    return run


def _dword_to_float(x):
    pattern = np.atleast_1d(_bits(x).astype(np.uint32))
    values = pattern.view(np.float32).astype(np.float64).reshape(np.shape(x))
    return _out(values, x)


def _map(x, *points):
    """Characteristic through (p1,q1)..(pN,qN), linear in between, end values held outside.

    At a repeated break point (a step) the value of the neighbouring sloped or
    flat segment applies, so ``map(x,-10,0,-10,1,10,1,10,0)`` is 1 for
    ``-10 <= x <= 10`` as the help states. Constant break points are sorted.
    """
    if len(points) < 2 or len(points) % 2:
        raise ExprError("MAP needs pairs of break points: MAP(x, p1, q1, ..., pN, qN)")
    xs = _arr(x)
    p = [_arr(v) for v in points[0::2]]
    q = [_arr(v) for v in points[1::2]]
    if all(a.ndim == 0 for a in p):
        order = sorted(range(len(p)), key=lambda i: float(p[i]))
        p, q = [p[i] for i in order], [q[i] for i in order]
    shape = np.broadcast_shapes(xs.shape, *(a.shape for a in p), *(a.shape for a in q))
    xb = np.broadcast_to(xs, shape)
    pb = [np.broadcast_to(a, shape) for a in p]
    qb = [np.broadcast_to(a, shape) for a in q]
    out = np.full(shape, np.nan)
    below, above = xb < pb[0], xb > pb[-1]
    out[below] = qb[0][below]
    out[above] = qb[-1][above]
    done = below | above
    for i in range(len(pb) - 1):
        a, b = pb[i], pb[i + 1]
        seg = ~done & (a < b) & (xb >= a) & (xb <= b)
        if seg.any():
            out[seg] = qb[i][seg] + (xb[seg] - a[seg]) * (qb[i + 1][seg] - qb[i][seg]) / (b[seg] - a[seg])
            done |= seg
    rest = ~done
    if rest.any():                                           # only steps around x (or a single pair)
        for i in range(len(pb)):
            hit = rest & (xb == pb[i])
            out[hit] = qb[i][hit]
    return _out(out, x, *points)


def _switch(c, a, b):
    return _out(np.where(_arr(c) != 0, _arr(a), _arr(b)), c, a, b)


def _reduce(func):
    def run(*args):
        out = _arr(args[0])
        for a in args[1:]:
            out = func(out, _arr(a))
        return _out(out, *args)
    return run


def _compare(func):
    return lambda a, b: _out(func(_arr(a), _arr(b)).astype(np.float64), a, b)


def _unary(func):
    return lambda v: _out(func(_arr(v)), v)


# name -> (callable, minimum arguments, maximum arguments or None, description)
_FUNCS: Dict[str, Tuple[Callable, int, object, str]] = {
    "ABS": (_unary(np.abs), 1, 1, "absolute value"),
    "AINT": (_unary(np.trunc), 1, 1, "cut off the fraction"),
    "ANINT": (_unary(lambda v: np.sign(v) * np.floor(np.abs(v) + 0.5)), 1, 1, "round to the nearest whole number"),
    "SGN": (_unary(lambda v: np.where(v >= 0, 1.0, -1.0)), 1, 1, "+1 if x >= 0, else -1"),
    "SQRT": (_unary(np.sqrt), 1, 1, "square root"),
    "EXP": (_unary(np.exp), 1, 1, "e to the power of x"),
    "SIN": (_unary(np.sin), 1, 1, "sine (radian)"), "COS": (_unary(np.cos), 1, 1, "cosine (radian)"),
    "TAN": (_unary(np.tan), 1, 1, "tangent (radian)"),
    "LOG": (_unary(np.log), 1, 1, "natural logarithm"), "LOG10": (_unary(np.log10), 1, 1, "logarithm to base 10"),
    "SINH": (_unary(np.sinh), 1, 1, "hyperbolic sine"), "COSH": (_unary(np.cosh), 1, 1, "hyperbolic cosine"),
    "TANH": (_unary(np.tanh), 1, 1, "hyperbolic tangent"),
    "ASIN": (_unary(np.arcsin), 1, 1, "arc sine (radian)"), "ACOS": (_unary(np.arccos), 1, 1, "arc cosine (radian)"),
    "ATAN": (_unary(np.arctan), 1, 1, "arc tangent (radian)"),
    "MOD": (lambda a, b: _out(np.fmod(_arr(a), _arr(b)), a, b), 2, 2, "remainder of x / y (sign of x)"),
    "MIN": (_reduce(np.minimum), 1, None, "smallest argument"),
    "MAX": (_reduce(np.maximum), 1, None, "largest argument"),
    "NOT": (_not, 1, 1, "bitwise negation"),
    "IBSET": (_ibset, 2, 2, "set bit n"),
    "IBCLR": (_ibclr, 2, 2, "clear bit n"),
    "IAND": (_logic(np.bitwise_and), 2, 2, "bitwise AND"),
    "IOR": (_logic(np.bitwise_or), 2, 2, "bitwise OR"),
    "IEOR": (_logic(np.bitwise_xor), 2, 2, "bitwise exclusive OR"),
    "BTEST": (_btest, 2, 2, "1 if bit n is set, else 0"),
    "IBITS": (lambda x, p, n: _field(x, p, n, False), 3, 3, "bit field of n bits starting at position p"),
    "SBITS": (lambda x, p, n: _field(x, p, n, True), 3, 3, "signed bit field of n bits at position p"),
    "ISHFT": (_ishft, 2, 2, "shift by n bits, n>0 left, n<0 right"),
    "ISHFTC": (_ishftc, 2, 3, "rotate the right m bits by n bits"),
    "EQ": (_compare(np.equal), 2, 2, "1 if x == y"), "NE": (_compare(np.not_equal), 2, 2, "1 if x != y"),
    "LT": (_compare(np.less), 2, 2, "1 if x < y"), "LE": (_compare(np.less_equal), 2, 2, "1 if x <= y"),
    "GE": (_compare(np.greater_equal), 2, 2, "1 if x >= y"), "GT": (_compare(np.greater), 2, 2, "1 if x > y"),
    "SWITCH": (_switch, 3, 3, "p2 if p1 is not 0, else p3"),
    "MAP": (_map, 3, None, "characteristic through the break points (p1,q1)..(pN,qN), linear in between"),
}
_CONSTANTS = {"e": math.e, "pi": math.pi}
_VARIABLES = {"x", "t"} | {f"x{i}" for i in range(1, 10)}

#: Function name -> short description, for the editor's help.
FUNCTIONS: Dict[str, str] = {name: spec[3] for name, spec in _FUNCS.items()}
FUNCTIONS["(DWORD~FLOAT)"] = "interpret a 32 bit integer bit pattern as IEEE float"


def _fold(first: Node, rest: Tuple[Tuple[str, Node], ...]) -> Node:
    """Left fold of a flat operator chain, evaluated in a loop (no deep recursion)."""
    def run(env):
        acc = first(env)
        for op, node in rest:
            value = node(env)
            if op == "+":
                acc = acc + value
            elif op == "-":
                acc = acc - value
            elif op == "*":
                acc = acc * value
            else:
                acc = np.true_divide(acc, value)
        return acc
    return run


class CompiledExpr:
    """A parsed expression. ``symbols`` are the variables it uses (lower case)."""

    def __init__(self, text: str, node: Node, symbols: Set[str]) -> None:
        self.text = text
        self._node = node
        self.symbols = symbols

    def evaluate(self, env: Mapping[str, object]) -> object:
        """Evaluate with ``env`` mapping variable -> array or number."""
        missing = self.symbols - set(env)
        if missing:
            raise ExprError("undefined variable " + ", ".join(sorted(missing)))
        values = {k: (np.asarray(v, dtype=np.float64) if isinstance(v, np.ndarray) else v) for k, v in env.items()}
        with np.errstate(all="ignore"):
            try:
                return self._node(values)
            except ExprError:
                raise
            except RecursionError as exc:
                raise ExprError("expression is too complex") from exc
            except (ValueError, TypeError, OverflowError, ZeroDivisionError) as exc:
                raise ExprError(str(exc)) from exc


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.tokens: List[Tuple[str, str]] = []
        self.symbols: Set[str] = set()
        if len(text) > _MAX_LENGTH:
            raise ExprError("expression is too long")
        pos = 0
        while pos < len(text):
            if text[pos:].strip() == "":
                break
            m = _TOKEN.match(text, pos)
            if not m:
                raise ExprError(f"unexpected character '{text[pos:].strip()[:1]}' at position {pos + 1}")
            kind = m.lastgroup
            self.tokens.append((kind, m.group(kind)))
            pos = m.end()
        self.i = 0
        self.depth = 0

    def peek(self) -> Tuple[str, str]:
        return self.tokens[self.i] if self.i < len(self.tokens) else ("end", "")

    def take(self) -> Tuple[str, str]:
        tok = self.peek()
        self.i += 1
        return tok

    def expect(self, value: str) -> None:
        _kind, tok = self.take()
        if tok != value:
            raise ExprError(f"'{value}' expected" + (f" but found '{tok}'" if tok else " at the end"))

    def parse(self) -> Node:
        if not self.tokens:
            raise ExprError("expression is empty")
        node = self.expr()
        if self.peek()[0] != "end":
            raise ExprError(f"unexpected '{self.peek()[1]}'")
        return node

    # expr  := term (('+'|'-') term)*
    def expr(self) -> Node:
        self.depth += 1
        if self.depth > _MAX_NESTING:
            raise ExprError("expression is nested too deeply")
        first = self.term()
        rest: List[Tuple[str, Node]] = []
        while self.peek()[1] in ("+", "-"):
            op = self.take()[1]
            rest.append((op, self.term()))
        self.depth -= 1
        return _fold(first, tuple(rest)) if rest else first

    # term  := unary (('*'|'/') unary)*
    def term(self) -> Node:
        first = self.unary()
        rest: List[Tuple[str, Node]] = []
        while self.peek()[1] in ("*", "/"):
            op = self.take()[1]
            rest.append((op, self.unary()))
        return _fold(first, tuple(rest)) if rest else first

    # unary := ('+'|'-')* power
    def unary(self) -> Node:
        negative = False
        while self.peek()[1] in ("+", "-"):
            negative ^= self.take()[1] == "-"
        node = self.power()
        if negative:
            return lambda env, n=node: -n(env)
        return node

    # power := cast ('**' ('+'|'-')* cast)*      (right associative)
    def power(self) -> Node:
        base = self.cast()
        chain: List[Tuple[bool, Node]] = []
        while self.peek()[1] == "**":
            self.take()
            negative = False
            while self.peek()[1] in ("+", "-"):
                negative ^= self.take()[1] == "-"
            chain.append((negative, self.cast()))
        if not chain:
            return base
        items = tuple(chain)

        def run(env):
            negative, node = items[-1]
            acc = node(env)
            acc = -acc if negative else acc
            for negative, node in reversed(items[:-1]):
                acc = np.power(_arr(node(env)), acc)
                acc = -acc if negative else acc
            return np.power(_arr(base(env)), acc)
        return run

    # cast  := '(DWORD~FLOAT)'* primary
    def cast(self) -> Node:
        casts = 0
        while self.peek()[0] == "cast":
            self.take()
            casts += 1
        node = self.primary()
        if not casts:
            return node

        def run(env):
            value = node(env)
            for _ in range(casts):
                value = _dword_to_float(value)
            return value
        return run

    def primary(self) -> Node:
        kind, tok = self.take()
        if kind == "num":
            value = float(tok)
            return lambda env: value
        if tok == "(":
            node = self.expr()
            self.expect(")")
            return node
        if kind == "name":
            low = tok.lower()
            if self.peek()[1] == "(":
                spec = _FUNCS.get(tok.upper())
                if spec is None:
                    raise ExprError(f"unknown function '{tok}'")
                self.take()
                args: List[Node] = []
                if self.peek()[1] != ")":
                    args.append(self.expr())
                    while self.peek()[1] == ",":
                        self.take()
                        args.append(self.expr())
                self.expect(")")
                func, lo, hi, _ = spec
                if len(args) < lo or (hi is not None and len(args) > hi):
                    need = f"{lo}" if hi == lo else (f"at least {lo}" if hi is None else f"{lo} to {hi}")
                    raise ExprError(f"{tok.upper()} takes {need} argument(s), {len(args)} given")
                if tok.upper() == "MAP" and len(args) % 2 == 0:
                    raise ExprError("MAP needs pairs of break points: MAP(x, p1, q1, ..., pN, qN)")
                return lambda env: func(*[a(env) for a in args])
            if low in _VARIABLES:
                self.symbols.add(low)
                return lambda env: env[low]
            if low in _CONSTANTS:
                value = _CONSTANTS[low]
                return lambda env: value
            raise ExprError(f"unknown name '{tok}' (variables are x, t, x1..x9; constants E, PI)")
        raise ExprError(f"unexpected '{tok}'" if tok else "expression ends unexpectedly")


def compile_expr(text: str) -> CompiledExpr:
    """Parse ``text``. Raises :class:`ExprError` with a readable message."""
    try:
        parser = _Parser(text)
        node = parser.parse()
    except RecursionError as exc:
        raise ExprError("expression is too complex") from exc
    return CompiledExpr(text, node, parser.symbols)
