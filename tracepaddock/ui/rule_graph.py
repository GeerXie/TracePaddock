"""Logic-chain picture of a rule: signals -> conditions -> AND/OR -> trigger / expectation -> verdict.

The picture is generated from the rule (it is always consistent with it).
Double-clicking a node asks for the rule to be edited; the last results are
written into the trigger and verdict nodes.

Layout: every section (trigger, scope, expectation) has its own signal nodes,
one row per signal (an operand signal gets a row of its own). Columns are as
wide as their widest node; long texts are wrapped. Nodes right of the
conditions sit at the height of the nodes feeding them and are pushed down
where they would overlap.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from ..i18n import tr
from ..rules.model import Condition, Group, Operand, Rule

__all__ = ["RuleGraphView", "build_graph", "wrap_text"]

_COLORS = {"signal": (220, 235, 250), "condition": (255, 255, 255), "group": (255, 243, 205),
           "trigger": (255, 224, 190), "scope": (235, 235, 235), "expect": (214, 240, 220),
           "verdict": (240, 240, 240), "verdict_ok": (190, 235, 200), "verdict_bad": (250, 200, 200)}
_ALREADY = {"hold": "already right: counts (delay 0)", "new_sample": "a new frame after the trigger",
            "change": "must change after the trigger"}


def _condition_label(cond: Condition, names: Dict[str, Dict[float, str]]) -> str:
    text = cond.text(names)
    return text[len(cond.signal):].strip() if text.startswith(cond.signal) else text


def build_graph(rule: Rule, summary: Optional[Dict[str, object]] = None,
                names: Optional[Dict[str, Dict[float, str]]] = None) -> Tuple[List[dict], List[tuple]]:
    """Nodes ``{"id", "kind", "text", "col", "row"}`` and edges ``(from, to, dashed)``."""
    names = names or {}
    nodes: List[dict] = []
    edges: List[tuple] = []
    rows = {"next": 0.0}

    def node(kind: str, text: str, col: int, row: float) -> str:
        nid = "n%d" % len(nodes)
        nodes.append({"id": nid, "kind": kind, "text": text, "col": col, "row": row})
        return nid

    def row_of(nid: str) -> float:
        return next(n["row"] for n in nodes if n["id"] == nid)

    def section(group: Optional[Group], label: str, kind: str = "group", extra: str = "") -> Optional[str]:
        if group is None or not group.items:
            return None
        signal_ids: Dict[str, str] = {}                    # own signal nodes per section
        first = rows["next"]
        cond_ids = []
        for item in group.items:
            r0 = rows["next"]
            operand = item.value.signal if isinstance(item, Condition) and isinstance(item.value, Operand) else None
            if isinstance(item, Condition):
                used = list(dict.fromkeys([item.signal] + ([operand] if operand else [])))
                text = _condition_label(item, names)
            else:
                used = list(dict.fromkeys(item.signals()))
                text = "(" + item.text(names) + ")"
            new = [n for n in used if n not in signal_ids]
            for i, name in enumerate(new):
                signal_ids[name] = node("signal", name, 0, r0 + i)
            height = max(1, len(new))
            cid = node("condition", text, 1, r0 + (height - 1) / 2.0)
            for name in used:
                edges.append((signal_ids[name], cid, name == operand and name != getattr(item, "signal", None)))
            rows["next"] = r0 + height
            cond_ids.append(cid)
        lines = [label]
        if len(cond_ids) > 1 or group.negate:
            lines.append(("NOT " if group.negate else "") + ("AND" if group.mode == "and" else "OR"))
        if extra:
            lines.append(extra)
        gid = node(kind, "\n".join(lines), 2, (first + rows["next"] - 1) / 2.0)
        for cid in cond_ids:
            edges.append((cid, gid, False))
        rows["next"] += 0.5
        return gid

    trig = section(rule.trigger, tr("Trigger"))
    scope = section(rule.scope, tr("Scope"), "scope", tr("only while the scope holds"))
    expect = section(rule.expect, tr("Expectation"))
    mode = tr("becomes true") if rule.trigger_mode == "becomes_true" else tr("while true")
    text = tr("Trigger: %s") % mode
    if rule.trigger_hold:
        text += "\n" + tr("at least %g s") % rule.trigger_hold
    if summary is not None:
        text += "\n" + tr("%d events") % int(summary.get("events", 0))
    tid = node("trigger", text, 3, row_of(trig) if trig else 0)
    if trig:
        edges.append((trig, tid, False))
    if scope:
        edges.append((scope, tid, True))
    if expect:
        etext = tr("Check: within %g .. %g s") % (rule.within_min, rule.within_max)
        if rule.expect_hold:
            etext += "\n" + tr("kept at least %g s") % rule.expect_hold
        etext += "\n" + tr(_ALREADY.get(rule.already, rule.already))
        cid = node("expect", etext, 3, row_of(expect))
        edges.append((expect, cid, False))
        edges.append((tid, cid, False))
        last = cid
    else:
        last = tid
    if summary is not None and expect:
        vtext = tr("%d pass / %d fail") % (int(summary.get("PASS", 0)),
                                           int(summary.get("FAIL", 0)) + int(summary.get("TOO_EARLY", 0)))
        if summary.get("INCONCLUSIVE"):
            vtext += "\n" + tr("%d not decided") % int(summary["INCONCLUSIVE"])
        if "delay_max" in summary:
            vtext += "\n" + tr("delay %.0f .. %.0f ms") % (summary["delay_min"] * 1000, summary["delay_max"] * 1000)
        kind = "verdict_bad" if int(summary.get("FAIL", 0)) + int(summary.get("TOO_EARLY", 0)) else "verdict_ok"
    else:
        vtext, kind = (tr("Events") if not expect else tr("Verdict")), "verdict"
    vid = node(kind, vtext, 4, row_of(last))
    edges.append((last, vid, False))
    return nodes, edges


# ---------------------------------------------------------------------- text wrapping
def _split_long(word: str, metrics: QtGui.QFontMetricsF, max_w: float) -> List[str]:
    """Cut a piece that alone is wider than ``max_w`` (e.g. Chinese text without spaces)."""
    out = []
    while len(word) > 1 and metrics.horizontalAdvance(word) > max_w:
        cut = len(word) - 1
        while cut > 1 and metrics.horizontalAdvance(word[:cut]) > max_w:
            cut -= 1
        out.append(word[:cut])
        word = word[cut:]
    out.append(word)
    return out


def wrap_text(text: str, metrics: QtGui.QFontMetricsF, max_w: float, max_lines: int = 4) -> str:
    """Wrap every line of ``text`` at spaces (after ", " of value lists) to at most ``max_w`` pixels;
    more than ``max_lines`` lines end with "…" (the tooltip shows the whole text)."""
    lines: List[str] = []
    for raw in text.split("\n"):
        current = ""
        for piece in re.split(r"(?<= )", raw):
            if not piece:
                continue
            if current and metrics.horizontalAdvance(current + piece.rstrip()) > max_w:
                lines.append(current.rstrip())
                current = ""
            parts = _split_long(current + piece, metrics, max_w)
            lines.extend(p.rstrip() for p in parts[:-1])
            current = parts[-1]
        lines.append(current.rstrip())
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1] + " …"
    return "\n".join(lines)


class _Node(QtWidgets.QGraphicsPathItem):
    def __init__(self, data: dict, rect: QtCore.QRectF) -> None:
        path = QtGui.QPainterPath()
        path.addRoundedRect(rect, 8, 8)
        super().__init__(path)
        self.data_ = data
        self.setBrush(QtGui.QColor(*_COLORS.get(data["kind"], (255, 255, 255))))
        self.setPen(QtGui.QPen(QtGui.QColor(120, 120, 120), 1.2))
        self.setToolTip(data["text"])
        self.setCursor(QtCore.Qt.PointingHandCursor)


class RuleGraphView(QtWidgets.QGraphicsView):
    """Shows the logic chain of one rule. ``nodeActivated(kind)`` on a double click."""

    nodeActivated = QtCore.Signal(str)

    MIN_W = (150, 170, 110, 190, 150)                       # smallest node width per column
    MAX_TEXT_W = (260, 330, 220, 300, 260)                  # texts wider than this are wrapped
    GAP_X = 60
    GAP_Y = 12
    ROW_H = 44
    LINE_H = 18

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setRenderHint(QtGui.QPainter.Antialiasing, True)
        self.setRenderHint(QtGui.QPainter.TextAntialiasing, True)
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.setBackgroundBrush(QtGui.QColor(250, 250, 250))
        self._items: Dict[str, _Node] = {}

    # ------------------------------------------------------------------ layout
    def compute_layout(self, nodes: List[dict], edges: List[tuple], metrics: QtGui.QFontMetricsF) -> Dict[str, QtCore.QRectF]:
        """Rectangles of the nodes (see the module text)."""
        texts, sizes = {}, {}
        for n in nodes:
            col = n["col"]
            text = wrap_text(n["text"], metrics, self.MAX_TEXT_W[col])
            lines = text.split("\n")
            texts[n["id"]] = text
            sizes[n["id"]] = (max(metrics.horizontalAdvance(line) for line in lines) + 26,
                              max(30.0, self.LINE_H * len(lines) + 12.0))
        cols = sorted({n["col"] for n in nodes})
        width = {c: max([self.MIN_W[c]] + [sizes[n["id"]][0] for n in nodes if n["col"] == c]) for c in cols}
        x, left = {}, 0.0
        for c in range(max(cols) + 1 if cols else 0):
            x[c] = left
            left += width.get(c, 0.0) + (self.GAP_X if c in width else 0.0)
        centre: Dict[str, float] = {}
        # signals and conditions: by rows, pushed down below the node above in the same column
        bottom: Dict[int, float] = {}
        prev_row, prev_y = None, 0.0
        for row in sorted({n["row"] for n in nodes if n["col"] <= 1}):
            here = [n for n in nodes if n["col"] <= 1 and n["row"] == row]
            y = prev_y + (row - prev_row) * self.ROW_H if prev_row is not None else 0.0
            for n in here:
                if n["col"] in bottom:
                    y = max(y, bottom[n["col"]] + self.GAP_Y + sizes[n["id"]][1] / 2)
            for n in here:
                centre[n["id"]] = y
                bottom[n["col"]] = y + sizes[n["id"]][1] / 2
            prev_row, prev_y = row, y
        # further columns: at the mean height of the solid edges from the columns on the left
        by_id = {n["id"]: n for n in nodes}
        for c in [c for c in cols if c >= 2]:
            column = [n for n in nodes if n["col"] == c]
            for n in column:
                feeds = [centre[a] for a, b, dashed in edges
                         if b == n["id"] and not dashed and by_id[a]["col"] < c and a in centre]
                centre[n["id"]] = sum(feeds) / len(feeds) if feeds else n["row"] * self.ROW_H
            low = None
            for n in sorted(column, key=lambda n: (centre[n["id"]], n["row"])):
                half = sizes[n["id"]][1] / 2
                if low is not None and centre[n["id"]] - half < low + self.GAP_Y * 2:
                    centre[n["id"]] = low + self.GAP_Y * 2 + half
                low = centre[n["id"]] + half
        self._texts = texts
        return {n["id"]: QtCore.QRectF(x[n["col"]], centre[n["id"]] - sizes[n["id"]][1] / 2,
                                       width[n["col"]], sizes[n["id"]][1]) for n in nodes}

    def show_rule(self, rule: Optional[Rule], summary: Optional[Dict[str, object]] = None,
                  names: Optional[Dict[str, Dict[float, str]]] = None) -> None:
        scene = self.scene()
        scene.clear()
        self._items.clear()
        if rule is None:
            text = scene.addSimpleText(tr("Choose a rule to see its logic chain."))
            text.setBrush(QtGui.QColor(130, 130, 130))
            return
        nodes, edges = build_graph(rule, summary, names)
        font = QtGui.QFont("Microsoft YaHei UI")
        font.setPixelSize(12)
        metrics = QtGui.QFontMetricsF(font)
        rects = self.compute_layout(nodes, edges, metrics)
        for data in nodes:
            rect = rects[data["id"]]
            item = _Node(data, rect)
            scene.addItem(item)
            label = QtWidgets.QGraphicsSimpleTextItem(self._texts[data["id"]], item)
            label.setFont(font)
            br = label.boundingRect()
            label.setPos(rect.center().x() - br.width() / 2, rect.center().y() - br.height() / 2)
            self._items[data["id"]] = item
        # incoming edges from the left side are spread over the left edge of a node, top to bottom
        ports: Dict[tuple, float] = {}
        for nid in rects:
            incoming = sorted([(rects[a].center().y(), a) for a, b, _d in edges
                               if b == nid and rects[a].right() <= rects[nid].left()])
            rb = rects[nid]
            step = min(8.0, rb.height() / (len(incoming) + 1))
            for i, (_y, a) in enumerate(incoming):
                ports[(a, nid)] = rb.center().y() + (i - (len(incoming) - 1) / 2.0) * step
        for a, b, dashed in edges:
            ra, rb = rects[a], rects[b]
            if ra.right() <= rb.left():
                start = QtCore.QPointF(ra.right(), ra.center().y())
                end = QtCore.QPointF(rb.left(), ports.get((a, b), rb.center().y()))
            else:                                              # same column (trigger -> check): go down
                start = QtCore.QPointF(ra.center().x(), ra.bottom())
                end = QtCore.QPointF(rb.center().x(), rb.top())
            path = QtGui.QPainterPath(start)
            if abs(start.y() - end.y()) > 0.5 and ra.right() <= rb.left():
                mid = (start.x() + end.x()) / 2
                path.cubicTo(QtCore.QPointF(mid, start.y()), QtCore.QPointF(mid, end.y()), end)
            else:
                path.lineTo(end)
            pen = QtGui.QPen(QtGui.QColor(110, 110, 110), 1.3, QtCore.Qt.DashLine if dashed else QtCore.Qt.SolidLine)
            edge = scene.addPath(path, pen)
            edge.setZValue(-1)
            angle = path.angleAtPercent(1.0)
            head = QtGui.QPolygonF([QtCore.QPointF(0, 0), QtCore.QPointF(-9, -4), QtCore.QPointF(-9, 4)])
            arrow = scene.addPolygon(head, QtGui.QPen(QtCore.Qt.NoPen), QtGui.QBrush(QtGui.QColor(110, 110, 110)))
            arrow.setPos(end)
            arrow.setRotation(-angle)
        scene.setSceneRect(scene.itemsBoundingRect().adjusted(-20, -20, 20, 20))

    def node_rects(self) -> Dict[str, QtCore.QRectF]:
        return {nid: item.path().boundingRect() for nid, item in self._items.items()}

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        item = self.itemAt(event.pos())
        while item is not None and not isinstance(item, _Node):
            item = item.parentItem()
        if isinstance(item, _Node):
            self.nodeActivated.emit(item.data_["kind"])
            return
        super().mouseDoubleClickEvent(event)

    def picture(self) -> QtGui.QImage:
        rect = self.scene().sceneRect()
        image = QtGui.QImage(int(rect.width()), int(rect.height()), QtGui.QImage.Format_ARGB32)
        image.fill(QtGui.QColor(255, 255, 255))
        painter = QtGui.QPainter(image)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        self.scene().render(painter, QtCore.QRectF(image.rect()), rect)
        painter.end()
        return image
