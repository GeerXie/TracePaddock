"""Excel report of rule results: overview, all events, settings."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .engine import Event

__all__ = ["RunResult", "write_xlsx", "VERDICT_TEXT"]

VERDICT_TEXT = {"PASS": "通过", "FAIL": "失败", "TOO_EARLY": "过早", "INCONCLUSIVE": "未判定", "FOUND": "找到"}
_FILL = {"PASS": "C6EFCE", "FAIL": "FFC7CE", "TOO_EARLY": "FFE0B2", "INCONCLUSIVE": "E7E6E6", "FOUND": "DDEBF7"}
#: control characters an Excel file cannot hold (the same set openpyxl refuses)
_ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _clean(values: Sequence[object]) -> List[object]:
    """The cells of a row; control characters in texts (a log line with colour codes ...) become '?'."""
    return [_ILLEGAL.sub("?", v) if isinstance(v, str) else v for v in values]


def _overview_fill(res: "RunResult") -> Optional[str]:
    """Colour of the rule's name in the overview: red for a failure or an error, green when something passed
    (or, for a rule without an expectation, was found), grey when nothing could be decided, none without events."""
    if res.count("FAIL") or res.count("TOO_EARLY") or res.error:
        return _FILL["FAIL"]
    if res.count("PASS"):
        return _FILL["PASS"]
    if res.count("INCONCLUSIVE"):
        return _FILL["INCONCLUSIVE"]
    return _FILL["PASS"] if res.events else None


@dataclass
class RunResult:
    """Result of one item of the rule list (a form rule, a script, a plug-in run)."""

    name: str
    kind: str                                   # "form", "script", "plugin"
    events: List[Event] = field(default_factory=list)
    error: Optional[str] = None
    trigger_text: str = ""
    expect_text: str = ""
    response_period: Optional[float] = None
    log: List[str] = field(default_factory=list)

    def count(self, verdict: str) -> int:
        return sum(1 for e in self.events if e.verdict == verdict)

    def delays(self) -> List[float]:
        return [e.delay for e in self.events if e.verdict == "PASS" and e.delay is not None]


def _value_columns(results: Sequence[RunResult]) -> List[str]:
    names: List[str] = []
    for res in results:
        for e in res.events:
            for key in list(e.values) + list(e.actual):
                if key not in names:
                    names.append(key)
    return names


def write_xlsx(path: "str | Path", results: Sequence[RunResult], meta: Dict[str, str]) -> None:
    """Write the report. Raises ``OSError`` / ``ImportError`` with a message."""
    import openpyxl
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    book = openpyxl.Workbook()
    bold = Font(bold=True)
    head_fill = PatternFill("solid", fgColor="DDDDDD")

    def header(sheet, titles):
        sheet.append(_clean(titles))
        for c in range(1, len(titles) + 1):
            cell = sheet.cell(row=1, column=c)
            cell.font, cell.fill = bold, head_fill
            cell.alignment = Alignment(vertical="center", wrap_text=True)
        sheet.freeze_panes = "A2"

    def widths(sheet, values):
        for i, w in enumerate(values, start=1):
            sheet.column_dimensions[get_column_letter(i)].width = w

    # overview
    ov = book.active
    ov.title = "概览"
    header(ov, ["规则", "类型", "事件数", "通过", "失败", "过早", "未判定", "找到", "最小延迟 [ms]", "平均延迟 [ms]",
                "最大延迟 [ms]", "响应信号周期 [ms]", "触发条件", "期望", "问题"])
    for res in results:
        d = res.delays()
        ov.append(_clean([res.name, {"form": "表单", "script": "脚本", "plugin": "插件"}.get(res.kind, res.kind),
                          len(res.events), res.count("PASS"), res.count("FAIL"), res.count("TOO_EARLY"),
                          res.count("INCONCLUSIVE"), res.count("FOUND"), round(min(d) * 1000, 1) if d else None,
                          round(sum(d) / len(d) * 1000, 1) if d else None, round(max(d) * 1000, 1) if d else None,
                          round(res.response_period * 1000, 1) if res.response_period else None, res.trigger_text,
                          res.expect_text, res.error or ""]))
        colour = _overview_fill(res)
        if colour:
            ov.cell(row=ov.max_row, column=1).fill = PatternFill("solid", fgColor=colour)
    widths(ov, [28, 8, 8, 7, 7, 7, 8, 7, 12, 12, 12, 14, 50, 50, 40])
    ov.auto_filter.ref = ov.dimensions

    # events
    ev = book.create_sheet("事件")
    extra = _value_columns(results)
    header(ev, ["#", "规则", "触发时刻 [s]", "结束 [s]", "判定", "延迟 [ms]", "触发前已满足", "满足时刻 [s]", "备注"]
           + ["%s" % n for n in extra] + ["%s (文字)" % n for n in extra])
    number = 0
    for res in results:
        for e in sorted(res.events, key=lambda x: x.t0):
            number += 1
            values = {**e.values, **{k: v for k, v in e.actual.items() if k not in e.values}}
            texts = {**e.texts, **{k: v for k, v in e.actual_texts.items() if k not in e.texts}}
            ev.append(_clean([number, res.name, round(e.t0, 6), None if e.t_end is None else round(e.t_end, 6),
                              VERDICT_TEXT.get(e.verdict, e.verdict),
                              None if e.delay is None else round(e.delay * 1000, 3), "是" if e.already else "",
                              None if e.t1 is None else round(e.t1, 6), e.note]
                             + [values.get(n) for n in extra] + [texts.get(n, "") for n in extra]))
            ev.cell(row=ev.max_row, column=5).fill = PatternFill("solid", fgColor=_FILL.get(e.verdict, "FFFFFF"))
    widths(ev, [6, 26, 13, 11, 8, 10, 12, 13, 30] + [16] * (2 * len(extra)))
    ev.auto_filter.ref = ev.dimensions

    # settings
    st = book.create_sheet("设置")
    header(st, ["项目", "内容"])
    for key, value in list(meta.items()) + [("报告时间", time.strftime("%Y-%m-%d %H:%M:%S"))]:
        st.append(_clean([key, value]))
    widths(st, [24, 100])
    for res in results:
        for line in res.log:
            st.append(_clean(["%s 日志" % res.name, line]))
    book.save(str(path))
