"""在制品滞留处置判断。

出口堵塞或退火温度告警闩锁期间，网带被压到保护低速，窑内会积一批在
制品；以前排堵后整批照常放行，出窑才发现一部分已经过冷或过热、不能
再用。本模块在现有退火记录（闩锁置位／释放事件与段温履历）之上补一
层处置判断：按滞留时长与滞留期间的窑内温度履历，逐批给出 继续／降级
／判废 的结论，每条结论都带上依据；窗口未闭合、温度履历缺失或覆盖率
不足时按待判挂起，不允许在没有依据的情况下放行。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import DispositionLimits

#: 处置结论取值。
CONTINUE = "continue"
DOWNGRADE = "downgrade"
SCRAP = "scrap"
PENDING = "pending"

#: 结论的中文标签，报告与控制台共用。
VERDICT_LABELS = {
    CONTINUE: "继续",
    DOWNGRADE: "降级",
    SCRAP: "判废",
    PENDING: "待判",
}


@dataclass(frozen=True)
class HoldupWindow:
    """一段滞留窗口：闩锁置位到释放之间，窑内积压的一批在制品。"""

    latch_epoch: int
    started_at: float
    released_at: float | None
    samples: tuple[tuple[float, float], ...] = ()
    release_missing: bool = False

    @property
    def closed(self) -> bool:
        """窗口是否随闩锁释放正常闭合。"""

        return self.released_at is not None and not self.release_missing

    def duration_s(self) -> float | None:
        """滞留时长；窗口未闭合时返回 None。"""

        if self.released_at is None:
            return None
        return self.released_at - self.started_at


@dataclass(frozen=True)
class Disposition:
    """一批滞留在制品的处置结论。"""

    episode_id: str
    verdict: str
    evidence: tuple[str, ...]
    metrics: dict = field(default_factory=dict)
    started_at: float = 0.0
    released_at: float | None = None

    def as_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "verdict": self.verdict,
            "verdict_label": VERDICT_LABELS[self.verdict],
            "evidence": list(self.evidence),
            "metrics": self.metrics,
            "started_at": round(self.started_at, 3),
            "released_at": None
            if self.released_at is None
            else round(self.released_at, 3),
        }


class DispositionBoard:
    """滞留批次处置板：读退火记录，逐批给出处置结论。"""

    def __init__(self, limits: DispositionLimits) -> None:
        self._limits = limits

    def evaluate(self, records: list[dict], now: float) -> dict:
        """按审计记录逐批判定，返回处置报告（含每批结论与汇总）。"""

        ordered = sorted(records, key=lambda item: float(item.get("ts", 0.0)))
        windows = self._extract_windows(ordered)
        episodes = [self._judge(window, float(now)).as_dict() for window in windows]
        summary = {verdict: 0 for verdict in VERDICT_LABELS}
        for episode in episodes:
            summary[episode["verdict"]] += 1
        summary["total"] = len(episodes)
        return {
            "generated_at": round(float(now), 3),
            "records_scanned": len(records),
            "limits": self._limits_snapshot(),
            "episodes": episodes,
            "summary": summary,
        }

    def _extract_windows(self, records: list[dict]) -> list[HoldupWindow]:
        """从记录里还原滞留窗口：闩锁置位开窗、释放闭窗，段温落入窗口。"""

        windows: list[HoldupWindow] = []
        open_window: dict | None = None
        for record in records:
            action = record.get("action")
            ts = float(record.get("ts", 0.0))
            detail = record.get("detail") or {}
            if action == "anneal_latch_set":
                if open_window is not None:
                    windows.append(self._close_abnormally(open_window))
                open_window = {
                    "epoch": int(detail.get("epoch", len(windows) + 1)),
                    "start": ts,
                    "samples": [],
                }
            elif action == "anneal_latch_released":
                if open_window is not None:
                    windows.append(
                        HoldupWindow(
                            latch_epoch=open_window["epoch"],
                            started_at=open_window["start"],
                            released_at=ts,
                            samples=tuple(open_window["samples"]),
                        )
                    )
                    open_window = None
            elif action == "anneal_zones_applied" and open_window is not None:
                mean_c = detail.get("mean_c")
                if mean_c is not None:
                    open_window["samples"].append((ts, float(mean_c)))
        if open_window is not None:
            windows.append(
                HoldupWindow(
                    latch_epoch=open_window["epoch"],
                    started_at=open_window["start"],
                    released_at=None,
                    samples=tuple(open_window["samples"]),
                )
            )
        return windows

    def _close_abnormally(self, open_window: dict) -> HoldupWindow:
        """记录里缺释放事件的窗口：被下一次置位截断，按不完整窗口收口。"""

        return HoldupWindow(
            latch_epoch=open_window["epoch"],
            started_at=open_window["start"],
            released_at=None,
            samples=tuple(open_window["samples"]),
            release_missing=True,
        )

    def _judge(self, window: HoldupWindow, now: float) -> Disposition:
        """对一批滞留在制品给出处置结论，依据不足的按待判挂起。"""

        episode_id = f"holdup-{window.latch_epoch}"
        metrics = self._metrics(window, now)
        pending = self._pending_reasons(window, metrics, now)
        if pending:
            return Disposition(
                episode_id=episode_id,
                verdict=PENDING,
                evidence=tuple(pending) + (self._history_note(metrics),),
                metrics=metrics,
                started_at=window.started_at,
                released_at=window.released_at,
            )

        limits = self._limits
        duration = metrics["duration_s"]
        scrap: list[str] = []
        downgrade: list[str] = []
        if duration >= limits.hold_scrap_s:
            scrap.append(
                f"滞留时长 {duration:.1f}s 达到判废线 {limits.hold_scrap_s:.1f}s"
            )
        if metrics["min_c"] <= limits.low_scrap_c:
            scrap.append(
                f"滞留期间窑内最低温 {metrics['min_c']:.1f}°C "
                f"低于过冷判废线 {limits.low_scrap_c:.1f}°C"
            )
        if metrics["max_c"] >= limits.high_scrap_c:
            scrap.append(
                f"滞留期间窑内最高温 {metrics['max_c']:.1f}°C "
                f"达到过热判废线 {limits.high_scrap_c:.1f}°C"
            )
        if duration >= limits.hold_downgrade_s:
            downgrade.append(
                f"滞留时长 {duration:.1f}s 达到降级线 {limits.hold_downgrade_s:.1f}s"
            )
        if metrics["min_c"] <= limits.low_downgrade_c:
            downgrade.append(
                f"滞留期间窑内最低温 {metrics['min_c']:.1f}°C "
                f"低于低温降级线 {limits.low_downgrade_c:.1f}°C"
            )
        if metrics["max_c"] >= limits.high_downgrade_c:
            downgrade.append(
                f"滞留期间窑内最高温 {metrics['max_c']:.1f}°C "
                f"达到高温降级线 {limits.high_downgrade_c:.1f}°C"
            )

        note = self._history_note(metrics)
        if scrap:
            verdict, evidence = SCRAP, scrap
        elif downgrade:
            verdict, evidence = DOWNGRADE, downgrade
        else:
            verdict = CONTINUE
            evidence = [
                f"滞留时长 {duration:.1f}s 低于降级线 {limits.hold_downgrade_s:.1f}s",
                f"滞留期间窑温 {metrics['min_c']:.1f}~{metrics['max_c']:.1f}°C "
                f"保持在 {limits.low_downgrade_c:.1f}~{limits.high_downgrade_c:.1f}°C "
                "允许带内",
            ]
        return Disposition(
            episode_id=episode_id,
            verdict=verdict,
            evidence=tuple(evidence) + (note,),
            metrics=metrics,
            started_at=window.started_at,
            released_at=window.released_at,
        )

    def _pending_reasons(
        self, window: HoldupWindow, metrics: dict, now: float
    ) -> list[str]:
        """收集"依据不足"的原因；为空表示可以下结论。"""

        reasons: list[str] = []
        if window.release_missing:
            reasons.append("记录中缺少闩锁释放事件，滞留窗口不完整，无法核定滞留时长")
            return reasons
        if not window.closed:
            held = max(0.0, now - window.started_at)
            reasons.append(
                f"滞留尚未结束（闩锁仍置位），已滞留 {held:.1f}s，时长仍在增长"
            )
            return reasons
        if metrics["samples"] < self._limits.min_samples:
            reasons.append(
                f"滞留期间窑温履历只有 {metrics['samples']} 个采样点，"
                f"不足判定所需 {self._limits.min_samples} 个"
            )
        elif metrics["coverage"] < self._limits.min_coverage:
            reasons.append(
                f"温度履历覆盖率 {metrics['coverage'] * 100:.0f}% "
                f"低于要求 {self._limits.min_coverage * 100:.0f}%，"
                "滞留中段缺乏温度记录"
            )
        return reasons

    def _metrics(self, window: HoldupWindow, now: float) -> dict:
        """窗口的结构化指标：滞留时长、温度统计与履历覆盖率。"""

        duration = window.duration_s()
        if duration is None:
            duration = max(0.0, now - window.started_at)
        values = [value for _ts, value in window.samples]
        count = len(values)
        if count >= 2 and duration > 0:
            span = window.samples[-1][0] - window.samples[0][0]
            coverage = max(0.0, min(1.0, span / duration))
        elif count >= self._limits.min_samples and duration <= 0:
            coverage = 1.0
        else:
            coverage = 0.0
        return {
            "duration_s": round(duration, 3),
            "samples": count,
            "min_c": round(min(values), 3) if values else None,
            "max_c": round(max(values), 3) if values else None,
            "mean_c": round(sum(values) / count, 3) if values else None,
            "coverage": round(coverage, 4),
        }

    def _history_note(self, metrics: dict) -> str:
        """把窗口履历浓缩成一行摘要，作为每条结论的公共依据。"""

        if metrics["samples"] == 0:
            return f"滞留 {metrics['duration_s']:.1f}s，窗口内无窑温采样记录"
        return (
            f"履历摘要：滞留 {metrics['duration_s']:.1f}s，"
            f"窑温 {metrics['min_c']:.1f}~{metrics['max_c']:.1f}°C"
            f"（{metrics['samples']} 个采样点，"
            f"覆盖率 {metrics['coverage'] * 100:.0f}%）"
        )

    def _limits_snapshot(self) -> dict:
        """本次判定生效的阈值，随报告输出便于复盘对照。"""

        limits = self._limits
        return {
            "hold_downgrade_s": limits.hold_downgrade_s,
            "hold_scrap_s": limits.hold_scrap_s,
            "low_downgrade_c": limits.low_downgrade_c,
            "low_scrap_c": limits.low_scrap_c,
            "high_downgrade_c": limits.high_downgrade_c,
            "high_scrap_c": limits.high_scrap_c,
            "min_samples": limits.min_samples,
            "min_coverage": limits.min_coverage,
        }
