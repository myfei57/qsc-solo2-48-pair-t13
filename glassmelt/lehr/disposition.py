"""出口堵塞滞留在制品的处置判定。

出口一堵，窑里就积一批在制品；排堵之后这批制品不能不问青红皂白全部照常
往下走——滞留期间窑温履历偏离了退火窗口的，出窑不是开裂就是变形。这里在
现有退火记录之上补一层处置判断：退火段每落定一次段温就攒一个履历样本，
登记出口堵塞到排堵结束之间的样本就是这批在制品的窑内温度履历。判定时按
滞留时长与履历给出继续／降级／判废的结论并写明依据；履历样本不够、覆盖
率不足的，不硬判，按待判挂起，等履历补齐后再判。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from ..config import DispositionLimits
from ..errors import InterlockError, SequenceError
from ..eventbus import Event, EventBus

VERDICT_PROCEED = "proceed"
VERDICT_DOWNGRADE = "downgrade"
VERDICT_SCRAP = "scrap"
VERDICT_PENDING = "pending"

#: 各结论对应的审计严重度：判废必须进告警清单，降级与待判留痕即可。
_VERDICT_SEVERITY = {
    VERDICT_PROCEED: "info",
    VERDICT_DOWNGRADE: "notice",
    VERDICT_SCRAP: "warning",
    VERDICT_PENDING: "notice",
}

#: 终态结论不再翻案；只有待判批次允许补履历重判。
_FINAL_VERDICTS = (VERDICT_PROCEED, VERDICT_DOWNGRADE, VERDICT_SCRAP)


@dataclass(frozen=True)
class TempSample:
    """滞留期间采到的一个窑温点。"""

    at: float
    temperature_c: float

    def as_dict(self) -> dict:
        return {"at": round(self.at, 3), "temperature_c": round(self.temperature_c, 3)}


@dataclass(frozen=True)
class DispositionRecord:
    """一批滞留在制品的处置结论与依据。"""

    hold_id: str
    verdict: str
    reason: str
    evidence: dict
    decided_at: float

    def as_dict(self) -> dict:
        return {
            "hold_id": self.hold_id,
            "verdict": self.verdict,
            "reason": self.reason,
            "evidence": self.evidence,
            "decided_at": round(self.decided_at, 3),
        }


@dataclass
class _HoldBatch:
    """一批滞留在制品的登记与判定状态。"""

    hold_id: str
    reason: str
    started_at: float
    ended_at: float | None = None
    record: DispositionRecord | None = None


class DispositionBoard:
    """滞留在制品的处置判定台。"""

    def __init__(self, clock, bus: EventBus, limits: DispositionLimits) -> None:
        self._clock = clock
        self._bus = bus
        self._limits = limits
        self._samples: list[TempSample] = []
        self._batches: list[_HoldBatch] = []
        self._open: _HoldBatch | None = None
        self._counter = 0
        self._lock = threading.RLock()
        bus.subscribe("lehr", self._on_lehr_event)

    def _on_lehr_event(self, event: Event) -> None:
        """把退火段落定事件攒成窑温履历样本。"""

        if event.action != "anneal_zones_applied":
            return
        mean_c = event.payload.get("mean_c")
        if mean_c is None:
            return
        with self._lock:
            self._samples.append(
                TempSample(at=self._clock.now(), temperature_c=float(mean_c))
            )

    def note_hold(self, reason: str) -> dict:
        """登记出口堵塞：从当下起窑内积压的在制品归为新的一批。"""

        with self._lock:
            if self._open is not None:
                raise InterlockError(
                    f"上一批滞留 {self._open.hold_id} 尚未排堵，不能重复登记",
                    component="lehr",
                )
            self._counter += 1
            batch = _HoldBatch(
                hold_id=f"hold-{self._counter}",
                reason=reason,
                started_at=self._clock.now(),
            )
            self._batches.append(batch)
            self._open = batch
            hold_id = batch.hold_id
            started_at = batch.started_at
        self._bus.publish(
            Event.of(
                "lehr",
                "lehr",
                "lehr_hold_noted",
                subject="lehr_disposition",
                severity="warning",
                hold_id=hold_id,
                reason=reason,
            )
        )
        return {"hold_id": hold_id, "started_at": round(started_at, 3), "reason": reason}

    def clear_hold(self, reason: str) -> DispositionRecord:
        """排堵结束：对积压在窑内的这批在制品做处置判定。"""

        with self._lock:
            batch = self._open
            if batch is None:
                raise InterlockError("当前没有挂起的出口堵塞批次", component="lehr")
            batch.ended_at = self._clock.now()
            self._open = None
            hold_id = batch.hold_id
        return self.evaluate(hold_id)

    def evaluate(
        self, hold_id: str, extra_samples: list[dict] | None = None
    ) -> DispositionRecord:
        """对指定滞留批次做处置判定；已有终态结论的批次原样返回结论。

        ``extra_samples`` 用于待判批次补录履历，每项是
        ``{"at": 时间戳, "temperature_c": 温度}``。
        """

        with self._lock:
            batch = self._find(hold_id)
            if batch.record is not None and batch.record.verdict in _FINAL_VERDICTS:
                return batch.record
            if batch.ended_at is None:
                raise InterlockError(
                    f"滞留批次 {hold_id} 尚未排堵，不能判定", component="lehr"
                )
            ended_at = batch.ended_at
            samples = [
                sample
                for sample in self._samples
                if batch.started_at <= sample.at <= ended_at
            ]
            if extra_samples:
                samples.extend(
                    TempSample(
                        at=float(item["at"]),
                        temperature_c=float(item["temperature_c"]),
                    )
                    for item in extra_samples
                )
            samples.sort(key=lambda sample: sample.at)
            record = self._decide(batch, samples, ended_at)
            batch.record = record
        self._bus.publish(
            Event.of(
                "lehr",
                "lehr",
                "lehr_disposition_decided",
                subject="lehr_disposition",
                severity=_VERDICT_SEVERITY[record.verdict],
                hold_id=record.hold_id,
                verdict=record.verdict,
                reason=record.reason,
            )
        )
        return record

    def records(self) -> list[dict]:
        """全部滞留批次的处置记录；还在堵塞中的批次结论为 ``holding``。"""

        with self._lock:
            batches = list(self._batches)
        result: list[dict] = []
        for batch in batches:
            entry = {
                "hold_id": batch.hold_id,
                "hold_reason": batch.reason,
                "started_at": round(batch.started_at, 3),
                "ended_at": round(batch.ended_at, 3)
                if batch.ended_at is not None
                else None,
            }
            if batch.record is not None:
                entry.update(batch.record.as_dict())
            else:
                entry["verdict"] = "holding"
            result.append(entry)
        return result

    def snapshot(self) -> dict:
        with self._lock:
            by_verdict: dict[str, int] = {}
            for batch in self._batches:
                key = batch.record.verdict if batch.record is not None else "holding"
                by_verdict[key] = by_verdict.get(key, 0) + 1
            open_hold = self._open.hold_id if self._open is not None else None
            recent_samples = [sample.as_dict() for sample in self._samples[-3:]]
            sample_count = len(self._samples)
            batch_count = len(self._batches)
        return {
            "open_hold": open_hold,
            "batches": batch_count,
            "by_verdict": by_verdict,
            "samples": sample_count,
            "recent_samples": recent_samples,
            "recent": self.records()[-3:],
        }

    def _find(self, hold_id: str) -> _HoldBatch:
        for batch in self._batches:
            if batch.hold_id == hold_id:
                return batch
        raise SequenceError(f"未知的滞留批次: {hold_id}", component="lehr")

    def _excursions(self, samples: list[TempSample], until: float) -> dict:
        """按阶梯保持积分滞留期间的越限时长：样本值一直保持到下一个样本。"""

        low_s = 0.0
        high_s = 0.0
        min_c: float | None = None
        max_c: float | None = None
        ordered = sorted(samples, key=lambda sample: sample.at)
        for index, sample in enumerate(ordered):
            end = ordered[index + 1].at if index + 1 < len(ordered) else until
            span = max(0.0, end - sample.at)
            temp = sample.temperature_c
            min_c = temp if min_c is None else min(min_c, temp)
            max_c = temp if max_c is None else max(max_c, temp)
            if self._limits.band_ok(temp):
                continue
            if temp < self._limits.temp_low_c:
                low_s += span
            else:
                high_s += span
        return {"low_s": low_s, "high_s": high_s, "min_c": min_c, "max_c": max_c}

    def _decide(
        self, batch: _HoldBatch, samples: list[TempSample], ended_at: float
    ) -> DispositionRecord:
        """按滞留时长与窑温履历给出结论；依据不足的一律待判挂起。"""

        limits = self._limits
        hold_s = ended_at - batch.started_at
        stats = self._excursions(samples, ended_at)
        excursion_s = stats["low_s"] + stats["high_s"]
        span_s = samples[-1].at - samples[0].at if len(samples) >= 2 else 0.0
        coverage = span_s / hold_s if hold_s > 0 else 0.0
        evidence = {
            "hold_reason": batch.reason,
            "hold_seconds": round(hold_s, 3),
            "samples": len(samples),
            "coverage": round(coverage, 4),
            "min_c": round(stats["min_c"], 3) if stats["min_c"] is not None else None,
            "max_c": round(stats["max_c"], 3) if stats["max_c"] is not None else None,
            "low_seconds": round(stats["low_s"], 3),
            "high_seconds": round(stats["high_s"], 3),
            "excursion_seconds": round(excursion_s, 3),
            "thresholds": {
                "hold_max_s": limits.hold_max_s,
                "temp_low_c": limits.temp_low_c,
                "temp_high_c": limits.temp_high_c,
                "downgrade_excursion_s": limits.downgrade_excursion_s,
                "scrap_excursion_s": limits.scrap_excursion_s,
                "min_samples": limits.min_samples,
                "min_coverage": limits.min_coverage,
            },
        }
        decided_at = self._clock.now()

        def record(verdict: str, reason: str) -> DispositionRecord:
            return DispositionRecord(
                hold_id=batch.hold_id,
                verdict=verdict,
                reason=reason,
                evidence=evidence,
                decided_at=decided_at,
            )

        if hold_s <= 0:
            return record(
                VERDICT_PENDING,
                "滞留时长无效（排堵时间不晚于堵塞登记时间），无法判定，待人工核对后重判",
            )
        if len(samples) < limits.min_samples:
            return record(
                VERDICT_PENDING,
                f"窑温履历样本不足：{len(samples)} 个，少于判定所需 "
                f"{limits.min_samples} 个，待履历补齐后重判",
            )
        if coverage < limits.min_coverage:
            return record(
                VERDICT_PENDING,
                f"窑温履历覆盖率不足：{coverage:.1%}，低于判定要求 "
                f"{limits.min_coverage:.0%}，待履历补齐后重判",
            )
        if hold_s > limits.hold_max_s:
            return record(
                VERDICT_SCRAP,
                f"滞留 {hold_s:.0f}s 超过上限 {limits.hold_max_s:.0f}s，"
                "在制品窑内滞留过久，判废",
            )
        if excursion_s > limits.scrap_excursion_s:
            return record(
                VERDICT_SCRAP,
                f"滞留期间窑温越限累计 {excursion_s:.0f}s（低于下限 "
                f"{stats['low_s']:.0f}s、高于上限 {stats['high_s']:.0f}s）"
                f"超过判废阈 {limits.scrap_excursion_s:.0f}s，判废",
            )
        if excursion_s > limits.downgrade_excursion_s:
            return record(
                VERDICT_DOWNGRADE,
                f"滞留期间窑温越限累计 {excursion_s:.0f}s（低于下限 "
                f"{stats['low_s']:.0f}s、高于上限 {stats['high_s']:.0f}s）"
                f"超过降级阈 {limits.downgrade_excursion_s:.0f}s，降级处理",
            )
        return record(
            VERDICT_PROCEED,
            f"滞留 {hold_s:.0f}s 未超上限，窑温越限累计 {excursion_s:.0f}s "
            f"低于降级阈 {limits.downgrade_excursion_s:.0f}s，继续随流下行",
        )
