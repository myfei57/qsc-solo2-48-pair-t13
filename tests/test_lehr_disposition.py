"""滞留处置：按滞留时长与窑内温度履历给出继续／降级／判废，依据不足待判。"""

from __future__ import annotations

import copy
import json
import os
import urllib.request

import pytest

from glassmelt.app import CycleInputs
from glassmelt.config import (
    PlatformConfig,
    config_to_mapping,
    default_config,
    load_config,
)
from glassmelt.console.server import ConsoleServer
from glassmelt.errors import ConfigError
from glassmelt.lehr.disposition import (
    CONTINUE,
    DOWNGRADE,
    PENDING,
    SCRAP,
    VERDICT_LABELS,
    DispositionBoard,
)

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "glassmelt.yaml",
)

NOW = 10_000.0


def _latch(ts: float, epoch: int, released: bool = False) -> dict:
    return {
        "ts": ts,
        "action": "anneal_latch_released" if released else "anneal_latch_set",
        "detail": {"epoch": epoch, "reason": "测试"},
    }


def _zones(ts: float, mean_c: float) -> dict:
    return {
        "ts": ts,
        "action": "anneal_zones_applied",
        "detail": {"mean_c": mean_c},
    }


def _noise(ts: float) -> dict:
    return {"ts": ts, "action": "belt_speed_settled", "detail": {"speed": 4.2}}


@pytest.fixture()
def board() -> DispositionBoard:
    return DispositionBoard(default_config().disposition)


def _episode(report: dict, index: int = 0) -> dict:
    return report["episodes"][index]


def test_short_hold_with_steady_temperature_continues(board):
    records = [
        _latch(100.0, 3),
        _zones(110.0, 560.0),
        _zones(240.0, 558.0),
        _latch(250.0, 4, released=True),
    ]
    report = board.evaluate(records, NOW)
    episode = _episode(report)
    assert episode["episode_id"] == "holdup-3"
    assert episode["verdict"] == CONTINUE
    assert episode["verdict_label"] == "继续"
    assert episode["metrics"]["duration_s"] == pytest.approx(150.0)
    assert episode["metrics"]["samples"] == 2
    assert episode["metrics"]["coverage"] == pytest.approx(1.0)
    assert any("低于降级线" in line for line in episode["evidence"])
    assert any("履历摘要" in line for line in episode["evidence"])


def test_hold_beyond_downgrade_line_is_downgraded(board):
    records = [
        _latch(100.0, 1),
        _zones(120.0, 560.0),
        _zones(480.0, 561.0),
        _latch(500.0, 2, released=True),
    ]
    episode = _episode(board.evaluate(records, NOW))
    assert episode["verdict"] == DOWNGRADE
    assert episode["verdict_label"] == "降级"
    assert any("滞留时长" in line and "降级线" in line for line in episode["evidence"])


def test_hold_beyond_scrap_line_is_scrapped(board):
    records = [
        _latch(100.0, 1),
        _zones(200.0, 560.0),
        _zones(1300.0, 560.0),
        _latch(1400.0, 2, released=True),
    ]
    episode = _episode(board.evaluate(records, NOW))
    assert episode["verdict"] == SCRAP
    assert episode["verdict_label"] == "判废"
    assert any("滞留时长" in line and "判废线" in line for line in episode["evidence"])


def test_overcold_kiln_during_hold_is_scrapped(board):
    records = [
        _latch(100.0, 1),
        _zones(110.0, 470.0),
        _zones(240.0, 465.0),
        _latch(250.0, 2, released=True),
    ]
    episode = _episode(board.evaluate(records, NOW))
    assert episode["verdict"] == SCRAP
    assert any("过冷判废线" in line for line in episode["evidence"])


def test_cool_kiln_above_scrap_line_is_downgraded(board):
    records = [
        _latch(100.0, 1),
        _zones(110.0, 500.0),
        _zones(240.0, 505.0),
        _latch(250.0, 2, released=True),
    ]
    episode = _episode(board.evaluate(records, NOW))
    assert episode["verdict"] == DOWNGRADE
    assert any("低温降级线" in line for line in episode["evidence"])


def test_overheated_kiln_during_hold_is_scrapped(board):
    records = [
        _latch(100.0, 1),
        _zones(110.0, 605.0),
        _zones(240.0, 608.0),
        _latch(250.0, 2, released=True),
    ]
    episode = _episode(board.evaluate(records, NOW))
    assert episode["verdict"] == SCRAP
    assert any("过热判废线" in line for line in episode["evidence"])


def test_hot_kiln_below_scrap_line_is_downgraded(board):
    records = [
        _latch(100.0, 1),
        _zones(110.0, 580.0),
        _zones(240.0, 582.0),
        _latch(250.0, 2, released=True),
    ]
    episode = _episode(board.evaluate(records, NOW))
    assert episode["verdict"] == DOWNGRADE
    assert any("高温降级线" in line for line in episode["evidence"])


def test_open_window_is_held_pending(board):
    records = [_latch(100.0, 1), _zones(150.0, 560.0), _zones(200.0, 560.0)]
    episode = _episode(board.evaluate(records, NOW))
    assert episode["verdict"] == PENDING
    assert episode["verdict_label"] == "待判"
    assert episode["released_at"] is None
    assert any("滞留尚未结束" in line for line in episode["evidence"])


def test_window_without_release_record_is_held_pending(board):
    records = [
        _latch(100.0, 1),
        _zones(150.0, 560.0),
        _latch(200.0, 3),
        _zones(210.0, 560.0),
        _zones(340.0, 560.0),
        _latch(350.0, 4, released=True),
    ]
    report = board.evaluate(records, NOW)
    assert report["summary"]["total"] == 2
    first, second = report["episodes"]
    assert first["verdict"] == PENDING
    assert any("缺少闩锁释放事件" in line for line in first["evidence"])
    assert second["verdict"] == CONTINUE


def test_window_with_too_few_samples_is_held_pending(board):
    records = [_latch(100.0, 1), _zones(150.0, 560.0), _latch(200.0, 2, released=True)]
    episode = _episode(board.evaluate(records, NOW))
    assert episode["verdict"] == PENDING
    assert any("采样点" in line for line in episode["evidence"])


def test_window_with_thin_temperature_coverage_is_held_pending(board):
    records = [
        _latch(100.0, 1),
        _zones(110.0, 560.0),
        _zones(120.0, 560.0),
        _latch(400.0, 2, released=True),
    ]
    episode = _episode(board.evaluate(records, NOW))
    assert episode["verdict"] == PENDING
    assert any("覆盖率" in line for line in episode["evidence"])


def test_samples_outside_the_window_are_ignored(board):
    records = [
        _zones(10.0, 60.0),
        _latch(100.0, 1),
        _zones(110.0, 560.0),
        _zones(240.0, 560.0),
        _latch(250.0, 2, released=True),
        _zones(300.0, 60.0),
        _noise(400.0),
    ]
    episode = _episode(board.evaluate(records, NOW))
    assert episode["verdict"] == CONTINUE
    assert episode["metrics"]["samples"] == 2
    assert episode["metrics"]["min_c"] == pytest.approx(560.0)


def test_multiple_episodes_are_summarized(board):
    records = [
        _latch(100.0, 1),
        _zones(110.0, 560.0),
        _zones(240.0, 560.0),
        _latch(250.0, 2, released=True),
        _latch(300.0, 3),
        _zones(310.0, 450.0),
        _zones(440.0, 445.0),
        _latch(450.0, 4, released=True),
        _latch(500.0, 5),
    ]
    report = board.evaluate(records, NOW)
    summary = report["summary"]
    assert summary["total"] == 3
    assert summary[CONTINUE] == 1
    assert summary[SCRAP] == 1
    assert summary[PENDING] == 1
    assert report["records_scanned"] == len(records)
    assert report["limits"]["hold_scrap_s"] == pytest.approx(1200.0)
    labels = {episode["verdict_label"] for episode in report["episodes"]}
    assert labels == {"继续", "判废", "待判"}
    assert set(VERDICT_LABELS) == {CONTINUE, DOWNGRADE, SCRAP, PENDING}


def test_empty_records_yield_an_empty_report(board):
    report = board.evaluate([], NOW)
    assert report["summary"]["total"] == 0
    assert report["episodes"] == []


def _warm_into_soak(running) -> None:
    """把工艺时钟推进恒温段并让段温收敛到退火目标附近。"""

    running.cycle(CycleInputs(load=0.8, pull_rate=1.0, gob_mass_g=186.0, dt=350.0))
    for _ in range(12):
        running.cycle(CycleInputs(load=0.8, pull_rate=1.0, gob_mass_g=186.0, dt=1.0))


def _hold_and_release(running, clock, hold_s: float, observed_c: float) -> None:
    """制造一段滞留：闩锁置位 → 窗口内留温度履历 → 复位释放。"""

    running.evaluate_anneal(running.config.temp.anneal_alarm_c + 8.0)
    half = hold_s / 2.0
    clock.advance(half)
    running.evaluate_anneal(observed_c)
    clock.advance(half)
    running.evaluate_anneal(observed_c)
    running.reset_lehr("排堵完成")


def test_platform_report_concludes_continue_for_a_mild_hold(running, clock):
    _warm_into_soak(running)
    _hold_and_release(running, clock, hold_s=240.0, observed_c=580.0)
    report = running.disposition_report()
    assert report["summary"]["total"] == 1
    episode = _episode(report)
    assert episode["verdict"] == CONTINUE
    assert episode["metrics"]["duration_s"] == pytest.approx(240.0)
    assert episode["metrics"]["samples"] >= 2
    assert episode["evidence"]


def test_platform_report_scraps_a_hold_with_cold_kiln(running, clock):
    _hold_and_release(running, clock, hold_s=200.0, observed_c=580.0)
    episode = _episode(running.disposition_report())
    assert episode["verdict"] == SCRAP
    assert episode["metrics"]["min_c"] <= running.config.disposition.low_scrap_c
    assert any("过冷判废线" in line for line in episode["evidence"])


def test_platform_report_downgrades_a_long_hold(running, clock):
    _warm_into_soak(running)
    _hold_and_release(running, clock, hold_s=400.0, observed_c=580.0)
    episode = _episode(running.disposition_report())
    assert episode["verdict"] == DOWNGRADE
    assert episode["metrics"]["duration_s"] == pytest.approx(400.0)


def test_platform_report_holds_an_open_window_pending(running, clock):
    running.evaluate_anneal(running.config.temp.anneal_alarm_c + 8.0)
    clock.advance(60.0)
    running.evaluate_anneal(580.0)
    report = running.disposition_report()
    episode = _episode(report)
    assert episode["verdict"] == PENDING
    assert episode["released_at"] is None
    assert any("滞留尚未结束" in line for line in episode["evidence"])


def test_platform_report_holds_a_window_without_history_pending(running):
    running.evaluate_anneal(running.config.temp.anneal_alarm_c + 8.0)
    running.reset_lehr("排堵完成")
    episode = _episode(running.disposition_report())
    assert episode["verdict"] == PENDING
    assert any("采样点" in line for line in episode["evidence"])


def test_disposition_endpoint_serves_the_report(running):
    server = ConsoleServer(running)
    base = server.start()
    try:
        with urllib.request.urlopen(f"{base}/api/lehr/disposition", timeout=5.0) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        assert "summary" in payload and "episodes" in payload
        with urllib.request.urlopen(f"{base}/api/state", timeout=5.0) as resp:
            state = json.loads(resp.read().decode("utf-8"))
        assert "/api/lehr/disposition" in state["endpoints"]["api"]
    finally:
        server.stop()


def test_disposition_limits_load_from_yaml_and_validate():
    loaded = load_config(CONFIG_PATH)
    assert loaded.disposition == default_config().disposition
    mapping = config_to_mapping(default_config())
    assert PlatformConfig.from_mapping(mapping).disposition == default_config().disposition
    broken = copy.deepcopy(mapping)
    broken["limits"]["disposition"]["hold_scrap_s"] = 100.0
    with pytest.raises(ConfigError):
        PlatformConfig.from_mapping(broken)
    thin = copy.deepcopy(mapping)
    thin["limits"]["disposition"]["min_coverage"] = 1.5
    with pytest.raises(ConfigError):
        PlatformConfig.from_mapping(thin)
