"""出口堵塞滞留在制品的处置判定：按滞留时长与窑温履历给结论。"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from glassmelt.config import PlatformConfig, config_to_mapping, default_config
from glassmelt.console.server import ConsoleServer
from glassmelt.errors import ConfigError, InterlockError


def _apply_anneal(platform, clock, setpoint_c, repeats=1, step_s=0.0):
    """按给定温度落退火段温，让处置台经退火记录攒到窑温履历样本。"""

    zones = platform.config.lehr.zone_names()
    for _ in range(repeats):
        platform.anneal.apply({zone: setpoint_c for zone in zones}, clock.now())
        if step_s:
            clock.advance(step_s)


def _get(url: str) -> tuple[int, object]:
    try:
        with urllib.request.urlopen(url, timeout=5.0) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body) if body.startswith("{") else body
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


def _post(url: str, payload: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(payload or {}).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=5.0) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


@pytest.fixture()
def console(platform):
    server = ConsoleServer(platform)
    base = server.start()
    yield server, base
    server.stop()


def test_normal_history_proceeds(platform, clock):
    _apply_anneal(platform, clock, 560.0, repeats=6, step_s=1.0)
    noted = platform.note_lehr_hold("出口堵塞")
    _apply_anneal(platform, clock, 560.0, repeats=4, step_s=10.0)
    record = platform.clear_lehr_hold("排堵结束")
    assert record["hold_id"] == noted["hold_id"]
    assert record["verdict"] == "proceed"
    assert record["evidence"]["excursion_seconds"] == pytest.approx(0.0)
    assert record["evidence"]["samples"] == 4
    assert record["evidence"]["hold_seconds"] == pytest.approx(40.0)
    assert "继续" in record["reason"]


def test_low_temp_excursion_downgrades(platform, clock):
    _apply_anneal(platform, clock, 560.0, repeats=6)
    platform.note_lehr_hold("出口堵塞")
    _apply_anneal(platform, clock, 450.0, repeats=8, step_s=30.0)
    record = platform.clear_lehr_hold("排堵结束")
    assert record["verdict"] == "downgrade"
    assert record["evidence"]["low_seconds"] == pytest.approx(210.0)
    assert record["evidence"]["high_seconds"] == pytest.approx(0.0)
    assert record["evidence"]["min_c"] < platform.config.disposition.temp_low_c
    assert "降级" in record["reason"]


def test_sustained_excursion_scraps(platform, clock):
    _apply_anneal(platform, clock, 560.0, repeats=6)
    platform.note_lehr_hold("出口堵塞")
    _apply_anneal(platform, clock, 450.0, repeats=8, step_s=130.0)
    record = platform.clear_lehr_hold("排堵结束")
    assert record["verdict"] == "scrap"
    assert record["evidence"]["excursion_seconds"] == pytest.approx(910.0)
    assert record["evidence"]["hold_seconds"] < platform.config.disposition.hold_max_s
    assert "判废" in record["reason"]


def test_overlong_hold_scraps_even_with_normal_temps(platform, clock):
    _apply_anneal(platform, clock, 560.0, repeats=6)
    platform.note_lehr_hold("出口堵塞")
    _apply_anneal(platform, clock, 560.0, repeats=4, step_s=1000.0)
    record = platform.clear_lehr_hold("排堵结束")
    assert record["verdict"] == "scrap"
    assert record["evidence"]["excursion_seconds"] == pytest.approx(0.0)
    assert record["evidence"]["hold_seconds"] > platform.config.disposition.hold_max_s
    assert "滞留" in record["reason"]


def test_missing_history_is_held_as_pending(platform, clock):
    platform.note_lehr_hold("出口堵塞")
    clock.advance(60.0)
    record = platform.clear_lehr_hold("排堵结束")
    assert record["verdict"] == "pending"
    assert record["evidence"]["samples"] == 0
    assert "样本不足" in record["reason"]


def test_sparse_history_is_held_as_pending(platform, clock):
    platform.note_lehr_hold("出口堵塞")
    _apply_anneal(platform, clock, 560.0, repeats=3, step_s=2.0)
    clock.advance(200.0)
    record = platform.clear_lehr_hold("排堵结束")
    assert record["verdict"] == "pending"
    assert record["evidence"]["samples"] == 3
    assert record["evidence"]["coverage"] < platform.config.disposition.min_coverage
    assert "覆盖率不足" in record["reason"]


def test_pending_batch_can_be_rejudged_with_supplied_history(platform, clock):
    platform.note_lehr_hold("出口堵塞")
    started_at = platform.clock.now()
    clock.advance(250.0)
    record = platform.clear_lehr_hold("排堵结束")
    assert record["verdict"] == "pending"
    samples = [
        {"at": started_at + 30.0 * index, "temperature_c": 560.0}
        for index in range(1, 8)
    ]
    rejudged = platform.rejudge_lehr_hold(record["hold_id"], samples)
    assert rejudged["verdict"] == "proceed"
    assert rejudged["evidence"]["samples"] == 7


def test_final_verdict_is_not_overturned(platform, clock):
    _apply_anneal(platform, clock, 560.0, repeats=6)
    noted = platform.note_lehr_hold("出口堵塞")
    _apply_anneal(platform, clock, 560.0, repeats=4, step_s=10.0)
    record = platform.clear_lehr_hold("排堵结束")
    assert record["verdict"] == "proceed"
    hostile = [
        {"at": record["decided_at"] - 30.0 + index, "temperature_c": 200.0}
        for index in range(5)
    ]
    again = platform.rejudge_lehr_hold(noted["hold_id"], hostile)
    assert again["verdict"] == "proceed"
    decided = [
        entry
        for entry in platform.audit.entries(limit=500)
        if entry["action"] == "lehr_disposition_decided"
        and entry["detail"].get("hold_id") == noted["hold_id"]
    ]
    assert len(decided) == 1


def test_hold_registration_is_singleton(platform):
    platform.note_lehr_hold("出口堵塞")
    with pytest.raises(InterlockError):
        platform.note_lehr_hold("重复登记")
    with pytest.raises(InterlockError):
        platform.rejudge_lehr_hold("hold-1")
    platform.clear_lehr_hold("排堵结束")
    with pytest.raises(InterlockError):
        platform.clear_lehr_hold("没有堵塞可排")


def test_disposition_is_audited_by_severity(platform, clock):
    _apply_anneal(platform, clock, 560.0, repeats=6)
    platform.note_lehr_hold("出口堵塞")
    _apply_anneal(platform, clock, 450.0, repeats=8, step_s=130.0)
    platform.clear_lehr_hold("排堵结束")
    alarms = platform.audit.alarms()
    assert any(
        entry["subject"] == "lehr_disposition"
        and entry["action"] == "lehr_disposition_decided"
        and entry["severity"] == "warning"
        for entry in alarms
    )
    assert any(
        entry["action"] == "lehr_hold_noted" for entry in alarms
    )


def test_status_and_report_expose_disposition(platform, clock):
    platform.note_lehr_hold("出口堵塞")
    clock.advance(30.0)
    platform.clear_lehr_hold("排堵结束")
    status = platform.status()
    disposition = status["anneal"]["disposition"]
    assert disposition["batches"] == 1
    assert disposition["by_verdict"]["pending"] == 1
    report = platform.disposition_report()
    assert report["records"][0]["verdict"] == "pending"
    assert report["snapshot"]["open_hold"] is None


def test_disposition_limits_are_validated():
    mapping = config_to_mapping(default_config())
    mapping["limits"]["disposition"]["temp_low_c"] = 700.0
    with pytest.raises(ConfigError):
        PlatformConfig.from_mapping(mapping)
    inverted = config_to_mapping(default_config())
    inverted["limits"]["disposition"]["scrap_excursion_s"] = 60.0
    with pytest.raises(ConfigError):
        PlatformConfig.from_mapping(inverted)


def test_console_hold_and_rejudge_flow(console, platform):
    _server, base = console
    status, noted = _post(f"{base}/api/lehr/hold", {"path": "note", "reason": "接口测试堵塞"})
    assert status == 200 and noted["ok"] is True
    hold_id = noted["result"]["hold_id"]
    started_at = noted["result"]["started_at"]

    status, again = _post(f"{base}/api/lehr/hold", {"path": "note"})
    assert status == 200 and again["ok"] is False
    assert again["error"]["component"] == "lehr"

    platform.clock.advance(30.0)
    status, cleared = _post(f"{base}/api/lehr/hold", {"path": "clear"})
    assert status == 200 and cleared["ok"] is True
    assert cleared["result"]["verdict"] == "pending"
    assert "样本不足" in cleared["result"]["reason"]

    samples = [
        {"at": started_at + offset, "temperature_c": 560.0}
        for offset in (5.0, 10.0, 15.0, 20.0, 25.0)
    ]
    status, rejudged = _post(
        f"{base}/api/lehr/hold",
        {"path": "rejudge", "hold_id": hold_id, "samples": samples},
    )
    assert status == 200 and rejudged["ok"] is True
    assert rejudged["result"]["verdict"] == "proceed"

    status, report = _get(f"{base}/api/lehr/disposition")
    assert status == 200
    assert report["snapshot"]["batches"] == 1
    assert report["records"][0]["verdict"] == "proceed"

    status, bad = _post(f"{base}/api/lehr/hold", {"path": "nope"})
    assert status == 400
