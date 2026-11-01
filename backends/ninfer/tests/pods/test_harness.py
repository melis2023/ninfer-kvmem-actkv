import subprocess

import pytest

from scripts.pods import harness


@pytest.mark.parametrize("output", ["starting instance 123.\n", '{"success": true}', "{'success': True}"])
def test_start_accepts_cli_or_json_acknowledgment(monkeypatch, output):
    monkeypatch.setattr(harness.subprocess, "run", lambda *a, **kw:
                        subprocess.CompletedProcess(a, 0, output, ""))
    harness.vast("start", "instance", 123)


@pytest.mark.parametrize("output", ["", "Insufficient available resources",
                                    '{"success": false, "msg": "resources unavailable"}'])
def test_start_refuses_exit_zero_without_success(monkeypatch, output):
    monkeypatch.setattr(harness.subprocess, "run", lambda *a, **kw:
                        subprocess.CompletedProcess(a, 0, output, ""))
    with pytest.raises(RuntimeError, match="start rejected"):
        harness.vast("start", "instance", 123)


def test_rejection_does_not_echo_provider_secrets():
    assert str(harness.provider_rejection("start", "resources unavailable; token=do-not-copy")) == \
        "vastai start rejected: resources unavailable"


@pytest.mark.parametrize("action", ["create", "start", "stop", "destroy"])
def test_stderr_http_error_is_not_a_successful_mutation(monkeypatch, action):
    diagnostic = '{"error": true, "status_code": 402, "msg": "insufficient credit; token=do-not-copy"}'
    monkeypatch.setattr(harness.subprocess, "run", lambda *a, **kw:
                        subprocess.CompletedProcess(a, 0, "", diagnostic))
    with pytest.raises(RuntimeError, match=rf"^vastai {action} rejected \(HTTP 402\): billing or balance$"):
        harness.vast(action, "instance", 123)


@pytest.mark.parametrize("now", [5, 15])
def test_obsolete_guard_cannot_stop_a_rearmed_stage(monkeypatch, now):
    monkeypatch.setattr(harness.sys, "argv", ["harness", "_guard", "123", "10"])
    monkeypatch.setattr(harness.time, "time", lambda: now)
    monkeypatch.setattr(harness, "state", lambda: {"id": 123, "deadline": 20})
    monkeypatch.setattr(harness, "vast", lambda *a: pytest.fail("obsolete guard called provider"))
    monkeypatch.setattr(harness.subprocess, "run", lambda *a, **kw: pytest.fail("a newer arm keeps the job"))
    harness.main()


@pytest.mark.parametrize("record", [{"id": 123, "deadline": 10, "destroyed": True},
                                    {"id": 456, "deadline": 10}])
def test_guard_of_an_ended_rental_removes_its_launchd_job(monkeypatch, record):
    commands = []
    monkeypatch.setattr(harness.sys, "argv", ["harness", "_guard", "123", "10"])
    monkeypatch.setattr(harness.sys, "platform", "darwin")
    monkeypatch.setattr(harness.time, "time", lambda: 5)
    monkeypatch.setattr(harness, "state", lambda: record)
    monkeypatch.setattr(harness, "vast", lambda *a: pytest.fail("ended rental called provider"))
    monkeypatch.setattr(harness.subprocess, "run", lambda args, **kw: commands.append(args))
    harness.main()
    assert commands == [["launchctl", "remove", "ninfer-rental-123"]]


def test_guard_without_its_state_file_still_stops_the_rental(monkeypatch):
    rental = {"id": 123, "actual_status": "running", "intended_status": "running"}
    stopped, commands = [], []
    monkeypatch.setattr(harness.sys, "argv", ["harness", "_guard", "123", "10"])
    monkeypatch.setattr(harness.sys, "platform", "darwin")
    monkeypatch.setattr(harness.time, "time", lambda: 15)
    monkeypatch.setattr(harness.time, "sleep", lambda _: None)
    monkeypatch.setattr(harness, "STATE", harness.Path("/nonexistent/pod.json"))
    monkeypatch.setattr(harness.subprocess, "run", lambda args, **kw: commands.append(args))

    def provider(*args):
        if args == ("show", "instances"):
            return [rental.copy()]
        assert args == ("stop", "instance", 123)
        stopped.append(args)
        rental.update(actual_status="exited", intended_status="stopped")
        return {}

    monkeypatch.setattr(harness, "vast", provider)
    harness.main()
    assert len(stopped) == 1
    assert commands == [["launchctl", "remove", "ninfer-rental-123"]]


def test_guard_carries_its_selected_rental_record(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(harness, "STATE", tmp_path / "second" / "pod.json")
    monkeypatch.setattr(harness.sys, "platform", "darwin")
    monkeypatch.setattr(harness.subprocess, "run", lambda args, **kw: commands.append(args))
    harness.arm_guard({"provider": "vast", "id": 123}, 0.5)
    command = next(args for args in commands if args[:2] == ["launchctl", "submit"])
    assert f"NINFER_POD_STATE={harness.STATE}" in command


@pytest.mark.parametrize("actual,intended,expected", [
    ("exited", "stopped", True), ("stopped", "stopped", True),
    ("exited", "running", False), ("running", "stopped", False),
    ("exited", None, False), (None, "stopped", False),
])
def test_stop_requires_both_terminal_compute_and_canceled_start(actual, intended, expected):
    assert harness.rental_stopped(actual, intended) is expected


@pytest.mark.parametrize("entry", ["stop", "_guard"])
def test_explicit_stop_and_guard_cancel_a_queued_start(monkeypatch, entry):
    pod = {"provider": "vast", "id": 123, "deadline": 10,
           "actual_status": "exited", "intended_status": "running"}
    stopped = []
    monkeypatch.setattr(harness.sys, "argv", ["harness", entry] +
                        (["123", "10"] if entry == "_guard" else []))
    monkeypatch.setattr(harness.time, "time", lambda: 15)
    monkeypatch.setattr(harness.time, "sleep", lambda _: None)
    monkeypatch.setattr(harness, "state", lambda: pod.copy())
    monkeypatch.setattr(harness, "endpoint", lambda p:
                        {**p, "status": pod["actual_status"], "intended_status": pod["intended_status"]})
    monkeypatch.setattr(harness, "ssh", lambda *a: pytest.fail("queued start needs no SSH"))

    def provider(*args):
        if args == ("show", "instances"):
            return [pod.copy()]
        assert args == ("stop", "instance", 123)
        stopped.append(args)
        pod["intended_status"] = "stopped"
        return {}

    monkeypatch.setattr(harness, "vast", provider)
    harness.main()
    assert len(stopped) == 1


CATALOG = [
    {"gpuId": "NVIDIA GeForce RTX 3090", "displayName": "RTX 3090", "communityCloud": True,
     "communityPricePerHr": 0.22, "securePricePerHr": 0.5, "available": True},
    {"gpuId": "NVIDIA GeForce RTX 3090 Ti", "displayName": "RTX 3090 Ti", "communityCloud": True,
     "communityPricePerHr": 0.27, "securePricePerHr": None, "available": False},
    {"gpuId": "NVIDIA L4", "displayName": "L4", "communityCloud": False,
     "communityPricePerHr": None, "securePricePerHr": 0.59, "available": True},
]


def test_download_cap_joins_only_a_query_without_one():
    assert harness.with_download_cap("gpu_name=RTX_3060", 0.005) == \
        "gpu_name=RTX_3060 inet_down_cost<=0.005"
    assert harness.with_download_cap("inet_down_cost<0.01", 0.005) == "inet_down_cost<0.01"


def test_vast_cost_counts_traffic_beside_hours():
    # The 2026-10-14 RTX 3060 Ti host: 80.8 GB at $0.039/GB outweighed its GPU time.
    offer = {"dph_total": 0.1, "inet_down_cost": 0.039, "inet_up_cost": 0.039}
    assert harness.vast_cost(offer, 2, 80.8, 1) == pytest.approx(0.2 + 80.8 * 0.039 + 0.039)


@pytest.mark.parametrize("name", ["RTX 3090", "rtx 3090", "NVIDIA GeForce RTX 3090"])
def test_runpod_quote_names_the_same_card(name):
    quote = harness.runpod_quote(CATALOG, name, 2, 10, 100)
    assert quote["gpu"] == "NVIDIA GeForce RTX 3090" and quote["cloud"] == "community"
    assert quote["cost"] == pytest.approx((0.44 + 100 * 0.10 / 730) * 10, abs=1e-3)
    assert "--gpus 2 --disk 100" in quote["command"] and "--secure" not in quote["command"]


def test_runpod_quote_takes_the_secure_price_without_a_community_one():
    quote = harness.runpod_quote(CATALOG, "L4", 1, 1, 0)
    assert quote["cloud"] == "secure" and quote["hourly"] == 0.59
    assert quote["command"].endswith(" --secure")


@pytest.mark.parametrize("catalog", [CATALOG, None])
def test_runpod_quote_without_the_card(catalog):
    assert harness.runpod_quote(catalog, "RTX 3060", 1, 1, 0) is None


@pytest.mark.parametrize("vast_total,available,expected", [
    (1.0, True, "runpod"), (0.5, True, "runpod"), (0.4, True, "vast"),
    (1.0, False, "vast"), (None, True, "runpod"), (None, False, None),
])
def test_runpod_is_preferred_unless_vast_costs_less(vast_total, available, expected):
    assert harness.preferred(vast_total, {"cost": 0.5, "available": available}) == expected


def rent_with(monkeypatch, tmp_path, offer, *options):
    class Created(Exception):
        pass

    def provider(*args):
        if args == ("show", "instances"):
            return []
        if args[:2] == ("search", "offers"):
            assert args[2] == "ask_contract_id=7 verified=any"
            return [offer]
        raise Created(args)

    key = tmp_path / "id_test"
    key.with_suffix(".pub").write_text("ssh-ed25519 AAAAtest test\n")
    monkeypatch.setattr(harness, "STATE", tmp_path / "pod.json")
    monkeypatch.setattr(harness, "SSH_KEY", key)
    monkeypatch.setattr(harness, "vast", provider)
    monkeypatch.setattr(harness, "runpod_catalog", lambda: CATALOG)
    monkeypatch.setattr(harness.sys, "argv", ["harness", "rent", "7", "--hours", "2",
                                              "--image", "image", *options])
    with pytest.raises((ValueError, Created)) as raised:
        harness.main()
    return raised.value


def test_rent_refuses_an_offer_over_the_download_price(monkeypatch, tmp_path):
    offer = {"gpu_name": "RTX 3060", "num_gpus": 1, "dph_total": 0.1, "inet_down_cost": 0.039}
    error = rent_with(monkeypatch, tmp_path, offer, "--download-gb", "80")
    assert "bills $0.0390/GB of downloads" in str(error)


def test_rent_sends_a_card_runpod_quotes_for_less_to_runpod(monkeypatch, tmp_path):
    offer = {"gpu_name": "RTX 3090", "num_gpus": 1, "dph_total": 0.3, "inet_down_cost": 0.004}
    # $0.3/h for 2 h and 100 GB at $0.004/GB against RunPod's $0.22/h and container disk.
    error = rent_with(monkeypatch, tmp_path, offer, "--download-gb", "100", "--disk", "100")
    assert isinstance(error, ValueError) and "pod-up.sh --gpu 'NVIDIA GeForce RTX 3090'" in str(error)
    error = rent_with(monkeypatch, tmp_path, offer, "--download-gb", "100", "--disk", "100",
                      "--over-runpod")
    assert type(error).__name__ == "Created"
