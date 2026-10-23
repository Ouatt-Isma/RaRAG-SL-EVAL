"""Empty LLM answers (failed requests) are found, dropped with what was computed from them, and stop a setup; an
Ollama server on the wrong backend is refused before any worker starts."""
import json

import pytest

from rarag_eval import answers, launch


def fake_run(tmp_path, warm_empty=0, test_empty=0):
    d = tmp_path / "NQpaper__run"
    d.mkdir()
    for name in ("corpora.pkl", "manifest.json", "src_0.faiss", "rarag_online__m.json", "minimal_swap__m__x.json"):
        (d / name).write_text("{}")
    raw = [["paris"] * 5, ["i don't know"] * 5]
    for i in range(warm_empty):
        raw[0][i] = ""
    (d / "estimation__m.json").write_text(json.dumps({"raw_answers": raw}))
    bank = {f"{s}␟q{i}": {"source": s, "question": f"q{i}", "raw": "rome" if i >= test_empty else ""}
            for s in range(2) for i in range(4)}
    (d / "answers__m.json").write_text(json.dumps(bank))
    return d


def test_a_clean_run_is_left_alone(tmp_path):
    d = fake_run(tmp_path)
    rep = answers.check(str(d), "m")
    assert rep["warmup"] == (0, 10) and rep["test"] == (0, 8) and not answers.has_empty(rep)


def test_empty_test_answers_go_with_what_was_computed_from_them(tmp_path):
    d = fake_run(tmp_path, test_empty=1)
    rep = answers.check(str(d), "m")
    assert rep["test"] == (2, 8) and answers.has_empty(rep) and "''" in rep["examples"][0]
    answers.clean(str(d), "m")
    left = sorted(p.name for p in d.iterdir())
    assert left == ["answers__m.json", "corpora.pkl", "estimation__m.json", "manifest.json", "src_0.faiss"]
    assert len(json.loads((d / "answers__m.json").read_text())) == 6       # the good answers stay
    assert not answers.has_empty(answers.check(str(d), "m"))


def test_empty_warmup_answers_drop_the_warmup_file(tmp_path):
    d = fake_run(tmp_path, warm_empty=2)
    assert answers.check(str(d), "m")["warmup"] == (2, 10)
    answers.clean(str(d), "m")
    assert not (d / "estimation__m.json").exists() and (d / "corpora.pkl").exists()


def test_a_server_on_vulkan_or_answering_nothing_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "ROOT", str(tmp_path))
    (tmp_path / "logs").mkdir()
    logf = tmp_path / "logs" / "ollama.11500.log"
    replies = {"generate": {"response": "Paris"}, "ps": {"models": [{"size_vram": 17 * 2**30}]}}
    monkeypatch.setattr(launch, "http", lambda url, body=None, timeout=5: replies["generate" if "generate" in url else "ps"])
    gpu = {"index": 0, "name": "RTX 4090", "mib": 24564}
    logf.write_text('msg="selecting GPU backend for llama-server model" library=Vulkan gpu_count=1\n')
    with pytest.raises(SystemExit, match="Vulkan"):
        launch.verify(gpu, 11500, "llama3:8b-instruct-fp16", str(tmp_path / "run.log"))
    logf.write_text('msg="selecting GPU backend for llama-server model" library=CUDA gpu_count=1\n')
    launch.verify(gpu, 11500, "llama3:8b-instruct-fp16", str(tmp_path / "run.log"))
    assert "test answer 'Paris'" in (tmp_path / "run.log").read_text()
    replies["generate"] = {"response": "  "}
    with pytest.raises(SystemExit, match="with nothing"):
        launch.verify(gpu, 11500, "llama3:8b-instruct-fp16", str(tmp_path / "run.log"))
    replies["generate"], replies["ps"] = {"response": "Paris"}, {"models": [{"size_vram": 0}]}
    with pytest.raises(SystemExit, match="CPU"):
        launch.verify(gpu, 11500, "llama3:8b-instruct-fp16", str(tmp_path / "run.log"))
