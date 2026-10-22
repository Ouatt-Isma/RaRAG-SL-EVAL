"""The multi-GPU launch: reading nvidia-smi, how many workers each GPU gets, and that a setup goes to one worker."""
from multiprocessing import Pool

from rarag_eval import launch, share

SMI = """0, NVIDIA GeForce RTX 4090, 24564
1, NVIDIA GeForce RTX 4090, 24564
2, NVIDIA RTX A6000, 49140
3, NVIDIA A100-SXM4-80GB, 81920
"""


def test_parse_smi_and_visible_devices():
    gpus = launch.parse_smi(SMI)
    assert [g["index"] for g in gpus] == [0, 1, 2, 3] and gpus[0]["mib"] == 24564
    assert [g["index"] for g in launch.parse_smi(SMI, "1,3")] == [1, 3]
    assert len(launch.parse_smi(SMI, "GPU-1234")) == 4   # UUIDs: leave the list as nvidia-smi gives it
    assert launch.parse_smi("") == []


def test_workers_per_gpu_follow_memory():
    gpus = launch.parse_smi(SMI)
    per = [g["index"] for g, _ in launch.plan(gpus)]
    assert per == [0, 1, 2, 2, 3, 3, 3]                       # 24 GB: 1, 48 GB: 2, 80 GB: 3
    ports = [p for _, p in launch.plan(gpus)]
    assert len(set(ports)) == len(ports)                      # one Ollama server per worker
    assert [g["index"] for g, _ in launch.plan(gpus, n_gpus=2)] == [0, 1]
    assert [g["index"] for g, _ in launch.plan(gpus, per_gpu=1)] == [0, 1, 2, 3]
    assert launch.plan([]) == [(None, launch.FIRST_PORT)]     # no GPU: one worker (e.g. Ollama on a Mac's GPU)
    assert [g for g, _ in launch.plan([], workers=2)] == [None, None]


def _try(args):
    root, worker = args
    return [(u, worker) for u in range(20) if share.claim("NQ_paper.json", 0, f"u{u}", root=root)]


def test_each_setup_is_claimed_once(tmp_path):
    with Pool(4) as pool:
        taken = [t for part in pool.map(_try, [(str(tmp_path), w) for w in range(4)]) for t in part]
    assert sorted(u for u, _ in taken) == list(range(20))
    share.clear_claims(str(tmp_path))
    assert share.claim("NQ_paper.json", 0, "u0", root=str(tmp_path))


def test_progress_helpers():
    from rarag_eval import progress
    assert progress.only_filters(["--offline", "--only", "dataset=NQ_paper", "seed=0", "--runs-dir", "x"]) == \
        {"dataset": "NQ_paper", "seed": "0"}
    assert progress.duration(59) == "0 min" and progress.duration(3 * 3600 + 5 * 60) == "3 h 05 min"
    assert progress.sources({"kind": "mix5"}) == 5 and progress.sources({"kind": "attackers"}) == 9
