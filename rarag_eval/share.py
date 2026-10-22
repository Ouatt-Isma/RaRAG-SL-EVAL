"""What workers on one machine share (one worker per GPU, see launch.py): a file lock, and the queue of the
grid's setups. A setup is claimed by creating its claim file, which only one worker can do."""
import contextlib
import fcntl
import os
import socket


@contextlib.contextmanager
def locked(path):
    """Hold an exclusive lock on path + ".lock" (other workers wait)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".lock", "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def claims_dir(root=None):
    from . import store   # (thesis.py imports this module; store imports thesis)
    return os.path.join(root or store.RESULTS, ".claims")


def claim(dataset, seed, setup_id, root=None):
    """True for the one worker that takes this setup; False for every other."""
    from . import store
    d = claims_dir(root)
    os.makedirs(d, exist_ok=True)
    try:
        fd = os.open(os.path.join(d, f"{store.dataset_name(dataset)}__seed{seed}__{setup_id}"),
                     os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        f.write(f"{socket.gethostname()} pid {os.getpid()} gpu {os.environ.get('CUDA_VISIBLE_DEVICES', '-')}\n")
    return True


def clear_claims(root=None):
    """Before the workers start: a claim left by a stopped run must not hold its setup back."""
    d = claims_dir(root)
    if os.path.isdir(d):
        for name in os.listdir(d):
            os.remove(os.path.join(d, name))
