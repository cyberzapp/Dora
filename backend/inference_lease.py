"""Process-scoped coordination for memory-heavy local renderers on one Mac."""
import fcntl
import os
from pathlib import Path
import time


class InferenceLease:
    def __init__(self, path, check_cancel=lambda: None, on_wait=lambda: None):
        self.path = Path(path)
        self.check_cancel, self.on_wait = check_cancel, on_wait
        self.stream = None

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('a')
        notified = False
        try:
            while True:
                self.check_cancel()
                try:
                    fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return self
                except BlockingIOError:
                    if not notified:
                        self.on_wait(); notified = True
                    time.sleep(.25)
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.stream is not None:
            self.stream.close(); self.stream = None

    def __enter__(self): return self.acquire()
    def __exit__(self, *args): self.close()


class GPUInferenceLease(InferenceLease):
    """One heavy renderer across source checkouts and installed runtime copies."""
    def __init__(self,check_cancel=lambda:None,on_wait=lambda:None):
        directory=Path('/private/tmp')/f'avatar-studio-{os.getuid()}'
        directory.mkdir(mode=0o700,exist_ok=True)
        if directory.is_symlink() or directory.stat().st_uid!=os.getuid():
            raise ValueError('The local rendering lock directory has an unexpected owner')
        super().__init__(directory/'gpu.lock',check_cancel,on_wait)
