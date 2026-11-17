import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

spec = importlib.util.spec_from_file_location('worker', Path(__file__).resolve().parents[1]/'backend/worker.py')
w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)

class WorkerTests(unittest.TestCase):
    def test_consent_required(self):
        with self.assertRaisesRegex(ValueError, 'consent'):
            w.validate_request({'consent': False})
    def test_missing_media_rejected(self):
        with self.assertRaisesRegex(ValueError, 'missing'):
            w.validate_request({'consent': True, 'photo': '/missing'})
    def test_video_source_validated_without_photo(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'source.mp4'; source.touch()
            audio = Path(folder)/'voice.wav'; audio.touch()
            w.validate_request({'consent': True, 'source_video': str(source), 'audio': str(audio)})
            source.unlink()
            with self.assertRaisesRegex(ValueError, 'source_video'):
                w.validate_request({'consent': True, 'source_video': str(source), 'audio': str(audio)})
    def test_atomic_status(self):
        with tempfile.TemporaryDirectory() as folder:
            job = w.Job(folder); job.status(.25, 'Preparing')
            self.assertEqual(json.loads((Path(folder)/'status.json').read_text())['progress'], .25)
            self.assertFalse((Path(folder)/'status.tmp').exists())
    def test_cancel_before_process(self):
        with tempfile.TemporaryDirectory() as folder:
            job = w.Job(folder); (Path(folder)/'cancel').touch()
            with self.assertRaises(w.Cancelled): job.run([sys.executable, '-c', 'raise Exception("must not run")'])
    def test_child_failure_propagates(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError, 'code 7'):
                w.Job(folder).run([sys.executable, '-c', 'raise SystemExit(7)'])
    def test_paths_with_spaces_and_shell_symbols(self):
        with tempfile.TemporaryDirectory(prefix='avatar ; $ test ') as folder:
            target = Path(folder)/'value.txt'
            w.Job(folder).run([sys.executable, '-c', 'import pathlib,sys; pathlib.Path(sys.argv[1]).write_text("ok")', target])
            self.assertEqual(target.read_text(), 'ok')
    def test_running_child_cancelled(self):
        import threading
        with tempfile.TemporaryDirectory() as folder:
            job = w.Job(folder)
            timer = threading.Timer(.2, lambda: (Path(folder)/'cancel').touch()); timer.start()
            started = time.monotonic()
            with self.assertRaises(w.Cancelled): job.run([sys.executable, '-c', 'import time; time.sleep(30)'])
            timer.join(); self.assertLess(time.monotonic()-started, 6); self.assertEqual(job.children, [])
    def test_cancellation_stops_encoder_descendant(self):
        import threading
        with tempfile.TemporaryDirectory() as folder:
            marker = Path(folder)/'orphan-ran.txt'
            child = 'import time,pathlib,sys; time.sleep(1); pathlib.Path(sys.argv[1]).write_text("orphan")'
            parent = 'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",sys.argv[1],sys.argv[2]]); time.sleep(30)'
            job = w.Job(folder)
            timer = threading.Timer(.3, lambda: (Path(folder)/'cancel').touch()); timer.start()
            try:
                with self.assertRaises(w.Cancelled):
                    job.run([sys.executable, '-c', parent, child, str(marker)])
            finally:
                timer.join()
            time.sleep(1.1)
            self.assertFalse(marker.exists(), 'Encoder descendant survived cancellation')

if __name__ == '__main__': unittest.main()
