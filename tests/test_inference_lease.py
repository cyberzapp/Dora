import sys
from pathlib import Path
import subprocess
import tempfile
import unittest
import shutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'backend'))
from inference_lease import InferenceLease,GPUInferenceLease


class LeaseTests(unittest.TestCase):
    def test_owner_exit_releases_and_waiter_can_cancel(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'render.lock'
            script = 'from inference_lease import InferenceLease; import sys,time; lease=InferenceLease(sys.argv[1]).acquire(); print("ready",flush=True); time.sleep(30)'
            child = subprocess.Popen([sys.executable, '-c', script, str(path)],
                                     cwd=Path(__file__).resolve().parents[1]/'backend', stdout=subprocess.PIPE, text=True)
            try:
                self.assertEqual(child.stdout.readline().strip(), 'ready')
                checks = []
                def cancel():
                    checks.append(1)
                    if len(checks) >= 2: raise InterruptedError('cancelled')
                with self.assertRaises(InterruptedError):
                    InferenceLease(path, cancel).acquire()
            finally:
                child.terminate(); child.wait(timeout=5); child.stdout.close()
            with InferenceLease(path): pass

    def test_exception_releases(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'render.lock'
            with self.assertRaises(ValueError):
                with InferenceLease(path): raise ValueError('model failure')
            with InferenceLease(path): pass

    def test_installed_runtime_and_checkout_share_gpu_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            copy=Path(temporary)/'second-runtime';copy.mkdir()
            shutil.copy2(Path(__file__).resolve().parents[1]/'backend/inference_lease.py',copy/'inference_lease.py')
            script='''from inference_lease import GPUInferenceLease
checks=[]
def cancel():
    checks.append(1)
    if len(checks)>1:raise InterruptedError('cancelled')
try:
    with GPUInferenceLease(cancel):raise RuntimeError('second runtime incorrectly acquired the GPU')
except InterruptedError:print('shared lock verified')
'''
            with GPUInferenceLease():
                result=subprocess.run([sys.executable,'-c',script],cwd=copy,check=True,capture_output=True,text=True,timeout=5)
                self.assertIn('shared lock verified',result.stdout)


if __name__ == '__main__': unittest.main()
