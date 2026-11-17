import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from head_stabilization import transforms


class StabilizationTests(unittest.TestCase):
    def test_smooth_linear_motion_is_preserved(self):
        t=np.arange(25);samples=np.column_stack((t,t*.1,.4+t*.001,.5+t*.002))
        matrices,_=transforms(samples,(512,512))
        np.testing.assert_allclose(matrices,np.tile(np.eye(2,3),(25,1,1)),atol=1e-10)

    def test_spike_is_bounded_and_remains_rigid(self):
        samples=np.tile([0.,0.,.5,.5],(25,1));samples[:,0]=np.arange(25)
        samples[12,1:]=[10,.6,.6]
        matrices,report=transforms(samples,(512,512))
        for matrix in matrices:
            np.testing.assert_allclose(matrix[:,:2].T@matrix[:,:2],np.eye(2),atol=1e-12)
        self.assertLessEqual(report['observed_max_translation_pixels'],3.0000001)
        self.assertLessEqual(report['observed_max_rotation_degrees'],.75)

    def test_missing_tracking_is_refused(self):
        samples=np.column_stack((np.arange(25),np.zeros(25),np.ones(25)*.5,np.ones(25)*.5))
        samples[12,0]=13
        with self.assertRaisesRegex(ValueError,'missing or repeated'):transforms(samples,(512,512))


if __name__=='__main__':unittest.main()
