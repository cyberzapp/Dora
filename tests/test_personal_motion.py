import sys,unittest
from pathlib import Path
import torch
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from personal_motion import PersonalMotionMixer,SourceAnchoredMouth,LIP_POINTS,EYE_POINTS


def info(value=0,pitch=0):
    return dict(exp=torch.full((1,21,3),float(value)),pitch=torch.tensor([float(pitch)]),yaw=torch.zeros(1),roll=torch.zeros(1),scale=torch.ones((1,1)),t=torch.zeros((1,3)),kp=torch.zeros((1,21,3)))


class PersonalMotionTests(unittest.TestCase):
    def test_source_anchor_survives_changing_driver_pose_and_output_mutation(self):
        source=info(2,7);driver=info(9,-30)
        driver['yaw'].fill_(60);driver['t'].fill_(3);driver['scale'].fill_(4)
        mixer=SourceAnchoredMouth(source)
        result=mixer(driver)
        self.assertTrue(torch.equal(result['exp'][:,LIP_POINTS],driver['exp'][:,LIP_POINTS]))
        others=[i for i in range(21) if i not in LIP_POINTS]
        self.assertTrue(torch.equal(result['exp'][:,others],source['exp'][:,others]))
        for key in ('pitch','yaw','roll','t','scale','kp'):
            self.assertTrue(torch.equal(result[key],source[key]))
        result['exp'].zero_();result['pitch'].zero_()
        again=mixer(driver)
        self.assertEqual(float(again['pitch'][0]),7)
        self.assertEqual(float(source['exp'][0,0,0]),2)
        self.assertEqual(float(driver['exp'][0,0,0]),9)
        driver['exp'][0,6,0]=float('nan')
        with self.assertRaisesRegex(ValueError,'Invalid driving'):mixer(driver)

    def test_mouth_unchanged_personal_blinks_and_restrained_pose(self):
        first=info();second=info(1,10)
        second['t'].fill_(1);second['scale'].fill_(2)
        mixer=PersonalMotionMixer([first,second],pose_strength=.5,expression_strength=.6,position_strength=.2)
        mixer(info())
        audio=info(7);before=audio['exp'].clone()
        result=mixer(audio)
        self.assertTrue(torch.equal(result['exp'][:,LIP_POINTS],audio['exp'][:,LIP_POINTS]))
        self.assertTrue(torch.equal(result['exp'][:,EYE_POINTS],torch.ones((1,len(EYE_POINTS),3))))
        self.assertAlmostEqual(float(result['exp'][0,0,0]),.6,places=6)
        self.assertAlmostEqual(float(result['pitch'][0]),5,places=5)
        self.assertAlmostEqual(float(result['scale'][0,0]),1.2,places=5)
        self.assertTrue(torch.equal(audio['exp'],before))
        with self.assertRaisesRegex(ValueError,'shorter'):mixer(info())

    def test_full_strength_rotation_matches_recorded_pose(self):
        a,b=info(),info()
        a['pitch'].fill_(8);a['yaw'].fill_(-7);a['roll'].fill_(4)
        b['pitch'].fill_(-12);b['yaw'].fill_(14);b['roll'].fill_(-9)
        mixer=PersonalMotionMixer([a,b],pose_strength=1)
        for reference in (a,b):
            result=mixer(info())
            actual=[float(result[k][0]) for k in ('pitch','yaw','roll')]
            expected=[float(reference[k][0]) for k in ('pitch','yaw','roll')]
            np.testing.assert_allclose(Rotation.from_euler('xyz',actual,degrees=True).as_matrix(),Rotation.from_euler('xyz',expected,degrees=True).as_matrix(),atol=1e-6)


if __name__=='__main__':unittest.main()
