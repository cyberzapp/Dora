import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from face_composite import composite_face


class CompositeTests(unittest.TestCase):
    def test_nonfinite_model_output_rejected(self):
        source=np.zeros((32,32,3),dtype=np.uint8)
        face=np.full((32,32,3),np.nan,dtype=np.float32)
        with self.assertRaisesRegex(ValueError,'Non-finite'):
            composite_face(source,face,np.eye(2,3),np.ones((32,32)),(32,32))

    def test_unsynthesized_pixels_remain_exact(self):
        source=np.random.default_rng(7).integers(0,256,(64,64,3),dtype=np.uint8)
        generated=np.full_like(source,230)
        mask=np.zeros((64,64),dtype=np.float32);mask[32:56,16:48]=1
        result=composite_face(source,generated,np.eye(2,3,dtype=np.float32),mask,(64,64))
        np.testing.assert_array_equal(result[mask==0],source[mask==0])
        np.testing.assert_array_equal(result[40:48,24:40],generated[40:48,24:40])

    def test_empty_mask_changes_nothing(self):
        source=np.full((64,64,3),120,dtype=np.uint8)
        result=composite_face(source,np.zeros_like(source),np.eye(2,3),np.zeros((64,64)),(64,64))
        np.testing.assert_array_equal(result,source)

    def test_inverse_translation_places_face_correctly(self):
        source=np.full((96,96,3),100,dtype=np.uint8)
        generated=np.full((32,32,3),220,dtype=np.uint8)
        # Original point (20,30) maps to aligned point (0,0).
        affine=np.array([[1,0,-20],[0,1,-30]],dtype=np.float32)
        mask=np.zeros((32,32),dtype=np.float32);mask[4:28,4:28]=1
        result=composite_face(source,generated,affine,mask,(32,32))
        np.testing.assert_array_equal(result[:30],source[:30])
        np.testing.assert_array_equal(result[42:50,32:40],generated[12:20,12:20])


if __name__=='__main__':unittest.main()
