import sys
from pathlib import Path
import unittest
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from flashhead_layout import Layout


class LayoutTests(unittest.TestCase):
    def test_body_is_exact_after_export_framing(self):
        photo=Image.fromarray(np.random.default_rng(5).integers(0,256,(512,768,3),dtype=np.uint8))
        for size in [(640,360),(360,640)]:
            layout=Layout(photo,size,[.375,.514,.17,.255],[200,50,500,350])
            generated=np.full((512,512,3),220,dtype=np.uint8)
            result=layout.frame(generated)
            self.assertEqual(result.shape,(size[1],size[0],3))
            np.testing.assert_array_equal(result[layout.mask==0],layout.source[layout.mask==0])

    def test_crop_that_would_cut_the_face_uses_fit(self):
        layout=Layout(Image.new('RGB',(400,800)),(640,360),[.2,.2,.6,.6],[0,100,400,700])
        self.assertEqual(layout.mode,'fit')
        result=layout.frame(np.zeros((512,512,3),dtype=np.uint8))
        self.assertEqual(result.shape,(360,640,3))


if __name__=='__main__':unittest.main()
