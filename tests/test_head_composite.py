import sys,json,tempfile
from pathlib import Path
import unittest
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from head_composite import head_mask, composite_head, source_face_bounds


class HeadCompositeTests(unittest.TestCase):
    def test_native_metadata_fallback_and_missing_guidance(self):
        with tempfile.TemporaryDirectory() as folder:
            photo=Path(folder)/'outfit-prepared.png'
            metadata=Path(folder)/'outfit-landmarks.json'
            bounds=[.3,.4,.2,.25]
            metadata.write_text(json.dumps({'face':bounds}))
            self.assertEqual(source_face_bounds({'photo':str(photo)}),bounds)
            self.assertEqual(source_face_bounds({'photo':str(photo),'face_bounds':bounds}),bounds)
            metadata.unlink()
            with self.assertRaisesRegex(ValueError,'Reimport'):
                source_face_bounds({'photo':str(photo)})

    def test_body_and_background_outside_head_are_exact(self):
        source=np.full((256,384,3),80,dtype=np.uint8)
        generated=np.full_like(source,200)
        mask=head_mask(Image.fromarray(source),(384,256),[.375,.514,.17,.255])
        result=composite_head(source,generated,mask)
        np.testing.assert_array_equal(result[mask==0],source[mask==0])
        np.testing.assert_array_equal(result[190:],source[190:])
        self.assertTrue((result[mask==1]==200).all())

    def test_padding_is_included_in_face_transform(self):
        photo=Image.new('RGB',(400,400))
        mask=head_mask(photo,(800,400),[.4,.4,.2,.2])
        self.assertGreater(mask[200,400],.99)
        self.assertEqual(float(mask[200,100]),0)

    def test_invalid_face_and_mask_rejected(self):
        with self.assertRaises(ValueError):head_mask(Image.new('RGB',(20,20)),(20,20),[0,0,float('nan'),1])
        with self.assertRaises(ValueError):composite_head(np.zeros((2,2,3),dtype=np.uint8),np.zeros((2,2,3),dtype=np.uint8),np.full((2,2),np.nan))


if __name__=='__main__':unittest.main()
