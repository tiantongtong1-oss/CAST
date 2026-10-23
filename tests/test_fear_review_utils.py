import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from fear_review_utils import select_review_groups,nearest_source,model_crop_image,render_pages

class ReviewTests(unittest.TestCase):
    def test_selection_high_confidence_no_backfill(self):
        p=np.zeros((4,7));p[0,4]=.9;p[0,1]=.1;p[1,6]=.6;p[1,1]=.4;p[2,1]=1;p[3,4]=1
        g=select_review_groups(p,[1,1,1,4],20,.8)
        np.testing.assert_array_equal(g['failed_fear'],[0]);np.testing.assert_array_equal(g['correct_fear'],[2])

    def test_retrieval_ignores_labels(self):
        ids,sim=nearest_source([[1,0],[0,1],[-1,0]],[[3,0]],2)
        np.testing.assert_array_equal(ids,[[0,1]]);np.testing.assert_allclose(sim,[[1,0]])

    def test_actual_tensor_roundtrip_and_pages(self):
        rgb=np.full((224,224,3),128,dtype='uint8')
        x=((rgb/255-np.array([.485,.456,.406]))/np.array([.229,.224,.225])).transpose(2,0,1)
        crop=model_crop_image(x);np.testing.assert_array_equal(np.asarray(crop),rgb)
        with tempfile.TemporaryDirectory(dir='.') as root:
            root=Path(root);path=root/'input.png';crop.save(path)
            pages=render_pages([dict(path=path,crop=crop,caption='sample')]*6,root,'test')
            self.assertEqual(len(pages),2)
            self.assertTrue(all(Path(p).exists() for p in pages))

if __name__=='__main__':unittest.main()
