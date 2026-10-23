import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from resolution_diagnostic import LowResolutionBottleneck,save_resolution_preview

class ResolutionTests(unittest.TestCase):
    def test_native_is_identity(self):
        image=Image.new('RGB',(100,100))
        self.assertIs(LowResolutionBottleneck(0)(image),image)

    def test_48_rgb_and_no_input_mutation(self):
        image=Image.new('RGB',(100,100),(255,100,30))
        output=LowResolutionBottleneck(48)(image)
        self.assertEqual(output.size,(48,48));self.assertEqual(output.mode,'RGB')
        self.assertEqual(image.size,(100,100))
        np.testing.assert_array_equal(np.asarray(output)[0,0],[255,100,30])

    def test_detail_loss_and_determinism(self):
        checker=(np.indices((100,100)).sum(0)%2*255).astype('uint8')
        image=Image.fromarray(checker).convert('RGB')
        a=np.asarray(LowResolutionBottleneck(48)(image))
        b=np.asarray(LowResolutionBottleneck(48)(image))
        np.testing.assert_array_equal(a,b)
        self.assertLess(a.std(),checker.std())

    def test_preview_and_invalid(self):
        with tempfile.TemporaryDirectory(dir='.') as d:
            path=Path(d)/'source.png';Image.new('RGB',(100,100)).save(path)
            save_resolution_preview([path],Path(d)/'preview.jpg',48)
            self.assertTrue((Path(d)/'preview.jpg').exists())
        for value in [-1,1.5,True]:
            with self.assertRaises(ValueError):LowResolutionBottleneck(value)

if __name__=='__main__':unittest.main()
