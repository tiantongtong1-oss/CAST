import ast
import random
import unittest
from pathlib import Path
import numpy as np
from source_downsample import RandomSourceDownsample

class DownsampleTests(unittest.TestCase):
    def test_disabled_preserves_image_and_rng(self):
        image=np.zeros((100,100,3),dtype=np.uint8)
        state=random.getstate()
        self.assertIs(RandomSourceDownsample(0)(image),image)
        self.assertEqual(state,random.getstate())
    def test_enabled_reduces_details_without_changing_shape_or_color_channels(self):
        image=np.indices((100,100)).sum(0)%2*255
        image=np.repeat(image[:,:,None],3,axis=2).astype(np.uint8)
        result=RandomSourceDownsample(1,48)(image)
        self.assertEqual(result.shape,image.shape)
        self.assertEqual(result.dtype,image.dtype)
        self.assertLess(result.std(),image.std())
        np.testing.assert_array_equal(result[:,:,0],result[:,:,1])
    def test_small_image_and_validation(self):
        image=np.ones((32,32,3),dtype=np.uint8)
        np.testing.assert_array_equal(RandomSourceDownsample(1,48)(image),image)
        for probability,size in [(-1,48),(float('nan'),48),(.5,0)]:
            with self.assertRaises(ValueError):
                RandomSourceDownsample(probability,size)
    def test_source_only_wiring(self):
        tree=ast.parse(Path('train.py').read_text())
        bindings={}
        for node in ast.walk(tree):
            if isinstance(node,ast.Assign) and isinstance(node.value,ast.Call):
                for target in node.targets:
                    if isinstance(target,ast.Name): bindings[target.id]=node.value
        self.assertIn('source_transform',ast.unparse(bindings['source_train']))
        self.assertNotIn('source_transform',ast.unparse(bindings['target_train']))
        self.assertNotIn('source_transform',ast.unparse(bindings['source_prototype']))
