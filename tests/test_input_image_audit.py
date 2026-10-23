import tempfile
import unittest
from pathlib import Path
import numpy as np
from input_image_audit import audit_images,save_preview

class InputAuditTests(unittest.TestCase):
    def test_grayscale_and_color_statistics_preview(self):
        from PIL import Image
        with tempfile.TemporaryDirectory(dir='.') as root:
            root=Path(root);gray=root/'gray.png';color=root/'color.png'
            Image.new('RGB',(48,48),(120,120,120)).save(gray)
            Image.new('RGB',(100,80),(255,0,0)).save(color)
            result=audit_images([gray,color],2)
            self.assertEqual(result['records'][0]['mean_channel_spread'],0)
            self.assertEqual(result['records'][1]['mean_channel_spread'],255)
            self.assertEqual(result['records'][0]['width'],48)
            save_preview([color],[gray],root/'preview.jpg')
            self.assertTrue((root/'preview.jpg').exists())

    def test_limit_and_disabled(self):
        self.assertEqual(audit_images(['not_read'],0)['sample_count'],0)

if __name__=='__main__':unittest.main()
