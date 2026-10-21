import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from compare_fear_images import main, test_view, image_stats

class ComparisonTests(unittest.TestCase):
    def test_preprocessing_and_statistics(self):
        image=Image.new('RGB',(48,48),(100,100,100))
        self.assertEqual(test_view(image).size,(224,224))
        self.assertEqual(image_stats(image)['mean_channel_gap'],0)
        self.assertAlmostEqual(image_stats(image)['brightness'],100,places=4)
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/'EmoLabel').mkdir()
            (root/'EmoLabel/list_patition_label.txt').write_text('train_001.jpg 2\ntest_002.jpg 2\ntrain_003.jpg 4\n')
            csv_path=root/'pred.csv'
            with csv_path.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['path','true_class','predicted_class','fear_probability','fear_rank'])
                w.writeheader()
                for i,pred in enumerate(['Fear','Sad','Neutral','Surprise']):
                    w.writerow(dict(path=str(root/('img%d.jpg'%i)),true_class='Fear',predicted_class=pred,fear_probability=.2,fear_rank=2))
            args=['compare_fear_images.py','--source_path',str(root),'--prediction_csv',str(csv_path),'--output_dir',str(root/'out')]
            with patch('sys.argv',args), patch('compare_fear_images.load_image',return_value=(Image.new('RGB',(48,64),'gray'),'RGB')):
                main()
            summary=json.loads((root/'out/comparison_summary.json').read_text())
            self.assertEqual(summary['groups']['raf_train_fear']['total'],1)
            self.assertEqual(summary['groups']['fer_fear_as_sad_neutral']['total'],2)
            self.assertEqual(len(list((root/'out').glob('*.png'))),4)
