import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path
import numpy as np
from stage_diagnostics import StageStats, stage_masks

class StageDiagnosticTests(unittest.TestCase):
    def test_funnel_and_direct_branch(self):
        a=np.array([True]*4); confidence=np.array([True,False,False,False])
        distance=np.array([False,True,True,False]); vote=np.array([False,True,False,False])
        p=np.array([False,False,True,False]); proto=np.ones(4,dtype=bool)
        masks=stage_masks(a,confidence,distance,vote,p,proto,confidence,confidence,confidence)
        self.assertEqual(masks['rescue_candidate'].sum(),3)
        self.assertEqual(masks['rescue_distance'].sum(),2)
        self.assertEqual(masks['rescue_vote'].sum(),1)
        self.assertEqual(masks['rescue_probability'].sum(),0)
        self.assertEqual(masks['high_confidence'].sum(),1)

    def test_precision_coverage_and_confusion_orientation(self):
        stats=StageStats(7,True)
        stats.add([1,1,0],{'teacher_raw':np.ones(3,bool),'selected':np.array([True,False,False])},[1,0,1])
        self.assertEqual(stats.matrices['teacher_raw'][0,1],1)
        self.assertEqual(stats.matrices['teacher_raw'][1,0],1)
        with tempfile.TemporaryDirectory(dir='.') as d:
            path=str(Path(d)/'stats.csv')
            with contextlib.redirect_stdout(io.StringIO()): stats.report(path,3,'validation')
            rows=list(csv.DictReader(Path(path).read_text().splitlines()))
            r=next(r for r in rows if r['stage']=='selected' and r['class']=='Fear')
            self.assertEqual(float(r['precision']),1)
            self.assertEqual(float(r['correct_coverage']),.5)
            self.assertTrue(Path(path.replace('.csv','_confusion.csv')).exists())

    def test_training_labels_rejected(self):
        with self.assertRaises(ValueError): StageStats().add([1],{'teacher_raw':[True]},[1])

    def test_unselected_class_is_nan_not_perfect(self):
        stats=StageStats(7,True)
        stats.add([1],{'teacher_raw':[True]},[1])
        with tempfile.TemporaryDirectory(dir='.') as d:
            path=str(Path(d)/'stats.csv')
            with contextlib.redirect_stdout(io.StringIO()): stats.report(path,0,'validation')
            rows=list(csv.DictReader(Path(path).read_text().splitlines()))
            self.assertTrue(np.isnan(float(rows[0]['precision'])))

if __name__=='__main__': unittest.main()
