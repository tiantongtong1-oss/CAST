import unittest
from diagnose_selection import classify_path

class SelectionPathTests(unittest.TestCase):
    def test_direct_bypasses_radius_and_neighbor(self):
        self.assertEqual(classify_path(True, True, False, False, 0, .92), 'direct')
    def test_rescue_requires_all_gates_and_inclusive_threshold(self):
        self.assertEqual(classify_path(True, False, True, True, .92, .92), 'rescue')
        for agreement, radius, available, score in [(False,True,True,1), (True,False,True,1), (True,True,False,1), (True,True,True,.91)]:
            self.assertEqual(classify_path(agreement,False,radius,available,score,.92),'rejected')
