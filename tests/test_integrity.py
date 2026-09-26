import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pandas as pd
from class_mapping import probability_order,category
from export_metrics import PUBLISHED,audit
from audit_reader_agreement import analyze
class IntegrityTests(unittest.TestCase):
    def test_checkpoint_class_order(self):
        self.assertEqual(probability_order({0:'syndesmofity',1:'normal',2:'osteofity',3:'parasyndesmofity'}),[1,2,3,0])
        with self.assertRaises(ValueError):probability_order(['normal']*4)
        with self.assertRaises(ValueError):category('grade4')
    def test_precision_recall_orientation(self):
        a=audit(PUBLISHED['external'],['normal','osteophytes','parasyndesmophytes','syndesmophytes'],20)
        self.assertEqual(a['n'],150)
        self.assertAlmostEqual(a['overall']['accuracy']['estimate'],.9)
        self.assertAlmostEqual(a['per_class']['osteophytes']['sensitivity']['estimate'],49/54)
        self.assertAlmostEqual(a['per_class']['osteophytes']['PPV']['estimate'],49/58)
    def test_reader_missingness_and_invalid_codes(self):
        d=pd.DataFrame({'Case_ID':['a','b','c','d'],'Cohort':['X']*4,'M.P.':[0,1,2,3],'P.G.':[0,1,2,3],'Consensus':[0,1,2,3]})
        self.assertEqual(analyze(d,20)['All']['agreement'],4)
        d.loc[0,'M.P.']=np.nan
        with self.assertRaises(ValueError):analyze(d,20)
        d.loc[0,'M.P.']=4
        with self.assertRaises(ValueError):analyze(d,20)
if __name__=='__main__':unittest.main()
