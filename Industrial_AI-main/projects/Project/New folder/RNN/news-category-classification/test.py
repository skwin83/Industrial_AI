from unittest import TestCase
import unittest
import PredictNewsCategory
import pandas as pd
import pandas.testing as pd_testing

class MyTest(TestCase):
    def test_normalization(self):
        data = {'title': ['매우 가벼운 노트북 출시!'], 'content': ['<html><body>이것은 노트북계의 새로운 혁명</body></html>']}
        pre = PredictNewsCategory.Preprocessor(pd.DataFrame(data))
        a = pre.preprocesse_normalization()
        ans_a = {'title': ['매우 가벼운 노트북 출시'], 'content': ['이것은 노트북계의 새로운 혁명']}
        ans_a = pd.DataFrame(ans_a)
        pd_testing.assert_frame_equal(a, ans_a)

if __name__ == '__main__':
    unittest.main()

