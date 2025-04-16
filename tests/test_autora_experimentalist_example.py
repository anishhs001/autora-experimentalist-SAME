"""from autora.experimentalist.autora_experimentalist_example import sample
import numpy as np

def test_output_dimensions():
    X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    n = 2
    X_new = sample(X, n)

    # Check that the sampler returns n experiment conditions
    assert X_new.shape == (n, X.shape[1])


# Note: We encourage you to adjust this test and write more tests."""

import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from experimentalist import score_sample, compute_disagreement, sample, SAME_sample_type_alpha, SAME_sample_type_beta, SAME_sample_type_gamma

class TestExperimentalist(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.df, cls.X = cls.create_sample_data()
        cls.model_a, cls.model_b = cls.create_sample_models()
        
    @staticmethod
    def create_sample_data():
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        return df, X
    
    @staticmethod
    def create_sample_models():
        model_a = LogisticRegression()
        model_b = LogisticRegression()
        model_a.fit(*TestExperimentalist.create_sample_data()[1:])
        model_b.fit(*TestExperimentalist.create_sample_data()[1:])
        return model_a, model_b

    def test_score_sample(self):
        result = score_sample(self.df, [self.model_a, self.model_b], num_samples=5)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('score', result.columns)
    
    def test_compute_disagreement(self):
        disagreement = compute_disagreement(self.model_a.predict_proba, self.model_b.predict_proba, self.X)
        self.assertEqual(disagreement.shape, self.X.shape)
        self.assertTrue(np.issubdtype(disagreement.dtype, np.number))
    
    def test_sample(self):
        result = sample(self.df, [self.model_a, self.model_b], num_samples=5)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape[0], 5)
    
    def test_SAME_sample_type_alpha(self):
        result = SAME_sample_type_alpha(self.df, [self.model_a, self.model_b], self.df, 1, 10, num_samples=5)
        self.assertTrue(isinstance(result, pd.DataFrame) or isinstance(result, np.ndarray))
        self.assertTrue(result.shape[0] <= 5)
    
    def test_SAME_sample_type_beta(self):
        result = SAME_sample_type_beta(self.df, [self.model_a, self.model_b], self.df, 1, 10, num_samples=5)
        self.assertTrue(isinstance(result, pd.DataFrame) or isinstance(result, np.ndarray))
        self.assertTrue(result.shape[0] <= 5)
    
    def test_SAME_sample_type_gamma(self):
        result = SAME_sample_type_gamma(self.df, [self.model_a, self.model_b], self.df, 1, 10, num_samples=5)
        self.assertTrue(isinstance(result, pd.DataFrame) or isinstance(result, np.ndarray))
        self.assertTrue(result.shape[0] <= 5)

if __name__ == '__main__':
    unittest.main()
