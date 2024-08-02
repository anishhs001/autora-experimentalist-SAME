"""
Example Experimentalist
"""
import numpy as np
import pandas as pd

from typing import Union, List
from autora.experimentalist.falsification import falsification_sample
from autora.experimentalist.model_disagreement import model_disagreement_sample
from autora.experimentalist.novelty import novelty_sample

class SampleExperimentalist:
    def __init__(self, 
                exp_data,
                conditions: Union[pd.DataFrame, np.ndarray], 
                models: List,
                reference_conditions: Union[pd.DataFrame, np.ndarray],
                num_samples: int = 1) -> None:
        """
        Class initialiser.

    
        Args:
            conditions: The pool to sample from.
                Attention: `conditions` is a field of the standard state
            models: The sampler might use output from the theorist.
                Attention: `models` is a field of the standard state
            reference_conditions: The sampler might use reference conditons
            num_samples: number of experimental conditions to select
        
        Returns:
            None
        """
        self.exp_data = exp_data
        self.conditions = conditions
        self.models = models
        self.reference_conditions = reference_conditions
        self.num_samples = num_samples

    def sample(self) -> pd.DataFrame:
        """
        When called, returns samples with the values of class initiation.

        Returns:
            Sampled pool of experimental conditions

        *Optional*
        Examples:
            These examples add documentation and also work as tests
            >>> example_sampler([1, 2, 3, 4])
            1
            >>> example_sampler(range(3, 10))
            3

        """
        if self.num_samples is None:
            self.num_samples = self.conditions.shape[0]

        new_conditions = novelty_sample(
          self.conditions,
          reference_conditions= self.reference_conditions,
          num_samples = self.num_samples
      )

        return new_conditions
