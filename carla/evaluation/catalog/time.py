import pandas as pd
from typing import Dict

from carla.evaluation.api import Evaluation
from carla.recourse_methods.api import RecourseMethod


class AvgTime(Evaluation):
    """
    Computes average time for generated counterfactual
    """

    def __init__(self, recourse_method:RecourseMethod, hyperparameters: Dict={}):
        hyperparameters["time"] = recourse_method.time
        super().__init__(hyperparameters)
        self.time = hyperparameters["time"]
        self.columns = ["avg_time"]

    def get_evaluation(
        self, factuals: pd.DataFrame, counterfactuals: pd.DataFrame
    ) -> pd.DataFrame:
        avg_time = self.time / len(counterfactuals)
        return pd.DataFrame([[avg_time]], columns=self.columns)
