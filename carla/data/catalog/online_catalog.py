from typing import Any, Dict, List, Union

from carla.data.catalog import DataCatalog
from carla.data.load_catalog import load

from .load_data import load_dataset


class OnlineCatalog(DataCatalog):
    """
    Implements DataCatalog using already implemented datasets. These datasets are loaded from an online repository.

    Parameters
    ----------
    data_name : {'adult', 'compas', 'give_me_some_credit', 'heloc'}
        Used to get the correct dataset from online repository.

    Returns
    -------
    DataCatalog
    """

    def __init__(
        self,
        data_name: str,
        scaling_method: str = "MinMax",
        encoding_method: str = "OneHot_drop_binary",
    ):
        catalog_content = ["continuous", "categorical", "immutable", "target"]
        self.catalog: Dict[str, Any] = load(  # type: ignore
            "data_catalog.yaml", data_name, catalog_content
        )

        for key in ["continuous", "categorical", "immutable"]:
            if self.catalog[key] is None:
                self.catalog[key] = []

        # Load the raw data
        raw, train_raw, test_raw = load_dataset(data_name)

        super().__init__(
            data_name, raw, train_raw, test_raw, scaling_method, encoding_method
        )

    @property
    def categorical(self) -> List[str]:
        return self.catalog["categorical"]

    @property
    def continuous(self) -> List[str]:
        return self.catalog["continuous"]

    @property
    def immutables(self) -> List[str]:
        return self.catalog["immutable"]
    
    @property
    def mutables(self)-> List[str]:
        return list(set(self.categorical + self.continuous) - set(self.immutables))

    @property
    def target(self) -> str:
        return self.catalog["target"]
    
    def add_immutables(self, new_immutables: Union[str, List[str]]) -> None:
        """
        Adds new immutable features to the catalog.

        Parameters
        ----------
        new_immutables : str or List[str]
            The feature(s) to be added as immutable.
        """
        if isinstance(new_immutables, str):
            new_immutables = [new_immutables]

        self.catalog["immutable"] = list(set(self.catalog["immutable"] + new_immutables))

    def remove_immutables(self, to_remove: Union[str, List[str]]) -> None:
        """
        Removes features from the immutable list in the catalog.

        Parameters
        ----------
        to_remove : str or List[str]
            The feature(s) to be removed from immutable.
        """
        if isinstance(to_remove, str):
            to_remove = [to_remove]

        self.catalog["immutable"] = [
            feature for feature in self.catalog["immutable"] if feature not in to_remove
        ]
