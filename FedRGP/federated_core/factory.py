from typing import List, Type

from .base_federated_learner import BaseFederatedLearner
from .trainers.fedrgp_learner import FedRGPLearner


class FederatedLearnerFactory:
    """Factory for the federated learner provided by this repository."""

    _learner_registry = {"FedRGP": FedRGPLearner}

    @classmethod
    def create_learner(cls, model_name: str, cfg, args) -> BaseFederatedLearner:
        learner_class = cls._learner_registry.get(model_name)
        if learner_class is None:
            supported_models = ", ".join(cls.get_supported_models())
            raise ValueError(
                f"Unsupported federated model '{model_name}'. "
                f"Supported model: {supported_models}"
            )

        print(f"Creating federated learner: {learner_class.__name__} (model: {model_name})")
        return learner_class(cfg, args)

    @classmethod
    def register_learner(cls, model_name: str, learner_class: Type[BaseFederatedLearner]) -> None:
        cls._learner_registry[model_name] = learner_class
        print(f"Registered new learner: {model_name} -> {learner_class.__name__}")

    @classmethod
    def get_supported_models(cls) -> List[str]:
        return list(cls._learner_registry.keys())

    @classmethod
    def list_supported_models(cls) -> None:
        print("Supported federated learning models:")
        for model_name, learner_class in cls._learner_registry.items():
            print(f"  - {model_name}: {learner_class.__name__}")

    @classmethod
    def is_supported(cls, model_name: str) -> bool:
        return model_name in cls._learner_registry
