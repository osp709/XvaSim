"""Model registry and factory functions for modular stochastic models in XvaSim.

This module provides a centralized registry and factory pattern for discovering,
registering, and instantiating risk factor models across all supported categories
(interest rate, FX, credit, equity, etc.).

Public API
----------
- :class:`ModelRegistry` — registry for mapping risk factors and model names to classes.
- :func:`create_ir_model` — factory helper for creating interest rate models.
- :func:`create_credit_model` — factory helper for creating credit / hazard rate models.
- :func:`create_fx_model` — factory helper for creating FX models.
- :func:`create_inflation_model` — factory helper for creating inflation models.
- :func:`list_available_models` — query registered models by risk factor category.
"""

from __future__ import annotations

import typing

from .base import (
    CreditModel,
    FXModel,
    InflationModel,
    InterestRateModel,
    RiskFactorType,
    StochasticModel,
)

TModel = typing.TypeVar("TModel", bound=StochasticModel)


class ModelRegistry:
    """Central registry and factory for stochastic risk factor models."""

    _registry: typing.ClassVar[
        dict[tuple[RiskFactorType, str], type[StochasticModel]]
    ] = {}

    @typing.overload
    @classmethod
    def register(
        cls,
        risk_factor_type: RiskFactorType | str,
        model_name: str,
        model_cls: None = None,
    ) -> typing.Callable[[type[TModel]], type[TModel]]:
        ...

    @typing.overload
    @classmethod
    def register(
        cls,
        risk_factor_type: RiskFactorType | str,
        model_name: str,
        model_cls: type[TModel],
    ) -> type[TModel]:
        ...

    @classmethod
    def register(
        cls,
        risk_factor_type: RiskFactorType | str,
        model_name: str,
        model_cls: type[TModel] | None = None,
    ) -> typing.Callable[[type[TModel]], type[TModel]] | type[TModel]:
        """Register a model class for a specific risk factor type and name.

        Can be used as a direct method call or as a decorator.

        Args:
            risk_factor_type: :class:`RiskFactorType` or string (e.g. 'interest_rate').
            model_name: Normalized unique identifier string (e.g. 'lgm', 'hull_white').
            model_cls: The model class being registered (if not used as decorator).

        Returns:
            The registered class, or a decorator if *model_cls* is None.
        """
        rf_type = (
            RiskFactorType(risk_factor_type)
            if isinstance(risk_factor_type, str)
            else risk_factor_type
        )
        norm_name = model_name.strip().lower()

        def decorator(target_cls: type[TModel]) -> type[TModel]:
            cls._registry[(rf_type, norm_name)] = target_cls
            return target_cls

        if model_cls is not None:
            return decorator(model_cls)
        return decorator

    @classmethod
    def get(
        cls,
        risk_factor_type: RiskFactorType | str,
        model_name: str,
    ) -> type[StochasticModel]:
        """Retrieve a registered model class.

        Args:
            risk_factor_type: :class:`RiskFactorType` or string name.
            model_name: Model identifier string.

        Returns:
            The registered model class.

        Raises:
            KeyError: If no model is registered for the specified type and name.
        """
        rf_type = (
            RiskFactorType(risk_factor_type)
            if isinstance(risk_factor_type, str)
            else risk_factor_type
        )
        norm_name = model_name.strip().lower()
        key = (rf_type, norm_name)
        if key not in cls._registry:
            available = [name for (t, name) in cls._registry if t == rf_type]
            msg = (
                f"Model {model_name!r} not registered for risk factor "
                f"{rf_type.value!r}. Available models: {available}"
            )
            raise KeyError(msg)
        return cls._registry[key]

    @classmethod
    def create(
        cls,
        risk_factor_type: RiskFactorType | str,
        model_name: str,
        **kwargs: typing.Any,
    ) -> StochasticModel:
        """Instantiate a registered model with the given arguments.

        Args:
            risk_factor_type: :class:`RiskFactorType` or string name.
            model_name: Model identifier string.
            **kwargs: Arguments forwarded to the model constructor.

        Returns:
            An instantiated :class:`StochasticModel`.
        """
        model_cls = cls.get(risk_factor_type, model_name)
        return model_cls(**kwargs)

    @classmethod
    def list_models(
        cls,
        risk_factor_type: RiskFactorType | str | None = None,
    ) -> list[str]:
        """List all registered model names, optionally filtered by risk factor type.

        Args:
            risk_factor_type: Optional filter for risk factor category.

        Returns:
            List of registered model name strings.
        """
        if risk_factor_type is None:
            return sorted({name for (_, name) in cls._registry})

        rf_type = (
            RiskFactorType(risk_factor_type)
            if isinstance(risk_factor_type, str)
            else risk_factor_type
        )
        return sorted([name for (t, name) in cls._registry if t == rf_type])

    @classmethod
    def clear(cls) -> None:
        """Clear all registered models (useful for testing)."""
        cls._registry.clear()


def create_ir_model(
    model_name: str,
    **kwargs: typing.Any,
) -> InterestRateModel:
    """Factory helper to create an interest rate model by name.

    Args:
        model_name: Model identifier (e.g. 'lgm', 'hull_white', 'vasicek', 'cir').
        **kwargs: Arguments forwarded to the model class constructor.

    Returns:
        An instantiated :class:`InterestRateModel`.
    """
    model = ModelRegistry.create(RiskFactorType.INTEREST_RATE, model_name, **kwargs)
    if not isinstance(model, InterestRateModel):
        msg = f"Expected InterestRateModel instance, got {type(model).__name__}"
        raise TypeError(msg)
    return model


def create_credit_model(
    model_name: str,
    **kwargs: typing.Any,
) -> CreditModel:
    """Factory helper to create a credit / hazard rate model by name.

    Args:
        model_name: Model identifier (e.g. 'cir').
        **kwargs: Arguments forwarded to the model class constructor.

    Returns:
        An instantiated :class:`CreditModel`.
    """
    model = ModelRegistry.create(RiskFactorType.CREDIT, model_name, **kwargs)
    if not isinstance(model, CreditModel):
        msg = f"Expected CreditModel instance, got {type(model).__name__}"
        raise TypeError(msg)
    return model


def create_fx_model(
    model_name: str,
    **kwargs: typing.Any,
) -> FXModel:
    """Factory helper to create an FX model by name.

    Args:
        model_name: Model identifier (e.g. 'two_currency', 'gbm').
        **kwargs: Arguments forwarded to the model class constructor.

    Returns:
        An instantiated :class:`FXModel`.
    """
    model = ModelRegistry.create(RiskFactorType.FX, model_name, **kwargs)
    if not isinstance(model, FXModel):
        msg = f"Expected FXModel instance, got {type(model).__name__}"
        raise TypeError(msg)
    return model


def create_inflation_model(
    model_name: str,
    **kwargs: typing.Any,
) -> InflationModel:
    """Factory helper to create an inflation model by name.

    Args:
        model_name: Model identifier (e.g. 'jarrow_yildirim', 'jy', 'black').
        **kwargs: Arguments forwarded to the model class constructor.

    Returns:
        An instantiated :class:`InflationModel`.
    """
    model = ModelRegistry.create(RiskFactorType.INFLATION, model_name, **kwargs)
    if not isinstance(model, InflationModel):
        msg = f"Expected InflationModel instance, got {type(model).__name__}"
        raise TypeError(msg)
    return model


def list_available_models(
    risk_factor_type: RiskFactorType | str | None = None,
) -> list[str]:
    """Query registered model names, optionally filtered by risk factor type.

    Args:
        risk_factor_type: Optional filter for risk factor category.

    Returns:
        List of registered model name strings.
    """
    return ModelRegistry.list_models(risk_factor_type)

