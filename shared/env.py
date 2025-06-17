import logging
import os
from typing import TypeVar

from pydantic import BaseModel, ValidationError

EnvSchema = TypeVar("EnvSchema", bound=BaseModel)


def get_validated_env(
    environment_schema: type[EnvSchema], logger: logging.Logger
) -> EnvSchema:
    """
    Validate and parse data into an instance of config_cls.
    Raises ValidationError if validation fails.
    """
    try:
        return environment_schema.model_validate(os.environ, strict=True)
    except ValidationError as e:
        logger.error("Environment Variables Validation Failed: %s", e)
        raise e
