from pydantic import BaseModel, field_validator


# TODO: Is this the right place for this? Host is consumed by worker, but typically stored in the master
class Host(BaseModel):
    host: str
    port: int

    def __str__(self) -> str:
        return f"{self.host}:{self.port}"

    @field_validator("port")
    @classmethod
    def check_port(cls, v: int) -> int:
        if not (0 <= v <= 65535):
            raise ValueError("Port must be between 0 and 65535")
        return v
