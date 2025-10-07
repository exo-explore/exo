import re
from typing import ClassVar

from pydantic import BaseModel, computed_field, field_validator


class Multiaddr(BaseModel):
    address: str

    PATTERNS: ClassVar[list[str]] = [
        r"^/ip4/(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(/tcp/(\d{1,5}))?(/p2p/[A-Za-z0-9]+)?$",
        r"^/ip6/([0-9a-fA-F:]+)(/tcp/(\d{1,5}))?(/p2p/[A-Za-z0-9]+)?$",
        r"^/dns[46]?/([a-zA-Z0-9.-]+)(/tcp/(\d{1,5}))?(/p2p/[A-Za-z0-9]+)?$",
    ]

    @field_validator("address")
    @classmethod
    def validate_format(cls, v: str) -> str:
        if not any(re.match(pattern, v) for pattern in cls.PATTERNS):
            raise ValueError(
                f"Invalid multiaddr format: {v}. "
                "Expected format like /ip4/127.0.0.1/tcp/4001 or /dns/example.com/tcp/443"
            )
        return v

    @computed_field
    @property
    def address_type(self) -> str:
        for pattern in self.PATTERNS:
            if re.match(pattern, self.address):
                return pattern.split("/")[1]
        raise ValueError(f"Invalid multiaddr format: {self.address}")

    @property
    def ipv6_address(self) -> str:
        match = re.match(r"^/ip6/([0-9a-fA-F:]+)", self.address)
        if not match:
            raise ValueError(
                f"Invalid multiaddr format: {self.address}. Expected format like /ip6/::1/tcp/4001"
            )
        return match.group(1)

    @property
    def ipv4_address(self) -> str:
        match = re.match(r"^/ip4/(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})", self.address)
        if not match:
            raise ValueError(
                f"Invalid multiaddr format: {self.address}. Expected format like /ip4/127.0.0.1/tcp/4001"
            )
        return match.group(1)

    @computed_field
    @property
    def ip_address(self) -> str:
        return self.ipv4_address if self.address_type == "ip4" else self.ipv6_address

    @computed_field
    @property
    def port(self) -> int:
        match = re.search(r"/tcp/(\d{1,5})", self.address)
        if not match:
            raise ValueError(
                f"Invalid multiaddr format: {self.address}. Expected format like /ip4/127.0.0.1/tcp/4001"
            )
        return int(match.group(1))

    def __str__(self) -> str:
        return self.address
