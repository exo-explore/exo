import re
from ipaddress import IPv4Address
from typing import ClassVar

from pydantic import BaseModel, computed_field, field_serializer, field_validator


class Multiaddr(BaseModel):
    address: str
    
    PATTERNS: ClassVar[list[str]] = [
        r'^/ip4/(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(/tcp/(\d{1,5}))?(/p2p/[A-Za-z0-9]+)?$',
        r'^/ip6/([0-9a-fA-F:]+)(/tcp/(\d{1,5}))?(/p2p/[A-Za-z0-9]+)?$',
        r'^/dns[46]?/([a-zA-Z0-9.-]+)(/tcp/(\d{1,5}))?(/p2p/[A-Za-z0-9]+)?$',
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
    def ipv4_address(self) -> IPv4Address:
        match = re.match(r'^/ip4/(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', self.address)
        if not match:
            raise ValueError(f"Invalid multiaddr format: {self.address}. Expected format like /ip4/127.0.0.1/tcp/4001")
        return IPv4Address(match.group(1))

    @field_serializer("ipv4_address")
    def serialize_ipv4_address(self, value: IPv4Address) -> str:
        return str(value)

    
    @computed_field
    @property
    def port(self) -> int:
        match = re.search(r'/tcp/(\d{1,5})', self.address)
        if not match:
            raise ValueError(f"Invalid multiaddr format: {self.address}. Expected format like /ip4/127.0.0.1/tcp/4001")
        return int(match.group(1))
    

    def __str__(self) -> str:
        return self.address
