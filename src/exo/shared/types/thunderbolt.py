import anyio
from pydantic import BaseModel, Field

from exo.utils.pydantic_ext import CamelCaseModel


class ThunderboltConnection(CamelCaseModel):
    source_uuid: str
    sink_uuid: str


class ThunderboltIdentifier(CamelCaseModel):
    rdma_interface: str
    domain_uuid: str


## Intentionally minimal, only collecting data we care about - there's a lot more


class _ReceptacleTag(BaseModel, extra="ignore"):
    receptacle_id_key: str | None = None


class _ConnectivityItem(BaseModel, extra="ignore"):
    domain_uuid_key: str | None = None


class ThunderboltConnectivityData(BaseModel, extra="ignore"):
    domain_uuid_key: str | None = None
    items: list[_ConnectivityItem] | None = Field(None, alias="_items")
    receptacle_1_tag: _ReceptacleTag | None = None

    def ident(self, ifaces: dict[str, str]) -> ThunderboltIdentifier | None:
        if (
            self.domain_uuid_key is None
            or self.receptacle_1_tag is None
            or self.receptacle_1_tag.receptacle_id_key is None
        ):
            return
        tag = f"Thunderbolt {self.receptacle_1_tag.receptacle_id_key}"
        assert tag in ifaces  # doesn't need to be an assertion but im confident
        # if tag not in ifaces: return None
        iface = f"rdma_{ifaces[tag]}"
        return ThunderboltIdentifier(
            rdma_interface=iface, domain_uuid=self.domain_uuid_key
        )

    def conn(self) -> ThunderboltConnection | None:
        if self.domain_uuid_key is None or self.items is None:
            return

        sink_key = next(
            (
                item.domain_uuid_key
                for item in self.items
                if item.domain_uuid_key is not None
            ),
            None,
        )
        if sink_key is None:
            return None

        return ThunderboltConnection(
            source_uuid=self.domain_uuid_key, sink_uuid=sink_key
        )


class ThunderboltConnectivity(BaseModel, extra="ignore"):
    SPThunderboltDataType: list[ThunderboltConnectivityData] = []

    @classmethod
    async def gather(cls) -> list[ThunderboltConnectivityData] | None:
        proc = await anyio.run_process(
            ["system_profiler", "SPThunderboltDataType", "-json"], check=False
        )
        if proc.returncode != 0:
            return None
        # Saving you from PascalCase while avoiding too much pydantic
        return ThunderboltConnectivity.model_validate_json(
            proc.stdout
        ).SPThunderboltDataType
