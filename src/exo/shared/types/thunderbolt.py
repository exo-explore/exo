import anyio
from pydantic import BaseModel, Field

from exo.utils.pydantic_ext import CamelCaseModel


class TBConnection(CamelCaseModel):
    source_uuid: str
    sink_uuid: str


class TBIdentifier(CamelCaseModel):
    rdma_interface: str
    domain_uuid: str


## Intentionally minimal, only collecting data we care about - there's a lot more


class TBReceptacleTag(BaseModel, extra="ignore"):
    receptacle_id_key: str | None = None


class TBConnectivityItem(BaseModel, extra="ignore"):
    domain_uuid_key: str | None = None


class TBConnectivityData(BaseModel, extra="ignore"):
    domain_uuid_key: str | None = None
    items: list[TBConnectivityItem] | None = Field(None, alias="_items")
    receptacle_1_tag: TBReceptacleTag | None = None

    def ident(self, ifaces: dict[str, str]) -> TBIdentifier | None:
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
        return TBIdentifier(rdma_interface=iface, domain_uuid=self.domain_uuid_key)

    def conn(self) -> TBConnection | None:
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

        return TBConnection(source_uuid=self.domain_uuid_key, sink_uuid=sink_key)


class TBConnectivity(BaseModel):
    SPThunderboltDataType: list[TBConnectivityData] = []

    @classmethod
    async def gather(cls) -> list[TBConnectivityData] | None:
        proc = await anyio.run_process(
            ["system_profiler", "SPThunderboltDataType", "-json"], check=False
        )
        if proc.returncode != 0:
            return None
        # Saving you from PascalCase while avoiding too much pydantic
        return TBConnectivity.model_validate_json(proc.stdout).SPThunderboltDataType
