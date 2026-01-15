from anyio import Path, open_file
import tomlkit

from exo.shared.types.models import ModelId, ModelMetadata
from exo.shared.models.model_meta import get_model_meta
from exo.utils.pydantic_ext import CamelCaseModel


class ModelCard(CamelCaseModel):
    short_id: str
    model_id: ModelId
    name: str
    description: str
    tags: list[str]
    metadata: ModelMetadata

    @staticmethod
    async def load(path: Path) -> "ModelCard":
        async with await open_file(path) as f:
            data = await f.read()
            py = tomlkit.loads(data)
            return ModelCard.model_validate(py)

    async def save(self, path: Path):
        async with await open_file(path, "w") as f:
            py = self.model_dump()
            data = tomlkit.dumps(py)  # pyright: ignore[reportUnknownMemberType]
            await f.write(data)

    @staticmethod
    async def from_hf(model_id: str) -> "ModelCard":
        short_name = model_id.split("/")[-1]
        return ModelCard(
            short_id=short_name,
            model_id=ModelId(model_id),
            name=short_name,
            description=f"Custom model from {model_id}",
            tags=[],
            metadata=await get_model_meta(model_id),
        )
