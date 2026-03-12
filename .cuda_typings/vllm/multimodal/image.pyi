from PIL import Image

def rescale_image_size(
    image: Image.Image, size_factor: float, transpose: int = -1
) -> Image.Image: ...
def rgba_to_rgb(
    image: Image.Image,
    background_color: tuple[int, int, int] | list[int] = (255, 255, 255),
) -> Image.Image: ...
def convert_image_mode(image: Image.Image, to_mode: str): ...
