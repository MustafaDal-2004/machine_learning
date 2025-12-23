import pandas as pd
from PIL import Image
from typing import Tuple


# ============================================================
# Image Processing Utilities
# ============================================================

def load_and_resize_image(filepath: str, size: Tuple[int, int] = (64, 64)) -> Image.Image:
    """
    Loads an image from disk and resizes it.

    Parameters:
        filepath (str): Path to the image file
        size (Tuple[int, int]): Target image size (width, height)

    Returns:
        PIL.Image.Image: Resized image
    """
    image = Image.open(filepath)
    return image.resize(size)


def flatten_image_pixels(image: Image.Image) -> list:
    """
    Flattens an RGB image into a single list of pixel values.

    Parameters:
        image (PIL.Image.Image): Input image

    Returns:
        list: Flattened pixel values [R, G, B, R, G, B, ...]
    """
    pixels = image.getdata()
    return [value for pixel in pixels for value in pixel]


def image_to_dataframe(image: Image.Image, flatten: bool = True) -> pd.DataFrame:
    """
    Converts an image into a pandas DataFrame.

    Parameters:
        image (PIL.Image.Image): Input image
        flatten (bool): If True, returns a single-row DataFrame.
                        If False, returns one row per pixel (R, G, B).

    Returns:
        pd.DataFrame: Image represented as a DataFrame
    """
    image = image.convert("RGB")
    pixels = list(image.getdata())

    if flatten:
        flat_pixels = [value for pixel in pixels for value in pixel]
        return pd.DataFrame([flat_pixels])
    else:
        return pd.DataFrame(pixels, columns=["R", "G", "B"])


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    IMAGE_PATH = "data/images/picture.jpg"
    TARGET_SIZE = (64, 64)

    image = load_and_resize_image(IMAGE_PATH, TARGET_SIZE)
    image_df = image_to_dataframe(image, flatten=True)

    print(image_df.head())
