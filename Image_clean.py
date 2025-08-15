import pandas as pd
from PIL import Image

def resize(filepath, size=(64, 64)):
    img = Image.open(filepath)
    img = img.resize(size)
    return img

def flatten_image(image):
    pixels = list(image.getdata()) 
    flat_pixels = [value for pixel in pixels for value in pixel]
    return flat_pixels

def img_to_csv(image, flatten=True):
    image = image.convert('RGB')
    pixels = list(image.getdata())
    if flatten:
        flat_pixels = [value for pixel in pixels for value in pixel]
        return pd.DataFrame([flat_pixels])
    else:
        return pd.DataFrame(pixels, columns=['R', 'G', 'B'])


img = resize('/home/mustafa/Pictures/picture.jpg')
img_csv = img_to_csv(img, flatten=True) 
print(img_csv)
