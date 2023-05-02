import pytest
from pathlib import Path
from fastai.vision.all import PILImage


@pytest.fixture
def negative_img():
    basepath = Path(__file__).parent
    imgpath = 'input.png'
    img = PILImage.create(basepath / imgpath)
    return img