import pytest
from pathlib import Path
from fastai.vision.all import PILImage
import requests


@pytest.fixture
def negative_img():
    basepath = Path(__file__).parent
    imgpath = 'input.png'
    img = PILImage.create(basepath / imgpath)
    return img

def is_responsive(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return True
    except requests.exceptions.ConnectionError:
        return False

@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    return pytestconfig.rootpath.joinpath("docker-compose.yml")

@pytest.fixture(scope="session")
def http_service(docker_ip, docker_services):
    """Ensure that HTTP service is up and responsive."""

    # `port_for` takes a container port and returns the corresponding host port
    port = docker_services.port_for("app", 8000)
    url = "http://{}:{}".format(docker_ip, port)
    docker_services.wait_until_responsive(
        timeout=1200, pause=1, check=lambda: is_responsive(url)
    )
    return url