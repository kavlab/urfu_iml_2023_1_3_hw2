from fastapi.testclient import TestClient
from api import app
from zvereva_ev.develop_api_app import Url
from zvereva_ev.get_description_image import get_description_image
import os

client = TestClient(app)


def test_get_description_image_predict():
    response = client.post("/get_description_image/predict/", json=Url(
        url="https://fikiwiki.com/uploads/posts/2022-02/1645000127_53-fikiwiki-com-p-kartinki-krasivie-babochki"
            "-narisovannie-55.png").model_dump())
    assert response.status_code == 200
    assert os.path.isfile("image.png") == True
    assert type(get_description_image()) == str
    assert get_description_image() == "Фото бабочки с оранжевыми крыльями и белыми точками"
