from fastapi.testclient import TestClient
from kuznetsov_av.api import Request, app

client = TestClient(app)

def test_root():
    response = client.get('/')

    assert response.status_code == 200
    assert response.json().get('message') is not None
    assert response.json().get('message') == "Converter method: /text-to-speech/convert/"

def test_text_to_speech():
    response = client.post(
        url='/text-to-speech/convert/',
        json=Request(text='Test').model_dump()
    )

    assert response.status_code == 200
    assert response.json().get('audio') is not None
    assert type(response.json().get('audio')) == str
    assert response.json().get('sampling_rate') is not None
    assert type(response.json().get('sampling_rate')) == int
