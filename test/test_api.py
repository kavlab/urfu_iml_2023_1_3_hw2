from fastapi.testclient import TestClient

from run_api import app

client = TestClient(app)


def test_get_description_image_predict() -> None:
    """
    Тест API Описание загруженного изображения
    """
    from zvereva_ev.develop_api_app import Url

    response = client.post(
        url="/get_description_image/predict/",
        json=Url(url="https://fikiwiki.com/uploads/posts/2022-02/1645000127_53-"
                     "fikiwiki-com-p-kartinki-krasivie-babochki"
                     "-narisovannie-55.png"
                 ).model_dump())

    assert response.status_code == 200
    assert response.text == '"Фото бабочки с оранжевыми крыльями и белыми точками"'


def test_langdetector_api_ru() -> None:
    """
    Тест API определения языка текста RU (mulyavin_aa)
    """
    import mulyavin_aa.model.langdetector

    api_resp = client.post(
        url="/langdetector/detect",
        json=mulyavin_aa.model.langdetector.Request(
            text='Доброго дня всем котам!').model_dump())

    response = mulyavin_aa.model.langdetector.Response.model_validate_json(api_resp.text)

    assert api_resp.status_code == 200
    assert len(response.langs) > 0
    assert response.langs[0].label == 'ru'


def test_langdetector_api_en() -> None:
    """
    Тест API определения языка текста EN (mulyavin_aa)
    """
    import mulyavin_aa.model.langdetector

    api_resp = client.post(
        url="/langdetector/detect",
        json=mulyavin_aa.model.langdetector.Request(
            text='So I checked functions in the class model').model_dump())

    response = mulyavin_aa.model.langdetector.Response.model_validate_json(api_resp.text)

    assert api_resp.status_code == 200
    assert len(response.langs) > 0
    assert response.langs[0].label == 'en'


def test_langdetector_api_err() -> None:
    """
    Тест API определения языка текста не Ru и En (mulyavin_aa)
    Модель не ограничена только Ru и En
    """
    import mulyavin_aa.model.langdetector

    api_resp = client.post(
        url="/langdetector/detect",
        json=mulyavin_aa.model.langdetector.Request(
            text='').model_dump())

    response = mulyavin_aa.model.langdetector.Response.model_validate_json(api_resp.text)

    assert api_resp.status_code == 200
    assert len(response.langs) > 0
    assert response.langs[0].label != 'en'
    assert response.langs[0].label != 'ru'


def test_translator_ru_to_en() -> None:
    """
    Тест API Перевод текста с языка Ru на En (mulyavin_aa)
    """
    import mulyavin_aa.model.translator

    api_resp = client.post(
        url="/translator/translate",
        json=mulyavin_aa.model.translator.Request(
            text='Доброго деня всем котам!').model_dump())

    response = mulyavin_aa.model.translator.Response.model_validate_json(api_resp.text)

    assert api_resp.status_code == 200
    assert len(response.text) > 0
    assert response.text == 'Good day to all cats!'


def test_translator_en_to_en() -> None:
    """
    Тест API Перевод текста с языка En на En (mulyavin_aa)
    """
    import mulyavin_aa.model.translator

    api_resp = client.post(
        url="/translator/translate",
        json=mulyavin_aa.model.translator.Request(
            text='Good day to all cats!').model_dump())

    response = mulyavin_aa.model.translator.Response.model_validate_json(api_resp.text)

    assert api_resp.status_code == 200
    assert len(response.text) > 0
    assert response.text == 'Good day to all cats!'
