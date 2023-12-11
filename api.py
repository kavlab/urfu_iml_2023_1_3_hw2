import uvicorn
from fastapi import FastAPI

import mulyavin_aa.langdetector
import mulyavin_aa.translator
import mulyavin_aa.model.langdetector
import mulyavin_aa.model.translator

import base64
from kuznetsov_av.api import Request, Response
import kuznetsov_av.text_to_speech_converter as t2s

import requests
from zvereva_ev import get_description_image, develop_api_app

app = FastAPI()


# Представление проекта урлом по умолчанию
@app.get("/")
async def root():
    """Получение базовой информации об API"""
    return {"message": "Проект ДЗ: Модуль 3. Разработка API для приложений искусственного интеллекта (vo_HW)",
            "git": "https://github.com/kavlab/urfu_iml_2023_1_3_hw2",
            "apis": [
                {
                    "descr": "API для определения языка текста ",
                    "base_url": "/langdetector"
                },
                {
                    "descr": "API для перевода текста из Ru в En",
                    "base_url": "/translator"
                },
                {
                    "descr": "API преобразования текста в речь",
                    "base_url": "/text-to-speech"
                },
                {
                    "descr": "API для описания загруженного изображения",
                    "base_url": "/get_description_image/predict/"
                }
            ]}


@app.post("/langdetector/detect")
def lang_detect(request: mulyavin_aa.model.langdetector.Request) \
        -> mulyavin_aa.model.langdetector.Response:
    """Определение языка текста"""
    pipe = mulyavin_aa.langdetector.load_text_detection_model()
    langs = pipe(request.text, )
    return mulyavin_aa.model.langdetector.Response(langs=langs)


@app.post("/translator/translate")
def lang_detect(request: mulyavin_aa.model.translator.Request) \
        -> mulyavin_aa.model.translator.Response:
    """Перевод текста из Ru в En"""
    pipe = mulyavin_aa.translator.load_text_translator_model()
    text = mulyavin_aa.translator.translate_to_en(request.text, pipe)

    return mulyavin_aa.model.translator.Response(text=text)


@app.post('/text-to-speech/convert/')
async def text_to_speech(entity: Request) -> Response:
    """
    Text-to-audio generation method using text_to_speech_converter.
    """
    synthesiser = t2s.load_model()
    embeddings_dataset = t2s.load_speaker_dataset()
    audio, sampling_rate = t2s.text_to_speech(entity.text, synthesiser, embeddings_dataset)
    return Response(audio=base64.b32encode(audio), sampling_rate=sampling_rate)


@app.post("/get_description_image/predict/")
async def get_description_image_predict(item: develop_api_app.Url):
    """
    Получение ссылки на изображение. Запись изображения под названием 'image.png для получения его описания
    """
    response = requests.get(item.url, stream=True)
    # сохранение изображения для дальнейшей передачи в модель
    with open("image.png", "wb") as f:
        f.write(response.content)

    return get_description_image.get_description_image()


# Запуск как приложения
if __name__ == '__main__':
    uvicorn.run(app, port=8000, host='127.0.0.1')
