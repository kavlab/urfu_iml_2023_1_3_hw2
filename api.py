import uvicorn
from fastapi import FastAPI

import mulyavin_aa.langdetector
import mulyavin_aa.translator
import mulyavin_aa.model.langdetector
import mulyavin_aa.model.translator

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
                    "base_url": "/translator"}
            ]}


@app.post("/langdetector/detect")
def lang_detect(request: mulyavin_aa.model.langdetector.Request) \
        -> mulyavin_aa.model.langdetector.Response:
    """Определение языка текста"""
    pipe = mulyavin_aa.langdetector.load_text_detection_model()
    langs = pipe(request.text, )
    return mulyavin_aa.model.langdetector.Response(langs=langs)


@app.post("/translator/translate")
def lang_detect(request: mulyavin_aa.model.translator.TranslatorRequest) \
        -> mulyavin_aa.model.translator.TranslatorResponse:
    """Перевод текста из Ru в En"""
    response = mulyavin_aa.model.translator.TranslatorResponse()
    pipe = mulyavin_aa.translator.load_text_translator_model()
    response.text = mulyavin_aa.translator.translate_to_en(request.text, pipe)

    return response


# Запуск как приложения
if __name__ == '__main__':
    uvicorn.run(app, port=8080, host='127.0.0.1')
