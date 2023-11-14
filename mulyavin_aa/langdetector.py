# Модуль определения языка

import transformers.pipelines.base
from transformers import pipeline


def load_text_detection_model() -> transformers.pipelines.base.Pipeline:
    """
    Подгрузка модели детектора языка
    :return: Класс пайплайна для модели детектора языка
    """
    return pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")


def lang_detect(text: str, langdetector: transformers.pipelines.base.Pipeline) -> str | None:
    """
    Определение языка для введенного текста
    :param text: Текст
    :param langdetector: Пайплайн для модели детектора языка
    :return: Код определенного языка (если определен)
    """
    text_langs = list(langdetector(text, ))

    if not text_langs:
        return None

    for i in range(3):
        if i > len(text_langs) - 1:
            break
        print(text_langs[i])
        if text_langs[i]['label'] in ['ru', 'en']:
            return text_langs[i]['label']

    return None
