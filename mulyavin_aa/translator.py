# Модуль перевода языка

import transformers.pipelines.base
from transformers import pipeline


def load_text_translator_model() -> transformers.pipelines.base.Pipeline:
    """
    Подгрузка модели переводчика языка
    :return: Класс пайплайна для модели переводчика языка
    """
    return pipeline("translation", model=f'Helsinki-NLP/opus-mt-ru-en')


def translate_to_en(text: str, translator: transformers.pipelines.base.Pipeline) -> str:
    """
    Перевод текста с русского на английский
    :param text: Текст
    :param translator: Пайплайна для модели переводчика языка
    :return: Переведенный текст
    """
    text = translator(text)[0]['translation_text']
    print(text)
    return text
