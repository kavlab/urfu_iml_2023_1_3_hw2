---
title: URFU PE
emoji: 💻
colorFrom: yellow
colorTo: blue
sdk: streamlit
sdk_version: 1.28.2
app_file: run.py
pinned: false
---

# Программная инженерия. Практическое задание №2 и №3

Приложение разработано с использованием фреймворка [Streamlit](https://streamlit.io/).
Состоит из двух страниц и Главной страницы:
1. Главная страница - содержит описание из README
2. Генератор аудио - позволяет сгенерировать аудио по введенному тексту на английском языке. Дополнительно осуществляет перевод с русского языка на английский (при вводе текста на русском языке). Используется 3 модели:
    - Определение языка текста
    - Перевод текста с языка Ru на En
    - Озвучивание текста на английском языке
3. Описание изображения - позволяет получить описание изображения на русском языке. Использует 1 модель:
    - Классификация и описание изображений

API разработано с использованием фреймворка [FastAPI]('https://fastapi.tiangolo.com/'). API доступные для использования:
1. ```/langdetector/detect``` - Определение языка текста
2. ```/translator/translate``` - Перевод текста с языка Ru на En
3. ```/text-to-speech/convert``` - Преобразование текста на английском языке в речь
4. ```/get_description_image/predict``` - Описание загруженного изображения

## Используемые модели
- Определение языка текста - [papluca/xlm-roberta-base-language-detection](https://huggingface.co/papluca/xlm-roberta-base-language-detection)
- Перевод текста с языка Ru на En - [Helsinki-NLP/opus-mt-ru-en](https://huggingface.co/Helsinki-NLP/opus-mt-ru-en)
- Озвучивание текста на английском языке - [microsoft/speecht5_tts](https://huggingface.co/microsoft/speecht5_tts)
- Классификация и описание изображений. Модель описания изображения [Salesforce/blip-image-captioning-large](https://huggingface.co/Salesforce/blip-image-captioning-large)

## Как запустить Web-приложение
Запуск осуществляется через модуль streamlit:
```
streamlit run run.py
```

## Как запустить API-сервер
```uvicorn api:app``` либо ```python api.py```

## Как использовать Web-приложение
После запуска приложение открывается на Главной странице. Выбор режима работы приложения доступен слева в меню

### Генератор аудио
Необходимо ввести текст в текстовое поле и нажать кнопку "Генерировать!!!". В результате появится аудио запись на английском языке с описанием введенного текста.

![Результат работы моделей "Генератор аудио / Текст"](https://raw.githubusercontent.com/kavlab/urfu_iml_2023_1_3_hw2/main/mulyavin_aa/audio_gen_image.png)
![Результат работы моделей "Генератор аудио / Аудио"](https://raw.githubusercontent.com/kavlab/urfu_iml_2023_1_3_hw2/main/kuznetsov_av/text_to_speech_image.png)

### Описание изображения
Необходимо выбрать изображение и нажать кнопку "Получить описание изображения". В результате появится текстовое описание изображения на русском языке.

![Результат работы моделей "Классификации и описания изображений"](https://raw.githubusercontent.com/kavlab/urfu_iml_2023_1_3_hw2/main/zvereva_ev/image_result.jpg)

### Как использовать API
Описание методов API генерируется Swagger и доступно по адресу
```
<host>/docs
```

#### Пример вызова сервиса Определение языка текста
Вызвать url сервиса ```<host>/langdetector/detect``` методом POST

![img.png](https://raw.githubusercontent.com/kavlab/urfu_iml_2023_1_3_hw2/main/mulyavin_aa/PostmanLangDetect.png)

Передаваемые параметры:
```
{
    "text": "Доброго деня всем котам!"
}
```

Результат выполнения:
```
{
    "langs": [
        {
            "label": "ru",
            "score": 0.9351206421852112
        }
    ]
}
```

#### Пример вызова сервиса Перевод текста с языка Ru на En
Вызвать url сервиса ```<host>/translator/translate``` методом POST

![img.png](https://raw.githubusercontent.com/kavlab/urfu_iml_2023_1_3_hw2/main/mulyavin_aa/PostmanTranslate.png)

Передаваемые параметры:
```
{
    "text": "Доброго деня всем котам!"
}
```

Результат выполнения:
```
{
    "text": "Good day to all cats!"
}
```


#### Пример вызова сервиса Преобразование текста в речь
Вызвать url сервиса ```<host>/text-to-speech/convert/``` методом POST

![img.png](https://raw.githubusercontent.com/kavlab/urfu_iml_2023_1_3_hw2/main/kuznetsov_av/text_to_speech_image_api.png)

Передаваемые параметры:
```
{
    "text": "FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.8+ based on standard Python type hints."
}
```

Результат выполнения (значение в audio сокращено):
```
{
    "audio": "4HXHYOWAMX2TVFFRYI5NQYFYHKESNWR2YAPHKOWFOCYTULNSF45YUZMDHJEKRJB2QNGMMOX3V6QDV6 ... F5Z5S37LOH7G2ZLSRYSKC4Z47GBXG56ZQ5YY6BEENY=",
    "sampling_rate": 16000
}
```

#### Пример вызова сервиса Описание загруженного изображения
Вызвать url сервиса ```<host>/get_description_image/predict/``` методом POST

![img.png](https://raw.githubusercontent.com/kavlab/urfu_iml_2023_1_3_hw2/main/zvereva_ev/API_image_postman.jpg)

Передаваемые параметры:
```
{
    "text": "https://fikiwiki.com/uploads/posts/2022-02/1645000127_53-fikiwiki-com-p-kartinki-krasivie-babochki-narisovannie-55.png"
}
```

Результат выполнения:
```
{
    "text": "Фото бабочки с оранжевыми крыльями и белыми точками"
}
```
