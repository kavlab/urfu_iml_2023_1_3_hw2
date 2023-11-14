import streamlit as st

from mulyavin_aa import langdetector
from mulyavin_aa import translator
from kuznetsov_av import text_to_speech_converter

LANG_DETECTOR = "LANG_DETECTOR"
TRANSLATOR = "TRANSLATOR"
TEXT_TO_SPEECH = "TEXT_TO_SPEECH"
SPEAKER_DATASET = "SPEAKER_DATASET"


@st.cache_resource
def load_models() -> dict:
    """
    Получение справочника моделей
    :return: Справочник моделей
    """
    models = dict()
    models[LANG_DETECTOR] = langdetector.load_text_detection_model()
    models[TRANSLATOR] = translator.load_text_translator_model()
    models[TEXT_TO_SPEECH] = text_to_speech_converter.load_model()
    models[SPEAKER_DATASET] = text_to_speech_converter.load_speaker_dataset()

    return models


def main_app():
    """
    Основная программа
    """

    models = load_models()

    st.title = 'Домашнее задание'

    # Оформление заголовка
    st.header('Домашнее задание', divider='gray')

    input_text = st.text_area(
        'Введите текст на русском или английском языке и нажмите кнопку генератора:')

    if st.button('Генерировать!!!'):
        # Определение языка
        text_lang = langdetector.lang_detect(input_text, models[LANG_DETECTOR])
        if text_lang not in ['ru', 'en']:
            st.error('Язык текста не может быть определен')
            return

        # Перевод языка если не en
        if text_lang in ['ru']:
            input_text = translator.translate_to_en(input_text, models[TRANSLATOR])

        tab1, tab2, tab3 = st.tabs(['Озвученный текст', 'Таб 2', 'Таб 3'])
        with tab1:
            st.header("Озвученный текст на английском языке")
            # Преобразование текста в речь
            audio_data, sampling_rate = text_to_speech_converter.text_to_speech(
                input_text, models[TEXT_TO_SPEECH], models[SPEAKER_DATASET])
            st.audio(data=audio_data, sample_rate=sampling_rate)

        with tab2:
            st.header("Таб 2")

        with tab3:
            st.header("Таб 3")


main_app()
