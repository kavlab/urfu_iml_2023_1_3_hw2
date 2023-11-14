import streamlit as st

from mulyavin_aa import langdetector
from mulyavin_aa import translator

LANG_DETECTOR = "LANG_DETECTOR"
TRANSLATOR = "TRANSLATOR"


@st.cache_resource
def load_models() -> dict:
    """
    Получение справочника моделей
    :return: Справочник моделей
    """
    models = dict()
    models[LANG_DETECTOR] = langdetector.load_text_detection_model()
    models[TRANSLATOR] = translator.load_text_translator_model()

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
            # st.audio()

        with tab2:
            st.header("Таб 2")

        with tab3:
            st.header("Таб 3")


main_app()
