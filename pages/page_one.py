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


def page_one():
    """
    Основная программа
    """

    models = load_models()

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

        st.subheader('Озвученный текст на английском языке', divider='gray')

        st.text(input_text)

        # Преобразование текста в речь
        with st.status('Пожалуйста подождите, идет преобразование текста в речь...') as status:
            audio_data, sampling_rate = text_to_speech_converter.text_to_speech(
                input_text, models[TEXT_TO_SPEECH], models[SPEAKER_DATASET])
            status.update(label='Преобразование завершено. Для прослушивания нажмите кнопку воспроизведения.', state='complete')
        
        st.audio(data=audio_data, sample_rate=sampling_rate)


page_one()
