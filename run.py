import streamlit as st

from pathlib import Path


def st_page_rename(pages_name: dict[str, str]) -> None:
    """
    Переименование страниц в сайдбаре
    Временный хак, так как другие способы не сработали
    :param pages_name: Список py файлов и имен
    """
    from streamlit.source_util import get_pages as st_get_pages
    from streamlit.source_util import _on_pages_changed as st_on_pages_changed

    pages = st_get_pages("")
    for page_k, page_v in pages.items():
        script_path = Path(page_v["script_path"])

        for page_name_k, page_name_v in pages_name.items():
            name_path = Path(page_name_k)

            if Path.samefile(script_path, name_path):
                page_v["page_name"] = page_name_v

    st_on_pages_changed.send()


def read_readme() -> str:
    """
    Чтение файла README.md
    :return: Текст
    """
    text = Path("README.md").read_text(encoding='utf-8')
    return text[text.find('#'):]


def main_app() -> None:
    """
    Запуск основного приложения
    """

    st_page_rename({"run.py": "Главная страница",
                    "pages/page_one.py": "Генератор аудио",
                    "pages/page_two.py": "Описание изображения"})

    st.markdown(read_readme(), unsafe_allow_html=True)


# Запуск через streamlit
main_app()
