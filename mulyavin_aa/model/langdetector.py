from pydantic import BaseModel


class Request(BaseModel):
    """Структура запроса"""
    text: str


class LangInfo(BaseModel):
    """Информация об определении языка"""
    label: str
    score: float


class Response(BaseModel):
    """Структура ответа"""
    langs: list[LangInfo]
