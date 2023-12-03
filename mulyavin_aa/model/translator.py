from pydantic import BaseModel


class Request(BaseModel):
    """Структура запроса"""
    text: str


class Response(BaseModel):
    """Структура ответа"""
    text: str
