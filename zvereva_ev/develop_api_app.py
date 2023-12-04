from pydantic import BaseModel


class Url(BaseModel):
    """
    Формат ответа
    """
    url: str
