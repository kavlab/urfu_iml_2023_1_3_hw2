from pydantic import BaseModel


class TranslatorRequest(BaseModel):
    text: str


class TranslatorResponse(BaseModel):
    text: str
