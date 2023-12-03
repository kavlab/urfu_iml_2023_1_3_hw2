from fastapi import FastAPI
from pydantic import BaseModel
import kuznetsov_av.text_to_speech_converter as t2s
import base64

class Request(BaseModel):
    """
    Input text.
    """
    text: str

class Response(BaseModel):
    """
    Result of text-to-audio generation.
    audio - base64 string
    """
    audio: str
    sampling_rate: int


app = FastAPI()

@app.get('/')
async def root() -> str:
    """
    Root method of API.
    """
    return '{"message": "Converter method: /text-to-speech/convert/"}'

@app.post('/text-to-speech/convert/')
async def text_to_speech(entity: Request) -> Response:
    """
    Text-to-audio generation method using text_to_speech_converter.
    """
    synthesiser = t2s.load_model()
    embeddings_dataset = t2s.load_speaker_dataset()
    audio, sampling_rate = t2s.text_to_speech(entity.text, synthesiser, embeddings_dataset)
    return Response(audio=base64.b32encode(audio), sampling_rate=sampling_rate)
