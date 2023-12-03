from fastapi import FastAPI
from pydantic import BaseModel
import text_to_speech_converter as t2s
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

@app.post('/text-to-speach/')
def text_to_speach(entity: Request) -> Response:
    """
    Text-to-audio generation method using text_to_speech_converter.
    """
    synthesiser = t2s.load_model()
    embeddings_dataset = t2s.load_speaker_dataset()
    audio, sampling_rate = t2s.text_to_speech(entity.text, synthesiser, embeddings_dataset)
    return Response(audio=base64.b32encode(audio), sampling_rate=sampling_rate)
