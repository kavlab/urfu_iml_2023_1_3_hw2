from fastapi import FastAPI
from transformers import pipeline

# Создание объекта FastAPI
app = FastAPI()
# Создание классификатора из библиотеки Hugging Face на основе пайплайна с типом "image-to-text"
classifier = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")


@app.get("/")
def root():
    """
    Приветственное сообщение
    """
    return {"message": "Follow the link http://127.0.0.1:8000/predict/ and get a description of the image"}


@app.get("/predict/")
async def predict():
    """
    Позволяет передать изображение 'image_result.jpg' для получения его описания
    """
    return classifier("image_result.jpg")[0]
