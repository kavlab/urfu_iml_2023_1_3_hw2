import datasets.arrow_dataset
import numpy as np
import torch
import transformers.pipelines.text_to_audio
from datasets import load_dataset
from transformers import pipeline


def load_model() -> transformers.pipelines.text_to_audio.TextToAudioPipeline:
    """
    Подгрузка модели преобразования текста в речь
    :return: class TextToAudioPipeline
    """
    return pipeline("text-to-speech", "microsoft/speecht5_tts")


def load_speaker_dataset() -> datasets.arrow_dataset.Dataset:
    """
    Подгрузка датасета для озвучивания текста
    :return: class Dataset
    """
    return load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")


def text_to_speech(
        text: str,
        synthesiser: transformers.pipelines.text_to_audio.TextToAudioPipeline,
        embeddings_dataset: datasets.arrow_dataset.Dataset
        ) -> (np.ndarray, int):
    """
    Преобразование текста в речь
    :param text: Текст
    :param synthesiser: pipeline для озвучивания текста
    :param embeddings_dataset: dataset для озвучивания текста
    :return: tuple (audio data, sampling rate)
    """
    speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    speech = synthesiser(text, forward_params={"speaker_embeddings": speaker_embedding})

    return speech['audio'], speech['sampling_rate']
