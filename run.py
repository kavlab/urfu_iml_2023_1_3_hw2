from transformers import pipeline
from datasets import load_dataset
import torch
import streamlit as st

@st.cache_resource
def load_speech_model():
    synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    return synthesiser, speaker_embedding

def speech_elements():
    synthesiser, speaker_embedding = load_speech_model()

    text = st.text_area('Enter English text here')
    st.write(f'You wrote {len(text)} characters.')

    if st.button('Speech'):
        speech = synthesiser(text, forward_params={"speaker_embeddings": speaker_embedding})

        st.audio(speech['audio'], sample_rate=speech['sampling_rate'])


speech_elements()
