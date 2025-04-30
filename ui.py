import gradio as gr
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import torchaudio
from fuzzywuzzy import process
import pyttsx3
import re

# Assuming your KinyarwandaAssistant model is defined here
class KinyarwandaAssistant:
    # Your model code here, like in the previous steps

    # Create an instance of your assistant
    assistant = KinyarwandaAssistant()

    def transcribe_and_respond(audio):
        transcription = assistant.transcribe_audio(audio)
        response = assistant.get_response(transcription)
        assistant.speak_response(response)
        return transcription, response

    # Create the Gradio interface
    interface = gr.Interface(
        fn=transcribe_and_respond,
        inputs=gr.Audio(source="microphone", type="file"),
        outputs=[gr.Textbox(), gr.Textbox()],
        live=True,
        title="Kinyarwanda Assistant",
        description="A virtual assistant that understands and responds in Kinyarwanda."
    )

# Launch the Gradio interface
if __name__ == "__main__":
    interface.launch(share=True)  # share=True generates a public URL
