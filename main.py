# from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
# import torch
# import torchaudio
# from fuzzywuzzy import process
# import soundfile as sf
# import numpy as np

# class KinyarwandaAssistant:
#     def __init__(self, model_path="benax-rw/KinyaWhisper", tts_model="facebook/mms-tts-kin"):
#         # ASR (Speech to Text)
#         self.processor = WhisperProcessor.from_pretrained(model_path)
#         self.model = WhisperForConditionalGeneration.from_pretrained(model_path)

#         # TTS (Text to Speech)
#         self.tts = pipeline("text-to-speech", model=tts_model)

#         # Q&A Base
#         self.qa_pairs = {
#             "umeze ute": "Ni meza, urakoze. Mbese uri umeze gute?",
#             "mwaramutse": "Mwaramutse! Mbega habaye iki?",
#             "amakuru": "Ni meza ayawe?",
#             "umurwa mukuru wu rwanda ni uwuhe": "Umurwa mukuru ni Kigali.",
#             "fungura browser": "Ndabifite. Fungura browser..."
#         }

#     def _preprocess_text(self, text):
#         return text.lower().replace("?", "").replace("!", "").strip()

#     def _find_best_match(self, query, threshold=70):
#         query_clean = self._preprocess_text(query)
#         best_match, score = process.extractOne(query_clean, self.qa_pairs.keys())
#         return best_match if score > threshold else None

#     def get_response(self, query):
#         best_match = self._find_best_match(query)
#         return self.qa_pairs.get(best_match, "Ntabwo nzibishoboye.")

#     def transcribe_audio(self, audio_path):
#         waveform, sample_rate = torchaudio.load(audio_path)

#         if sample_rate != 16000:
#             waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

#         inputs = self.processor(
#             waveform.squeeze().numpy(),
#             sampling_rate=16000,
#             return_tensors="pt"
#         )
#         inputs["attention_mask"] = torch.ones_like(inputs["input_features"][:, :, 0])
#         generation_config = self.model.generation_config
#         generation_config.forced_decoder_ids = None

#         predicted_ids = self.model.generate(
#             inputs.input_features,
#             attention_mask=inputs["attention_mask"],
#             max_new_tokens=10,
#             no_repeat_ngram_size=1,
#             generation_config=generation_config,
#         )

#         return self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

#     def synthesize_speech(self, text):
#         output = self.tts(text)
#         print("TTS output keys:", output.keys())
#         print("TTS output types:", {k: type(v) for k, v in output.items()})

#         audio = output.get("audio", None)
#         if audio is None:
#             audio = output.get("array", None)
#         if audio is None:
#             audio = output.get("wav", None)

#         import numpy as np
#         print("Audio type:", type(audio))
#         print("Audio shape:", getattr(audio, 'shape', 'Not a NumPy array'))
#         print("First 10 audio samples:", audio[:10] if hasattr(audio, '__getitem__') else "Not indexable")
#         print("Audio max:", np.max(audio) if isinstance(audio, np.ndarray) else "N/A")
#         print("Audio min:", np.min(audio) if isinstance(audio, np.ndarray) else "N/A")

#         output_path = "response.wav"
#         sf.write(output_path, audio, 22050)
#         print("Audio written to", output_path)





# # ==================== Usage ====================
# if __name__ == "__main__":
#     assistant = KinyarwandaAssistant()

#     transcription = assistant.transcribe_audio("003.wav")
#     print(f"User: {transcription}")

#     response = assistant.get_response(transcription)
#     print(f"Assistant: {response}")

#     assistant.synthesize_speech(response)
# main.py
from asr import ASR
from tts import TTS
from nlp import NLP
import gradio as gr

class Assistant:
    def __init__(self):
        self.asr = ASR()
        self.tts = TTS()
        self.nlp = NLP()

    def transcribe_and_respond(self, audio):
        transcription = self.asr.transcribe_audio(audio)
        response = self.nlp.get_response(transcription)
        self.tts.speak_response(response)
        return transcription, response

assistant = Assistant()

# Create the Gradio interface
interface = gr.Interface(
    fn=assistant.transcribe_and_respond,
    inputs=gr.Audio(source="microphone", type="file"),
    outputs=[gr.Textbox(), gr.Textbox()],
    live=True,
    title="Kinyarwanda Assistant",
    description="A virtual assistant that understands and responds in Kinyarwanda."
)

if __name__ == "__main__":
    interface.launch(share=True)
