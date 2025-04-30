# from transformers import WhisperProcessor, WhisperForConditionalGeneration
# import torch
# import torchaudio
# from fuzzywuzzy import process
# import pyttsx3
# import re


# class KinyarwandaAssistant:

#     def __init__(self, model_path="benax-rw/KinyaWhisper"):
#         # Initialize ASR components
#         self.processor = WhisperProcessor.from_pretrained(model_path)
#         self.model = WhisperForConditionalGeneration.from_pretrained(model_path)

#         # Initialize Q&A knowledge base
#         self.qa_pairs = {
#             "umeze ute": "Ni meza, urakoze. Mbese uri umeze gute?",
#             "mwaramutse": "Mwaramutse! Mbega habaye iki?",
#             "Amakuru": "Ni meza ayawe?",
#             "umurwa mukuru wu rwanda ni uwuhe": "Umurwa mukuru ni Kigali.",
#             "fungura browser": "Ndabifite. Fungura browser...",
#         }

#         # Initialize TTS with Kinyarwanda-specific settings
#         self.tts_engine = self._init_tts()

#         # Kinyarwanda pronunciation rules
#         self.pronunciation_rules = [
#             (r"([aeiou])", r"\1"),  # Keep vowels pure
#             (r"rw", "ɾw"),  # Special handling for 'rw' combination
#             (r"cy", "tʃ"),  # 'cy' sound
#             (r"ny", "ɲ"),  # 'ny' as in Spanish 'ñ'
#             (r"ki", "ci"),  # 'ki' pronunciation
#             (r"gi", "ɟi"),  # 'gi' sound
#             (r"([^aeiou])h([aeiou])", r"\1\2"),  # Remove silent h's
#         ]

#     def _init_tts(self):
#         """Initialize TTS engine with Kinyarwanda-optimized settings"""
#         try:
#             engine = pyttsx3.init()

#             # Configure voice (try to find best match)
#             voices = engine.getProperty("voices")

#             # Voice selection priority
#             preferred_voices = [
#                 "kinyarwanda",
#                 "rwanda",
#                 "africa",
#                 "swahili",
#                 "kiswahili",
#                 "french",
#             ]

#             for voice in voices:
#                 if any(v in voice.name.lower() for v in preferred_voices):
#                     engine.setProperty("voice", voice.id)
#                     print(f"Using voice: {voice.name}")
#                     break

#             # TTS parameters optimized for Kinyarwanda
#             engine.setProperty("rate", 140)  # Slightly slower for clarity
#             engine.setProperty("volume", 0.9)
#             engine.setProperty("pitch", 48)  # Neutral pitch

#             return engine

#         except Exception as e:
#             print(f"TTS initialization error: {e}")
#             return None

#     def _kinyarwanda_pronunciation(self, text):
#         """Apply Kinyarwanda-specific pronunciation rules"""
#         for pattern, replacement in self.pronunciation_rules:
#             text = re.sub(pattern, replacement, text)
#         return text

#     def _preprocess_text(self, text):
#         """Clean text for matching"""
#         return text.lower().replace("?", "").replace("!", "").strip()

#     def _find_best_match(self, query, threshold=70):
#         """Fuzzy match against Q&A keys"""
#         query_clean = self._preprocess_text(query)
#         best_match, score = process.extractOne(query_clean, self.qa_pairs.keys())
#         return best_match if score > threshold else None

#     def get_response(self, query):
#         """Get answer for transcribed text"""
#         best_match = self._find_best_match(query)
#         return self.qa_pairs.get(best_match, "Ntabwo nzibishoboye.")

#     def transcribe_audio(self, audio_path):
#         """Convert speech to text"""
#         waveform, sample_rate = torchaudio.load(audio_path)

#         if sample_rate != 16000:
#             waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

#         inputs = self.processor(
#             waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt"
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

#     def speak_response(self, text):
#         """Convert text to speech with Kinyarwanda pronunciation"""
#         if not self.tts_engine:
#             print(f"[TTS not available]: {text}")
#             return

#         try:
#             # Apply pronunciation rules
#             pronounced_text = self._kinyarwanda_pronunciation(text)
#             print(f"Speaking: {pronounced_text}")

#             # Split long sentences for better prosody
#             sentences = re.split(r"([,.!?] )", pronounced_text)
#             for sentence in sentences:
#                 if sentence.strip():
#                     self.tts_engine.say(sentence)

#             self.tts_engine.runAndWait()

#         except Exception as e:
#             print(f"TTS error: {e}")
#             print(f"[TTS failed]: {text}")


# if __name__ == "__main__":
#     assistant = KinyarwandaAssistant()

#     try:
#         # Step 1: Transcribe audio
#         transcription = assistant.transcribe_audio("003.wav")
#         print(f"User: {transcription}")

#         # Step 2: Get response
#         response = assistant.get_response(transcription)
#         print(f"Assistant: {response}")

#         # Step 3: Speak the response
#         assistant.speak_response(response)

#     except Exception as e:
#         print(f"Error: {e}")
#         input("Press Enter to exit...")



# asr.py
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import torchaudio

class ASR:
    def __init__(self, model_path="benax-rw/KinyaWhisper"):
        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path)

    def transcribe_audio(self, audio_path):
        """Converts audio to text using Whisper"""
        waveform, sample_rate = torchaudio.load(audio_path)

        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

        inputs = self.processor(
            waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt"
        )
        inputs["attention_mask"] = torch.ones_like(inputs["input_features"][:, :, 0])

        generation_config = self.model.generation_config
        generation_config.forced_decoder_ids = None

        predicted_ids = self.model.generate(
            inputs.input_features,
            attention_mask=inputs["attention_mask"],
            max_new_tokens=10,
            no_repeat_ngram_size=1,
            generation_config=generation_config,
        )

        return self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
