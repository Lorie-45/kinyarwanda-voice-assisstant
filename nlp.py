# from transformers import WhisperProcessor, WhisperForConditionalGeneration
# import torch
# import torchaudio
# from fuzzywuzzy import process

# class KinyarwandaAssistant:
#     def __init__(self, model_path="benax-rw/KinyaWhisper"):
#         # Initialize ASR components
#         self.processor = WhisperProcessor.from_pretrained(model_path)
#         self.model = WhisperForConditionalGeneration.from_pretrained(model_path)

#         # Initialize Q&A knowledge base
#         self.qa_pairs = {
#             "umeze ute": "Ni meza, urakoze. Mbese uri umeze gute?",
#             "mwaramutse": "Mwaramutse! Mbega habaye iki?",
#             "Amakuru": "Ni meza ayawe? ",
#             "umurwa mukuru wu rwanda ni uwuhe": "Umurwa mukuru ni Kigali.",
#             "fungura browser": "Ndabifite. Fungura browser..."
#         }

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

#         # Resample if needed
#         if sample_rate != 16000:
#             waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

#         # Prepare inputs
#         inputs = self.processor(
#             waveform.squeeze().numpy(),
#             sampling_rate=16000,
#             return_tensors="pt"
#         )
#         inputs["attention_mask"] = torch.ones_like(inputs["input_features"][:, :, 0])

#         # Generate transcription
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

# # ==================== Usage ====================
# if __name__ == "__main__":
#     assistant = KinyarwandaAssistant()

#     # Step 1: Transcribe audio
#     transcription = assistant.transcribe_audio("003.wav")
#     print(f"User: {transcription}")

#     # Step 2: Get response
#     response = assistant.get_response(transcription)
#     print(f"Assistant: {response}")



# nlp.py
from fuzzywuzzy import process

class NLP:
    def __init__(self):
        self.qa_pairs = {
            "umeze ute": "Ni meza, urakoze. Mbese uri umeze gute?",
            "mwaramutse": "Mwaramutse! Mbega habaye iki?",
            "Amakuru": "Ni meza ayawe?",
            "umurwa mukuru wu rwanda ni uwuhe": "Umurwa mukuru ni Kigali.",
            "fungura browser": "Ndabifite. Fungura browser...",
        }

    def _preprocess_text(self, text):
        """Clean text for matching"""
        return text.lower().replace("?", "").replace("!", "").strip()

    def _find_best_match(self, query, threshold=70):
        """Fuzzy match against Q&A keys"""
        query_clean = self._preprocess_text(query)
        best_match, score = process.extractOne(query_clean, self.qa_pairs.keys())
        return best_match if score > threshold else None

    def get_response(self, query):
        """Get answer for transcribed text"""
        best_match = self._find_best_match(query)
        return self.qa_pairs.get(best_match, "Ntabwo nzibishoboye.")
