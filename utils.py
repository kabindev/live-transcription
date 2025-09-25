import numpy as np
import os
import logging
from typing import Dict, List, Optional, Tuple
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioConfig:
    """Configuration for audio processing."""
    SAMPLE_RATE = 16000
    CHANNELS = 1
    CHUNK_DURATION_MS = 20  # Duration of a single audio chunk in ms
    CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)

    # Processing chunk is how much audio we feed to the models at once
    PROCESSING_CHUNK_SECONDS = 3
    PROCESSING_CHUNK_SIZE = int(SAMPLE_RATE * PROCESSING_CHUNK_SECONDS)

def get_audio_devices() -> List[Dict]:
    """Get a list of available audio input devices."""
    import sounddevice as sd
    devices = sd.query_devices()
    input_devices = []
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            input_devices.append({'id': i, 'name': device['name']})
    return input_devices


class TranscriptionEngine:
    """Handles audio transcription using Whisper."""
    def __init__(self, model, config: AudioConfig):
        self.model = model
        self.config = config

    def transcribe(self, audio_data: np.ndarray) -> Dict:
        """Transcribe audio using the Whisper model."""
        if not self.model:
            return {"text": "[Whisper model not loaded]"}

        try:
            # Whisper expects float32 audio data
            audio_float32 = audio_data.astype(np.float32)

            # Use the Whisper model to transcribe the audio
            result = self.model.transcribe(
                audio_float32,
                fp16=torch.cuda.is_available()  # Use GPU if available
            )
            return {"text": result.get("text", "").strip()}
        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            return {"text": f"[Transcription Error: {e}]"}


class SpeakerDiarizationEngine:
    """Handles speaker diarization using pyannote.audio."""
    def __init__(self, pipeline, config: AudioConfig):
        self.pipeline = pipeline
        self.config = config
        self.last_speaker = "SPEAKER_00"

    def is_available(self) -> bool:
        """Check if the diarization pipeline is loaded and authenticated."""
        return self.pipeline is not None

    def diarize(self, audio_data: np.ndarray) -> Dict:
        """
        Perform speaker diarization on an audio segment in memory.

        This updated method avoids writing to temporary files, which is more
        efficient and reliable. It processes the audio as a PyTorch tensor.
        """
        if not self.is_available():
            return {"speaker": self.last_speaker}

        try:
            # Convert numpy array to a PyTorch tensor and add a channel dimension
            audio_tensor = torch.from_numpy(audio_data).float().unsqueeze(0)

            # Create the input format expected by pyannote.audio
            input_data = {"waveform": audio_tensor, "sample_rate": self.config.SAMPLE_RATE}

            # Perform diarization
            diarization = self.pipeline(input_data)

            # Determine the dominant speaker in the segment
            # The timeline is a list of (segment, track, speaker_label)
            if diarization.get_timeline():
                # Get the speaker who spoke for the longest duration in the segment
                dominant_speaker = diarization.get_timeline().support().next()[2]

                # Only update if the speaker has changed to avoid redundant updates
                if dominant_speaker != self.last_speaker:
                    self.last_speaker = dominant_speaker

            return {"speaker": self.last_speaker}
        except Exception as e:
            # Log the error but don't crash; return the last known speaker
            logger.error(f"Diarization error: {e}")
            return {"speaker": self.last_speaker}


class EventDetectionEngine:
    """Handles non-speech event detection using YAMNet."""
    def __init__(self, model, labels: List[str], config: AudioConfig):
        self.model = model
        self.labels = labels
        self.config = config
        self.confidence_threshold = 0.4  # Stricter threshold for higher accuracy

        # A more refined mapping of YAMNet labels to user-friendly emojis
        self.event_mappings = {
            'music': '🎵', 'musical instrument': '🎵',
            'laughter': '😂', 'giggle': '😂',
            'clapping': '👏', 'applause': '👏',
            'cough': '😷', 'sneeze': '🤧',
            'typing': '⌨️',
            'door': '🚪', 'knock': '🚪',
            'telephone': '📞', 'alarm': '🚨', 'siren': '🚨',
            # We explicitly ignore 'speech' as it's handled by the transcription engine
            'speech': None
        }

    def is_available(self) -> bool:
        """Check if the event detection model is loaded."""
        return self.model is not None and self.labels is not None

    def detect(self, audio_data: np.ndarray) -> Dict:
        """
        Detect non-speech events in an audio segment with improved accuracy.

        This version uses a higher confidence threshold and a more robust mapping
        to reduce noise and improve the quality of detected events.
        """
        if not self.is_available():
            return {"events": []}

        try:
            # Get model predictions and average scores over the segment
            scores, _, _ = self.model(audio_data)
            scores = scores.numpy().mean(axis=0)

            detected_events = []
            # Get the top 5 most likely predictions
            top_indices = np.argsort(scores)[::-1][:5]

            processed_events = set()

            for i in top_indices:
                # Only consider predictions above our confidence threshold
                if scores[i] > self.confidence_threshold:
                    class_name = self.labels[i].lower()

                    # Match the detected class to our event map
                    for keyword, emoji in self.event_mappings.items():
                        if keyword in class_name and emoji and keyword not in processed_events:
                            detected_events.append({"event": keyword.title(), "emoji": emoji})
                            processed_events.add(keyword) # Ensure we don't add the same event type twice
                            break

            return {"events": detected_events}
        except Exception as e:
            logger.error(f"Event detection error: {e}")
            return {"events": []}

class AIAssistant:
    """Handles interaction with the Gemini AI model."""
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.model = None
        if self.api_key:
            self.configure_genai()

    def set_api_key(self, key: str):
        """Sets the API key and configures the model."""
        import google.generativeai as genai
        self.api_key = key
        os.environ["GEMINI_API_KEY"] = key
        self.configure_genai()

    def configure_genai(self):
        """Configures the generative AI model."""
        import google.generativeai as genai
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            logger.info("Gemini AI Assistant configured successfully.")
        except Exception as e:
            logger.error(f"Failed to configure Gemini: {e}")
            self.model = None

    def is_ready(self) -> bool:
        return self.model is not None

    def query(self, user_query: str, context: str) -> str:
        """Sends a query to the AI with the provided context."""
        if not self.is_ready():
            return "AI model not configured. Please set your API key."

        prompt = f"""
        You are an AI assistant integrated into a live transcription application.
        Your task is to answer questions based on the provided transcript of a conversation.

        Here is the live transcript so far:
        ---
        {context}
        ---

        Based on this transcript, please answer the following question:

        Question: "{user_query}"

        Answer:
        """

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini API query failed: {e}")
            return f"An error occurred while communicating with the AI: {e}"