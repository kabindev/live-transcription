import os
import threading
import queue
import time
import numpy as np
from datetime import datetime
import logging

# AI/ML Imports
import torch
import whisper
from pyannote.audio import Pipeline as DiarizationPipeline
import tensorflow as tf
import tensorflow_hub as hub
import google.generativeai as genai

# Custom Modules
from ui import TranscriptionUI
from utils import (
    AudioConfig,
    SpeakerDiarizationEngine,
    EventDetectionEngine,
    TranscriptionEngine,
    AIAssistant,
    get_audio_devices
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MainApp:
    def __init__(self, root):
        self.root = root
        self.config = AudioConfig()

        # --- Core Components ---
        self.audio_queue = queue.Queue(maxsize=50)
        self.is_recording = threading.Event()

        # --- AI/ML Engines ---
        self.whisper_model = self.load_whisper_model()
        self.diarization_pipeline = self.load_diarization_pipeline()
        self.yamnet_model, self.yamnet_labels = self.load_yamnet_model()

        # --- Processing Engines ---
        self.transcription_engine = TranscriptionEngine(self.whisper_model, self.config)
        self.diarization_engine = SpeakerDiarizationEngine(self.diarization_pipeline, self.config)
        self.event_engine = EventDetectionEngine(self.yamnet_model, self.yamnet_labels, self.config)
        self.ai_assistant = AIAssistant()

        # --- UI ---
        self.ui = TranscriptionUI(
            root,
            start_callback=self.start_capture,
            stop_callback=self.stop_capture,
            send_to_ai_callback=self.handle_ai_query,
            set_api_key_callback=self.ai_assistant.set_api_key
        )

        # --- Threads ---
        self.audio_thread = None
        self.processing_thread = None

        self.initialize_app()

    def initialize_app(self):
        """Initialize the application state and UI elements."""
        self.ui.update_status("App initialized. Ready to start.", "green")
        self.ui.populate_device_list(get_audio_devices())

        # Check for API keys and tokens
        if not os.environ.get("HUGGINGFACE_TOKEN"):
            self.ui.show_huggingface_token_dialog()

    def load_whisper_model(self):
        """Load the Whisper model."""
        try:
            logger.info("Loading Whisper model...")
            model = whisper.load_model("base.en")
            logger.info("Whisper model loaded successfully.")
            return model
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            self.ui.update_status(f"Error: Could not load Whisper model. {e}", "red")
            return None

    def load_diarization_pipeline(self):
        """Load the Pyannote.audio diarization pipeline."""
        try:
            hf_token = os.environ.get("HUGGINGFACE_TOKEN")
            if not hf_token:
                logger.warning("Hugging Face token not found. Diarization will be disabled.")
                return None

            logger.info("Loading speaker diarization pipeline...")
            pipeline = DiarizationPipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
            logger.info("Diarization pipeline loaded successfully.")
            return pipeline
        except Exception as e:
            logger.error(f"Failed to load diarization pipeline: {e}")
            self.ui.update_status(f"Error: Diarization disabled. {e}", "orange")
            return None

    def load_yamnet_model(self):
        """Load the YAMNet model for event detection."""
        try:
            logger.info("Loading YAMNet model...")
            model = hub.load('https://tfhub.dev/google/yamnet/1')
            labels = model.class_names.numpy().tolist()
            logger.info("YAMNet model loaded successfully.")
            return model, labels
        except Exception as e:
            logger.error(f"Failed to load YAMNet model: {e}")
            self.ui.update_status(f"Error: Event detection disabled. {e}", "orange")
            return None, None

    def start_capture(self):
        """Start the audio capture and processing threads."""
        if self.is_recording.is_set():
            logger.warning("Capture already in progress.")
            return

        self.is_recording.set()
        self.ui.update_status("Starting audio capture...", "blue")

        # Clear previous data
        self.ui.clear_transcript()
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()

        # Start threads
        self.audio_thread = threading.Thread(target=self.audio_capture_loop, daemon=True)
        self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)

        self.audio_thread.start()
        self.processing_thread.start()

        self.ui.set_recording_state(True)
        self.ui.update_status("Recording started.", "green")

    def stop_capture(self):
        """Stop the audio capture and processing."""
        if not self.is_recording.is_set():
            return

        self.is_recording.clear()

        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1)
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2)

        self.ui.set_recording_state(False)
        self.ui.update_status("Recording stopped.", "orange")

    def audio_capture_loop(self):
        """Continuously capture audio and put it into a queue."""
        import sounddevice as sd

        device_id = self.ui.get_selected_device_id()

        try:
            with sd.InputStream(
                samplerate=self.config.SAMPLE_RATE,
                channels=self.config.CHANNELS,
                dtype='float32',
                device=device_id,
                blocksize=self.config.CHUNK_SIZE,
                callback=self.audio_callback
            ) as stream:
                logger.info(f"Audio stream started on device {device_id}.")
                while self.is_recording.is_set():
                    time.sleep(0.1)
            logger.info("Audio stream stopped.")
        except Exception as e:
            logger.error(f"Audio capture error: {e}")
            self.ui.update_status(f"Error: {e}", "red")
            self.is_recording.clear()

    def audio_callback(self, indata, frames, time, status):
        """Callback function for the audio stream."""
        if status:
            logger.warning(f"Audio stream status: {status}")

        if self.is_recording.is_set():
            try:
                self.audio_queue.put_nowait(indata.copy())
            except queue.Full:
                pass  # Drop frames if the queue is full

    def processing_loop(self):
        """The main loop for processing audio from the queue."""
        audio_buffer = np.array([], dtype=np.float32)

        while self.is_recording.is_set():
            try:
                # Accumulate audio from the queue
                while not self.audio_queue.empty():
                    chunk = self.audio_queue.get_nowait()
                    audio_buffer = np.append(audio_buffer, chunk)

                # Process if we have enough audio
                if len(audio_buffer) >= self.config.PROCESSING_CHUNK_SIZE:
                    segment_to_process = audio_buffer[:self.config.PROCESSING_CHUNK_SIZE]
                    audio_buffer = audio_buffer[self.config.PROCESSING_CHUNK_SIZE:]

                    self.process_segment(segment_to_process)
                else:
                    time.sleep(0.1)  # Wait for more audio

            except Exception as e:
                logger.error(f"Processing loop error: {e}")
                time.sleep(1)

    def process_segment(self, audio_segment: np.ndarray):
        """Process a single segment of audio for all features."""
        timestamp = datetime.now()

        # --- Run all processing in parallel ---
        transcription_result = {}
        diarization_result = {}
        event_result = {}

        def run_transcription():
            nonlocal transcription_result
            transcription_result = self.transcription_engine.transcribe(audio_segment)

        def run_diarization():
            nonlocal diarization_result
            if self.diarization_engine.is_available():
                diarization_result = self.diarization_engine.diarize(audio_segment)

        def run_event_detection():
            nonlocal event_result
            if self.event_engine.is_available():
                event_result = self.event_engine.detect(audio_segment)

        # Start threads
        t_transcribe = threading.Thread(target=run_transcription)
        t_diarize = threading.Thread(target=run_diarization)
        t_events = threading.Thread(target=run_event_detection)

        t_transcribe.start()
        t_diarize.start()
        t_events.start()

        # Wait for them to complete
        t_transcribe.join()
        t_diarize.join()
        t_events.join()

        # --- Combine and display results ---
        transcript_text = transcription_result.get("text", "")
        speaker_label = diarization_result.get("speaker", "SPEAKER_00")
        events = event_result.get("events", [])

        # Update UI
        if transcript_text or events:
            self.root.after(0, self.ui.update_transcript, {
                "timestamp": timestamp,
                "speaker": speaker_label,
                "text": transcript_text,
                "events": events
            })

            # Update overlay
            if transcript_text:
                self.root.after(0, self.ui.update_overlay, f"{speaker_label}: {transcript_text}")

    def handle_ai_query(self, query: str):
        """Handle a query from the user to the AI assistant."""
        if not self.ai_assistant.is_ready():
            self.ui.add_ai_response("AI Assistant is not configured. Please set your API key.")
            return

        full_transcript = self.ui.get_full_transcript()

        self.ui.add_ai_response("Thinking...")

        def do_ai_query():
            try:
                response = self.ai_assistant.query(query, full_transcript)
                self.root.after(0, self.ui.update_ai_response, response)
            except Exception as e:
                logger.error(f"AI Assistant error: {e}")
                self.root.after(0, self.ui.update_ai_response, f"An error occurred: {e}")

        threading.Thread(target=do_ai_query, daemon=True).start()

    def on_closing(self):
        """Handle the application closing event."""
        logger.info("Application closing...")
        self.stop_capture()
        self.root.destroy()

if __name__ == "__main__":
    import tkinter as tk

    root = tk.Tk()
    app = MainApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()