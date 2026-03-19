"""
Text-to-Speech Service — Multi-language, cached, audio bytes output.

Supports:
  - gTTS (offline-capable with cache)
  - pyttsx3 (fully offline)
  - Azure Cognitive Services (cloud, highest quality)

Audio is LRU-cached by text+language so repeated announcements don't re-synthesise.
"""

import asyncio
import base64
import hashlib
import io
import logging
import time
from functools import lru_cache
from typing import Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


class TTSService:
    """
    Singleton TTS service. Produces base64-encoded MP3/WAV audio bytes
    suitable for WebSocket streaming to the mobile client.
    """
    _instance: Optional["TTSService"] = None

    def __init__(self):
        self._cache: dict = {}     # text_hash → base64 audio str
        self._lock = asyncio.Lock()
        logger.info("✅ TTSService ready (engine=%s)", settings.TTS_ENGINE)

    @classmethod
    def get_instance(cls) -> "TTSService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def synthesize(
        self,
        text: str,
        lang: str = "en",
        slow: bool = False,
    ) -> Optional[str]:
        """
        Synthesize speech. Returns base64-encoded MP3 audio string.

        Args:
            text: text to speak
            lang: language code ("en", "hi")
            slow: use slow rate (useful for critical warnings)

        Returns:
            base64 str of MP3 bytes, or None on failure
        """
        if not text.strip():
            return None

        cache_key = hashlib.md5(f"{text}{lang}{slow}".encode()).hexdigest()

        async with self._lock:
            if cache_key in self._cache:
                logger.debug("TTS cache hit: %s", cache_key[:8])
                return self._cache[cache_key]

        audio_b64 = await asyncio.get_event_loop().run_in_executor(
            None, self._synthesize_sync, text, lang, slow
        )

        if audio_b64:
            async with self._lock:
                # Bounded LRU — evict oldest if cache too large
                if len(self._cache) > 200:
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                self._cache[cache_key] = audio_b64

        return audio_b64

    def _synthesize_sync(self, text: str, lang: str, slow: bool) -> Optional[str]:
        """Synchronous TTS synthesis — runs in thread pool."""
        engine = settings.TTS_ENGINE

        try:
            if engine == "gtts":
                return self._gtts(text, lang, slow)
            elif engine == "pyttsx3":
                return self._pyttsx3(text)
            elif engine == "azure":
                return self._azure(text, lang)
            else:
                logger.error("Unknown TTS engine: %s", engine)
                return None
        except Exception as exc:
            logger.error("TTS synthesis failed (%s): %s", engine, exc)
            return None

    @staticmethod
    def _gtts(text: str, lang: str, slow: bool) -> Optional[str]:
        try:
            from gtts import gTTS
            tts = gTTS(text=text, lang=lang, slow=slow)
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            buf.seek(0)
            return base64.b64encode(buf.read()).decode("utf-8")
        except ImportError:
            logger.warning("gTTS not installed — pip install gtts")
            return None

    @staticmethod
    def _pyttsx3(text: str) -> Optional[str]:
        """Fully offline TTS via pyttsx3. Returns WAV base64."""
        try:
            import pyttsx3
            import tempfile, os
            engine = pyttsx3.init()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
            engine.save_to_file(text, tmp_path)
            engine.runAndWait()
            with open(tmp_path, "rb") as f:
                audio = base64.b64encode(f.read()).decode("utf-8")
            os.unlink(tmp_path)
            return audio
        except ImportError:
            logger.warning("pyttsx3 not installed — pip install pyttsx3")
            return None

    @staticmethod
    def _azure(text: str, lang: str) -> Optional[str]:
        """Azure Cognitive Services TTS — high quality, requires AZURE_TTS_KEY env var."""
        import os
        key = os.getenv("AZURE_TTS_KEY")
        region = os.getenv("AZURE_TTS_REGION", "eastus")
        if not key:
            logger.warning("AZURE_TTS_KEY not set")
            return None
        try:
            import azure.cognitiveservices.speech as speechsdk
            voice_map = {"en": "en-IN-NeerjaNeural", "hi": "hi-IN-SwaraNeural"}
            voice = voice_map.get(lang, "en-IN-NeerjaNeural")
            config = speechsdk.SpeechConfig(subscription=key, region=region)
            config.speech_synthesis_voice_name = voice
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=config, audio_config=None)
            result = synthesizer.speak_text_async(text).get()
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                return base64.b64encode(result.audio_data).decode("utf-8")
        except ImportError:
            logger.warning("Azure SDK not installed")
        return None

    def cleanup(self):
        self._cache.clear()
