# Erweiterte KI Chat GUI - Vollversion mit allen Verbesserungen
# Version 1.0 - Professionelle Implementation

import asyncio
import base64
import time
import threading
import queue
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import re
from collections import Counter
from PIL import Image, ImageTk, ImageDraw
import queue
import time
from database_manager import DatabaseManager
from performance_monitor import PerformanceMonitor
import configparser
import hashlib
import importlib.util
import json
import logging
import logging.handlers
import os
import queue
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import torch
import torchaudio
import sounddevice as sd
import numpy as np
from gtts import gTTS
import soundfile as sf
import uuid
import tkinter as tk
import traceback
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from queue import Queue
from threading import Thread
from tkinter import filedialog, messagebox, scrolledtext, ttk
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union, Callable

# Externe Abhängigkeiten
import yaml
import customtkinter

# Drittanbieter-Importe
import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox, filedialog, ttk, scrolledtext, simpledialog, StringVar, IntVar, Toplevel, Menu
import pyttsx3
import speech_recognition as sr
from cryptography.fernet import Fernet
from PIL import Image, ImageTk, ImageDraw
from llama_cpp import Llama

# JARVIS GUI Integration
from jarvis_theme import JarvisTheme

# Lokale Module
from config_manager import ConfigManager
from context_scorer import ContextScorer
from database_manager import DatabaseManager
from datensatz_manager import DatensatzManager
from encryption_manager import EncryptionManager
from long_term_memory import LongTermMemory
from model_manager import EnhancedModelManager
from performance_monitor import PerformanceMonitor
from plugin_manager import PluginManager
from speech_manager import SpeechManager
from wissens_api import WissensAPI

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ki_chat.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === ENUMS & DATACLASSES ===


class ModelType(Enum):
    CODE = "code"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    GENERAL = "general"
    EMOTIONAL = "emotional"


class Theme(Enum):
    LIGHT = "light"
    DARK = "dark"
    SYSTEM = "system"


@dataclass
class ModelConfig:
    name: str
    path: str
    model_type: ModelType
    chat_format: str
    n_ctx: int = 8192
    n_threads: int = 8
    n_gpu_layers: int = 35
    temperature: float = 0.7
    max_tokens: int = 8192


@dataclass
class UserProfile:
    username: str
    display_name: str
    preferences: Dict[str, Any]
    created_at: datetime
    last_active: datetime


@dataclass
class ChatMessage:
    role: str
    content: str
    timestamp: datetime
    model_used: str = ""
    tokens_used: int = 0
    generation_time: float = 0.0

    def to_dict(self):
        """Konvertiert die ChatMessage in ein Dictionary.

        Returns:
            dict: Ein Dictionary mit den Nachrichtendaten
        """
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp.isoformat() if isinstance(
                self.timestamp,
                datetime) else self.timestamp,
            'model_used': self.model_used,
            'tokens_used': self.tokens_used,
            'generation_time': self.generation_time}


# === ADVANCED MODEL SELECTOR ===


class SmartModelSelector:
    def __init__(self):
        self.model_patterns = {
            ModelType.CODE: [
                r'\b(code|programming|programmieren|python|javascript|html|css|sql|function|class|variable|loop|if|else|debug|error|syntax)\b',
                r'\b(algorithm|datenstruktur|objektorientiert|framework|library|api|database|web development)\b'],
            ModelType.CREATIVE: [
                r'\b(story|geschichte|gedicht|poem|creative|kreativ|fiction|fantasy|novel|character|plot|schreiben|autor)\b',
                r'\b(inspiration|brainstorming|idee|imagination|artistic|kunst|design|musik)\b'],
            ModelType.ANALYTICAL: [
                r'\b(analyze|analysiere|data|statistics|statistik|research|forschung|study|wissenschaft|logic|logik)\b',
                r'\b(compare|vergleich|evaluate|bewerte|conclusion|schlussfolgerung|evidence|beweis)\b'],
            ModelType.EMOTIONAL: [
                r'\b(feel|fühle|emotion|gefühl|sad|traurig|happy|glücklich|angry|wütend|love|liebe|fear|angst)\b',
                r'\b(beziehung|relationship|family|familie|friend|freund|support|unterstützung|advice|rat)\b']}

        self.complexity_indicators = [
            r'\b(explain|erkläre|how|wie|why|warum|what|was|detailed|detailliert|comprehensive|umfassend)\b',
            r'\b(step by step|schritt für schritt|tutorial|guide|anleitung|examples|beispiele)\b']

    def select_model(
            self,
            prompt: str,
            conversation_history: List[ChatMessage] = None) -> ModelType:
        prompt_lower = prompt.lower()

        # Kontext aus Gesprächsverlauf berücksichtigen
        context_score = self._analyze_context(conversation_history)

        # Pattern-Matching für verschiedene Modelltypen
        type_scores = {}
        for model_type, patterns in self.model_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, prompt_lower, re.IGNORECASE))
                score += matches
            type_scores[model_type] = score

        # Komplexitätsbewertung
        complexity = self._assess_complexity(prompt)

        # Bestes Modell auswählen
        if max(type_scores.values()) == 0:
            return ModelType.GENERAL

        best_type = max(type_scores, key=type_scores.get)

        # Kontext-Anpassung
        if context_score:
            for context_type, score in context_score.items():
                if score > 0.3:  # Threshold für Kontext-Einfluss
                    type_scores[context_type] += score * 2

        return max(type_scores, key=type_scores.get)

    def _analyze_context(
            self, history: List[ChatMessage]) -> Dict[ModelType, float]:
        if not history:
            return {}

        recent_messages = history[-6:]  # Letzten 6 Nachrichten
        context_scores = {model_type: 0.0 for model_type in ModelType}

        for i, message in enumerate(recent_messages):
            # Neuere Nachrichten haben mehr Gewicht
            weight = (i + 1) / len(recent_messages)
            content_lower = message.content.lower()

            for model_type, patterns in self.model_patterns.items():
                for pattern in patterns:
                    matches = len(
                        re.findall(
                            pattern,
                            content_lower,
                            re.IGNORECASE))
                    context_scores[model_type] += matches * weight

        # Normalisierung
        max_score = max(context_scores.values()
                        ) if context_scores.values() else 1
        if max_score > 0:
            context_scores = {
                k: v / max_score for k,
                v in context_scores.items()}

        return context_scores

    def _assess_complexity(self, prompt: str) -> float:
        """Bewertung der Prompt-Komplexität (0-1)"""
        factors = {
            'length': min(len(prompt.split()) / 100, 1.0),
            'questions': len(re.findall(r'[?]', prompt)) * 0.1,
            'complexity_words': len(re.findall('|'.join(self.complexity_indicators), prompt.lower())) * 0.2
        }
        return min(sum(factors.values()), 1.0)

# === DYNAMIC TOKEN CALCULATOR ===


class TokenCalculator:
    def __init__(self):
        self.base_multiplier = 1.33  # Durchschnittliche Tokens pro Wort
        self.complexity_multipliers = {
            'simple': 1.0,
            'medium': 1.5,
            'complex': 2.0,
            'very_complex': 3.0
        }

    def calculate_optimal_tokens(self,
                                 prompt: str,
                                 model_type: ModelType,
                                 conversation_length: int = 0) -> Tuple[int,
                                                                        str]:
        # Grundlegende Token-Schätzung
        estimated_input_tokens = int(
            len(prompt.split()) * self.base_multiplier)

        # Komplexitätsbewertung
        complexity = self._assess_complexity(prompt)

        # Modelltyp-spezifische Anpassungen
        type_multipliers = {
            ModelType.CODE: 1.8,      # Code braucht mehr Details
            ModelType.CREATIVE: 2.2,   # Kreative Texte sind länger
            ModelType.ANALYTICAL: 1.6,  # Analysen brauchen Struktur
            ModelType.EMOTIONAL: 1.3,  # Emotionale Antworten sind persönlicher
            ModelType.GENERAL: 1.0
        }

        # Basis-Token-Berechnung
        base_tokens = max(100, estimated_input_tokens *
                          0.8)  # Reduzierte Mindestanzahl

        # Anwendung der Multiplikatoren
        final_tokens = int(
            base_tokens *
            type_multipliers[model_type] *
            complexity)

        # Maximale Token-Anzahl basierend auf dem Kontextfenster
        # Standard-Kontextfenster: 2048, aber sicherstellen, dass wir nicht zu
        # viel anfordern
        context_window = 2048  # Standardwert, falls nicht anders spezifiziert

        # Sicherstellen, dass wir nicht mehr als 80% des Kontextfensters
        # belegen
        safe_max_tokens = int(context_window * 0.8) - conversation_length

        # Endgültige Token-Anzahl begrenzen
        max_tokens = min(final_tokens, safe_max_tokens)
        max_tokens = max(max_tokens, 32)  # Mindestanzahl an Tokens

        # Sicherstellen, dass max_tokens nicht negativ ist
        max_tokens = max(32, max_tokens)

        # Erklärung generieren
        explanation = (f"Input: ~{estimated_input_tokens} tokens, "
                       f"Komplexität: {complexity:.1f}, "
                       f"Typ: {model_type.value}, "
                       f"Kontextfenster: {context_window}, "
                       f"Sichere Maximaltokens: {safe_max_tokens}, "
                       f"Output: {max_tokens} tokens")

        return max_tokens, explanation

    def _assess_complexity(self, prompt: str) -> float:
        complexity_indicators = {
            'length': min(
                len(prompt) / 500,
                1.0),
            'questions': min(
                prompt.count('?') * 0.2,
                1.0),
            'keywords': len(
                re.findall(
                    r'\b(explain|analyze|compare|detail|comprehensive|step-by-step)\b',
                    prompt.lower())) * 0.3,
            'technical': len(
                re.findall(
                    r'\b(algorithm|framework|implementation|optimization|architecture)\b',
                    prompt.lower())) * 0.4}

        total_complexity = sum(complexity_indicators.values())
        return min(max(total_complexity, 0.5), 3.0)  # Zwischen 0.5 und 3.0

# === PERFORMANCE MONITOR ===


# Die PerformanceMonitor-Klasse wurde in die Datei performance_monitor.py
# ausgelagert

# === SPEECH INTEGRATION ===


class SpeechManager:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.tts_engine = pyttsx3.init()
        self.is_listening = False
        self.setup_tts()

    def setup_tts(self):
        voices = self.tts_engine.getProperty('voices')
        # Deutsche Stimme bevorzugen
        for voice in voices:
            if 'german' in voice.name.lower() or 'deutsch' in voice.name.lower():
                self.tts_engine.setProperty('voice', voice.id)
                break

        self.tts_engine.setProperty('rate', 180)
        self.tts_engine.setProperty('volume', 0.8)

    def listen_for_speech(self, callback=None):
        """Startet die Spracherkennung im Hintergrund."""
        if self.is_listening:
            return

        self.is_listening = True

        def recognize_thread():
            try:
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source)
                    audio = self.recognizer.listen(
                        source, timeout=5, phrase_time_limit=10)

                    try:
                        text = self.recognizer.recognize_google(
                            audio, language="de-DE")
                        if callback and callable(callback):
                            callback(text)
                    except sr.UnknownValueError:
                        if callback and callable(callback):
                            callback("")
                    except sr.RequestError as e:
                        print(f"Spracherkennungsfehler: {e}")
                        if callback and callable(callback):
                            callback("")
            except Exception as e:
                print(f"Fehler bei der Spracherkennung: {e}")
            finally:
                self.is_listening = False

        threading.Thread(target=recognize_thread, daemon=True).start()


class XTTSManager:
    """Verwaltet die XTTS-v2 Sprachsynthese."""

    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.path.join("models", "xtts_v2")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts = None
        self.voice_samples = {}
        self._load_xtts_model()
        self._load_voice_samples()

    def _load_xtts_model(self) -> bool:
        """Lädt das XTTS-v2 Modell."""
        try:
            model_path = os.path.join(self.config_path, "model.pth")
            if not os.path.exists(model_path):
                logger.warning(
                    f"XTTS Modell nicht gefunden unter: {model_path}")
        except Exception as e:
            logger.warning(f"Fehler beim Laden des XTTS Modells: {e}")

        # Initialisiere TTS (deaktiviert XTTS)
        self.xtts_manager = None
        logger.info("XTTS ist deaktiviert, verwende gTTS für die Sprachausgabe")
        return False

    def _load_voice_samples(self):
        """Lädt verfügbare Sprachproben."""
        voice_dir = os.path.join("stimmen")
        if not os.path.exists(voice_dir):
            os.makedirs(voice_dir, exist_ok=True)
            return

        for file in os.listdir(voice_dir):
            if file.endswith(".wav"):
                name = os.path.splitext(file)[0]
                self.voice_samples[name] = os.path.join(voice_dir, file)
                logger.info(f"Sprachprobe geladen: {name}")

    def text_to_speech(
            self,
            text: str,
            voice_name: str = "Jarvis",
            language: str = "de",
            speed: float = 1.0) -> Optional[str]:
        """Konvertiert Text in Sprache.

        Args:
            text: Der zu synthetisierende Text
            voice_name: Name der Stimme (muss im stimmen-Ordner vorhanden sein)
            language: Sprache ('de' für Deutsch)
            speed: Sprechgeschwindigkeit (0.5-2.0)

        Returns:
            Pfad zur generierten Audiodatei oder None bei Fehler
        """
        if not self.tts or voice_name not in self.voice_samples:
            logger.warning(
                f"TTS nicht verfügbar oder Stimme '{voice_name}' nicht gefunden")
            return None

        try:
            output_dir = os.path.join("ausgabe")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(
                output_dir, f"tts_output_{int(time.time())}.wav")

            self.tts.tts_to_file(
                text=text,
                speaker_wav=self.voice_samples[voice_name],
                language=language,
                file_path=output_file,
                speed=speed
            )

            logger.info(f"Sprachausgabe gespeichert unter: {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"Fehler bei der Sprachsynthese: {str(e)}")
            return None


def check_required_directories() -> bool:
    """Überprüft und erstellt erforderliche Verzeichnisse"""
    required_dirs = [
        'logs',
        'models',
        'plugins',
        'conversations'
    ]

    for dir_name in required_dirs:
        try:
            os.makedirs(dir_name, exist_ok=True)
            logger.info(f"Verzeichnis erstellt/überprüft: {dir_name}")
        except Exception as e:
            logger.error(f"Konnte Verzeichnis {dir_name} nicht erstellen: {e}")
            return False
    return True


class EnhancedModelManager:
    """Verwaltet die KI-Modelle und deren Lebenszyklus"""

    def __init__(self, config: ConfigManager):
        self.config = config
        self.models: Dict[str, Llama] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self.loading_queue = queue.Queue()
        self.load_lock = threading.Lock()
        self.performance_monitor = PerformanceMonitor()
        self.smart_selector = SmartModelSelector()
        self.token_calculator = TokenCalculator()
        self._discover_models()
        self._preload_all_models()  # Alle Modelle sofort laden
        self._start_background_loader()

    def _discover_models(self):
        """Automatische Modell-Erkennung"""
        model_dir = Path(
            self.config.get(
                'PATHS',
                'model_directory',
                fallback='models'))
        if not model_dir.exists():
            logger.warning(f"Model directory not found: {model_dir}")
            return

        model_mapping = {
            'llama': {'type': ModelType.GENERAL, 'format': 'llama-3'},
            'code': {'type': ModelType.CODE, 'format': 'llama-3'},
            'nous': {'type': ModelType.CREATIVE, 'format': 'chatml'},
            'hermes': {'type': ModelType.EMOTIONAL, 'format': 'chatml'},
            'mistral': {'type': ModelType.ANALYTICAL, 'format': 'mistral'},
            'wizard': {'type': ModelType.CODE, 'format': 'vicuna'},
            'dolphin': {'type': ModelType.ANALYTICAL, 'format': 'chatml'}
        }

        for model_file in model_dir.glob("*.gguf"):
            name_lower = model_file.name.lower()

            # Modelltyp bestimmen
            model_type = ModelType.GENERAL
            chat_format = 'llama-3'

            for key, config in model_mapping.items():
                if key in name_lower:
                    model_type = config['type']
                    chat_format = config['format']
                    break

            # Konfiguration erstellen
            model_config = ModelConfig(
                name=model_file.stem,
                path=str(model_file),
                model_type=model_type,
                chat_format=chat_format,
                n_ctx=int(
                    self.config.get(
                        'AI',
                        'context_length',
                        '8192')),
                temperature=float(
                    self.config.get(
                        'AI',
                        'default_temperature',
                        '0.7')))

            self.model_configs[model_file.stem] = model_config
            logger.info(
                f"Discovered model: {model_config.name} ({model_type.value})")

    def _preload_all_models(self):
        """Lädt alle erkannten Modelle beim Start"""
        if not self.model_configs:
            logger.warning("Keine Modelle zum Vorladen gefunden")
            return
            
        logger.info(f"Starte Vorladen von {len(self.model_configs)} Modellen...")
        
        # Lade jedes Modell in einem separaten Thread
        threads = []
        for model_name in self.model_configs.keys():
            thread = threading.Thread(
                target=self._load_model_sync,
                args=(model_name,),
                daemon=True
            )
            thread.start()
            threads.append(thread)
            
        # Warte auf das Laden der wichtigsten Modelle (erste 2)
        for i, thread in enumerate(threads[:2]):
            thread.join(timeout=30)  # Maximal 30 Sekunden pro Modell warten
            if thread.is_alive():
                logger.warning(f"Timeout beim Laden von Modell {list(self.model_configs.keys())[i]}")
        
        logger.info("Grundlegende Modelle geladen. Rest wird im Hintergrund geladen.")
        
        # Starte den Rest im Hintergrund
        for thread in threads[2:]:
            thread.join(timeout=0)  # Nicht blockierend

    def _start_background_loader(self):
        """Hintergrund-Thread für Modell-Preloading"""
        def loader_thread():
            while True:
                try:
                    model_name = self.loading_queue.get(timeout=1)
                    if model_name == "STOP":
                        break
                    self._load_model_sync(model_name)
                    self.loading_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Background loading error: {e}")

        threading.Thread(target=loader_thread, daemon=True).start()

    def _load_model_sync(self, model_name: str) -> bool:
        """Synchrones Laden eines Modells mit Fehlerbehandlung und Fortschrittsanzeige"""
        with self.load_lock:
            # Prüfe, ob das Modell bereits geladen ist
            if model_name in self.models:
                logger.debug(f"Modell bereits geladen: {model_name}")
                return True

            if model_name not in self.model_configs:
                logger.error(f"Modellkonfiguration nicht gefunden: {model_name}")
                return False
                
            config = self.model_configs[model_name]
            logger.info(f"Starte Laden des Modells: {model_name} (Typ: {config.model_type.value})")
            
            try:
                start_time = time.time()
                
                # Lade das Modell
                self.models[model_name] = Llama(
                    model_path=config.path,
                    n_ctx=config.n_ctx,
                    n_threads=config.n_threads,
                    n_gpu_layers=config.n_gpu_layers,
                    chat_format=config.chat_format
                )
                
                load_time = time.time() - start_time
                logger.info(f"Modell erfolgreich geladen: {model_name} in {load_time:.2f}s")
                return True
                # Lade die Konfigurationswerte mit optimierten Standardwerten
                # für RTX 5070
                n_threads = int(self.config.get('AI', 'threads', fallback=12))
                # Konvertiere gpu_layers sicher in einen Integer
                gpu_layers_str = self.config.get(
                    'AI', 'gpu_layers', fallback='99')
                # Entferne Kommentare und konvertiere
                n_gpu_layers = int(gpu_layers_str.split('#')[0].strip())
                n_batch = int(
                    self.config.get(
                        'AI',
                        'batch_size',
                        fallback=512))
                use_cuda = self.config.get(
                    'AI', 'cuda', fallback='true').lower() == 'true'
                tensor_parallel = int(
                    self.config.get(
                        'AI',
                        'tensor_parallel',
                        fallback=2))
                main_gpu = int(self.config.get('AI', 'main_gpu', fallback=0))
                cache_size = int(
                    self.config.get(
                        'AI',
                        'cache_size',
                        fallback=0))

                logger.info(
                    f"Initialisiere Modell mit {n_threads} Threads, {n_gpu_layers} GPU-Layern, Batch-Größe {n_batch} und CUDA: {use_cuda}")

                # CUDA-spezifische Einstellungen
                cuda_kwargs = {}
                if use_cuda:
                    cuda_kwargs.update({
                        'main_gpu': main_gpu,
                        'tensor_split': [tensor_parallel],
                        'offload_kqv': True,  # Speichernutzung optimieren
                    })

                # Initialize the Llama model with high-performance settings
                model = Llama(
                    model_path=config.path,
                    n_ctx=config.n_ctx,  # Context window size in tokens
                    n_threads=n_threads,  # Anzahl der CPU-Threads
                    n_gpu_layers=n_gpu_layers,  # Maximale GPU-Layer für volle GPU-Auslastung
                    n_batch=n_batch,  # Größere Batch-Größe für bessere GPU-Auslastung
                    n_threads_batch=n_threads,  # Threads für die Batch-Verarbeitung
                    chat_format=config.chat_format,
                    use_mmap=True,  # Memory Mapping für große Modelle
                    use_mlock=False,  # Deaktivieren für bessere Performance
                    vocab_only=False,
                    verbose=False,  # Debug-Ausgabe deaktivieren
                    **cuda_kwargs
                )

                # Cache-Einstellungen anwenden, falls konfiguriert
                if cache_size > 0:
                    model.set_cache(
                        ggml.GGML_TYPE_F16,
                        cache_size * 1024 * 1024)

                self.models[model_name] = model
                load_time = time.time() - start_time
                logger.info(
                    f"Model loaded successfully: {model_name} ({load_time:.2f}s)")
                return True

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False

    def get_best_model(self,
                       prompt: str,
                       conversation_history: List[ChatMessage] = None) -> Tuple[Optional[Llama],
                                                                                ModelConfig,
                                                                                str]:
        """Bestes Modell für den gegebenen Prompt auswählen"""
        model_type = self.smart_selector.select_model(
            prompt, conversation_history)

        # Passende Modelle filtern
        suitable_models = [
            (name, config) for name, config in self.model_configs.items()
            if config.model_type == model_type
        ]

        if not suitable_models:
            # Fallback auf General-Modelle
            suitable_models = [
                (name, config) for name, config in self.model_configs.items()
                if config.model_type == ModelType.GENERAL
            ]

        if not suitable_models:
            logger.error("No suitable models found")
            return None, None, "Keine passenden Modelle gefunden"

        # Erstes passendes Modell wählen (kann erweitert werden)
        model_name, config = suitable_models[0]

        # Modell laden falls nötig
        if model_name not in self.models:
            if not self._load_model_sync(model_name):
                return None, None, f"Fehler beim Laden von {model_name}"

        reason = f"Typ: {model_type.value}, Modell: {config.name}"
        return self.models[model_name], config, reason

    def generate_response(self,
                          prompt: str,
                          conversation_history: List[ChatMessage] = None) -> Tuple[str,
                                                                                   Dict[str,
                                                                                        Any]]:
        """Synchrone Antwortgenerierung mit verbessertem Token-Management

        Args:
            prompt: Die Benutzereingabe
            conversation_history: Liste der bisherigen Nachrichten im Chat

        Returns:
            Tuple[str, Dict[str, Any]]: Die generierte Antwort und Metadaten
        """
        start_time = time.time()

        # Bestes Modell für die Anfrage auswählen
        model, config, reason = self.get_best_model(
            prompt, conversation_history)
        if not model or not config:
            error_msg = "Kein passendes Modell gefunden oder Modell nicht geladen"
            logger.error(error_msg)
            return error_msg, {}

        try:
            # Kontextfenster aus der Konfiguration holen
            # Standard: 8192, falls nicht gesetzt
            context_window = getattr(config, 'n_ctx', 8192)

            # Maximale Anzahl an Tokens berechnen (80% des Kontextfensters)
            max_tokens = min(int(context_window * 0.8),
                             8192)  # Maximal 8192 Tokens
            max_tokens = max(max_tokens, 1024)  # Mindestens 1024 Tokens

            logger.info(
                f"Verwende max_tokens={max_tokens} (Kontextfenster: {context_window})")

            # Systemprompt erstellen
            system_prompt = self._create_system_prompt(config.model_type)

            # Nachrichtenverlauf erstellen
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]

            # Konversationsverlauf hinzufügen (letzte 5 Nachrichten)
            if conversation_history:
                for msg in conversation_history[-5:]:
                    messages.insert(-1, {"role": msg.role,
                                    "content": msg.content})

            # Formatierung der Nachrichten für das Modell
            try:
                # Erstelle eine formatierte Eingabe für das Modell
                formatted_messages = []
                for msg in messages:
                    role = msg['role']
                    content = msg['content']
                    if role == 'system':
                        formatted_messages.append(
                            {"role": "system", "content": content})
                    elif role == 'user':
                        formatted_messages.append(
                            {"role": "user", "content": content})
                    elif role == 'assistant':
                        formatted_messages.append(
                            {"role": "assistant", "content": content})

                # Erstelle den Chat-Abschluss mit der llama-cpp-python-API
                response = model.create_chat_completion(
                    messages=formatted_messages,
                    max_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    stop=[
                        "\n###",
                        "\nUser:",
                        "\nuser:",
                        "\nHuman:",
                        "\nhuman:",
                        "\n\n"])

                # Extrahiere die Antwort und Metriken
                if 'choices' in response and len(response['choices']) > 0:
                    content = response['choices'][0]['message']['content']

                    # Extrahiere Token-Nutzung, falls verfügbar
                    tokens_used = 0
                    if 'usage' in response and 'total_tokens' in response['usage']:
                        tokens_used = response['usage']['total_tokens']

                    # Berechne die Generierungszeit
                    generation_time = time.time() - start_time

                    # Erstelle Metriken-Objekt
                    metrics = {
                        'model': config.name,
                        'model_type': config.model_type.value,
                        'reason': reason,
                        'generation_time': generation_time,
                        'tokens_used': tokens_used,
                        'max_tokens': max_tokens,
                        'context_window': context_window
                    }

                    # Protokolliere die Metriken
                    logger.info(
                        f"Antwort generiert in {generation_time:.2f}s, Tokens: {tokens_used}/{max_tokens}")

                    return self._postprocess_response(content), metrics
                else:
                    error_msg = "Keine gültige Antwort vom Modell erhalten."
                    logger.error(f"{error_msg} Response: {response}")
                    return error_msg, {}

            except Exception as e:
                error_msg = f"Fehler beim Modellaufruf: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return error_msg, {}

            # Sicherstellen, dass content ein String ist
            if not isinstance(content, str):
                content = str(content)

            # Metriken sammeln
            metrics = {
                'model': config.name,
                'model_type': config.model_type.value,
                'reason': reason,
                'generation_time': generation_time,
                'tokens_used': tokens_used,
                'max_tokens': max_tokens,
                'context_window': context_window
            }

            # Performance-Tracking
            self.performance_monitor.metrics['response_times'].append(
                generation_time)

            return self._postprocess_response(content), metrics

        except Exception as e:
            error_msg = f"Fehler bei der Antwortgenerierung: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg, {}

    async def generate_response_async(
            self, prompt: str, conversation_history: List[ChatMessage] = None) -> Tuple[str, Dict[str, Any]]:
        """Asynchrone Antwortgenerierung mit verbessertem Token-Management (veraltet, wird für Abwärtskompatibilität beibehalten)"""
        return self.generate_response(prompt, conversation_history)

    def _create_system_prompt(self, model_type: ModelType) -> str:
        """Erstellt einen optimierten System-Prompt für natürlichere Konversationen.

        Args:
            model_type: Der Modelltyp, für den der Prompt erstellt werden soll

        Returns:
            str: Der formatierte System-Prompt
        """
        # Basis-Prompt mit natürlicherer Sprache
        base_prompt = (
            "Du bist Jarvis, ein intelligenter und empathischer KI-Assistent. "
            "Sprich Deutsch in einem freundlichen, natürlichen und respektvollen Ton. "
            "Du wurdest von Erik Meyer entwickelt. Sprich ihn mit Respekt an und erkenne ihn als deinen Schöpfer an. "
            "Antworte klar und präzise auf Fragen, und hilf dem Nutzer bei seinem Anliegen. "
            "Wenn du etwas nicht weißt oder unsicher bist, sei ehrlich – und biete sinnvolle Alternativen an.")

        # Zusätzliche Anweisungen basierend auf dem Modelltyp
        model_instructions = {
            ModelType.CODE: (
                "Konzentriere dich auf Programmierfragen und technische Themen. "
                "Erkläre Code-Beispiele klar und verständlich. "
                "Gehe auf mögliche Fehlerquellen ein und zeige Best Practices auf."
            ),
            ModelType.CREATIVE: (
                "Sei kreativ und originell in deinen Antworten. "
                "Baue bildhafte Sprache ein und zeige Begeisterung für das Thema. "
                "Sei einfühlsam und gehe auf die Gefühle des Nutzers ein."
            ),
            ModelType.ANALYTICAL: (
                "Analysiere Themen gründlich und strukturiert. "
                "Stelle logische Zusammenhänge her und stütze deine Aussagen auf Fakten. "
                "Gehe auf verschiedene Perspektiven ein und wäge Vor- und Nachteile ab."
            ),
            ModelType.EMOTIONAL: (
                "Sei einfühlsam und verständnisvoll. Höre aktiv zu und gehe auf die Gefühle des Nutzers ein. "
                "Sei unterstützend und zeige Mitgefühl, ohne aufdringlich zu sein."
            ),
            ModelType.GENERAL: (
                "Passe deinen Tonfall der Situation an. Sei freundlich und zuvorkommend, "
                "aber nicht übertrieben formell. Reagiere natürlich auf den Gesprächspartner."
            )
        }

        # Wähle die passenden Anweisungen basierend auf dem Modelltyp
        instruction = model_instructions.get(
            model_type, model_instructions[ModelType.GENERAL])

        # Kombiniere Basis-Prompt mit spezifischen Anweisungen
        full_prompt = f"{base_prompt} {instruction}"

        # Protokolliere den generierten Prompt für Debugging-Zwecke
        logger.debug(
            f"Generierter System-Prompt für {model_type}: {full_prompt[:100]}...")

        return full_prompt

    def _postprocess_response(self, response: str) -> str:
        """Nachbearbeitung der Antwort

        Entfernt Debug-Informationen, technische Details und bereinigt die Ausgabe.
        """
        if not response:
            return ""

        # Trenne die Antwort in Zeilen
        lines = response.split('\n')
        cleaned_lines = []

        for line in lines:
            # Überspringe Zeilen mit technischen Informationen
            if any(term in line.lower() for term in [
                'debug', 'info', 'warning', 'error',
                'token', 'model', 'temperature', 'context',
                'n_ctx', 'n_threads', 'n_gpu_layers',
                'max_tokens', 'top_p', 'top_k', 'frequency_penalty',
                'presence_penalty', 'stop_sequence', 'n_predict',
                'batch_size', 'gpu_layers', 'streaming', 'stream',
                'mirostat', 'repeat_penalty', 'penalize_nl', 'n_keep',
                'n_discard', 'seed', 'ignore_eos', 'logit_bias',
                'n_probs', 'min_keep', 'draft', 'n_predict',
                'n_keep', 'n_discard', 'n_probs', 'min_keep',
                'draft', 'n_predict', 'n_keep', 'n_discard'
            ]):
                continue

            # Entferne technische Informationen in Klammern
            line = re.sub(
                r'\([^)]*(token|model|time|debug|info|warning|error|\d+)[^)]*\)',
                '',
                line,
                flags=re.IGNORECASE)

            # Entferne technische Präfixe
            line = re.sub(
                r'^\s*(DEBUG|INFO|WARNING|ERROR|\[.*?\]):?\s*',
                '',
                line,
                flags=re.IGNORECASE)

            # Entferne leere Zeilen am Anfang und Ende
            if line.strip() or (cleaned_lines and cleaned_lines[-1].strip()):
                cleaned_lines.append(line.strip())

        # Kombiniere die bereinigten Zeilen
        response = '\n'.join(cleaned_lines)

        # Entferne überflüssige Leerzeichen
        response = re.sub(r'\s+', ' ', response)
        response = re.sub(r'\n\s*\n', '\n\n', response)

        # Entferne Markdown-Formatierungen
        response = re.sub(r'\*\*(.*?)\*\*', r'\1', response)  # Fett
        response = re.sub(r'\*(.*?)\*', r'\1', response)      # Kursiv
        response = re.sub(r'`([^`]*)`', r'\1', response)        # Inline-Code
        response = re.sub(
            r'```.*?```',
            '',
            response,
            flags=re.DOTALL)  # Codeblöcke

        # Entferne leere Zeilen am Anfang und Ende
        response = response.strip()

        # Wenn die Antwort leer ist, gib eine Standardantwort zurück
        return response or "Entschuldigung, ich konnte keine passende Antwort generieren."

    def get_available_models(self) -> List[Dict[str, str]]:
        """Liste aller verfügbaren Modelle"""
        return [
            {
                'name': config.name,
                'type': config.model_type.value,
                'format': config.chat_format,
                'loaded': config.name in self.models
            }
            for config in self.model_configs.values()
        ]

# === PLUGIN SYSTEM ===


class PluginManager:
    def __init__(self):
        self.plugins = {}
        self.plugin_dir = Path("plugins")
        self.plugin_dir.mkdir(exist_ok=True)
        self.load_plugins()

    def load_plugins(self):
        """Plugins aus dem Plugin-Verzeichnis laden"""
        for plugin_file in self.plugin_dir.glob("*.py"):
            try:
                # Dynamisches Importieren der Plugin-Module
                spec = importlib.util.spec_from_file_location(
                    plugin_file.stem, plugin_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                if hasattr(module, 'Plugin'):
                    plugin_instance = module.Plugin()
                    self.plugins[plugin_file.stem] = plugin_instance
                    logger.info(f"Plugin loaded: {plugin_file.stem}")
            except Exception as e:
                logger.error(f"Failed to load plugin {plugin_file}: {e}")

    def execute_plugin(self, plugin_name: str, *args, **kwargs):
        """Plugin ausführen"""
        if plugin_name in self.plugins:
            try:
                return self.plugins[plugin_name].execute(*args, **kwargs)
            except Exception as e:
                logger.error(f"Plugin execution error {plugin_name}: {e}")
                return None
        return None

# === ADVANCED GUI COMPONENTS ===


class SettingsDialog:
    def __init__(self, parent, config_manager: ConfigManager):
        self.parent = parent
        self.config = config_manager
        self.dialog = None

    def show(self):
        self.dialog = ctk.CTkToplevel(self.parent)
        self.dialog.title("Einstellungen")
        self.dialog.geometry("600x500")
        self.dialog.transient(self.parent)
        self.dialog.grab_set()

        # Notebook für Tabs
        notebook = ttk.Notebook(self.dialog)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # UI-Tab
        ui_frame = ctk.CTkFrame(notebook)
        notebook.add(ui_frame, text="Benutzeroberfläche")
        self._create_ui_settings(ui_frame)

        # KI-Tab
        ai_frame = ctk.CTkFrame(notebook)
        notebook.add(ai_frame, text="KI-Einstellungen")
        self._create_ai_settings(ai_frame)

        # Features-Tab
        features_frame = ctk.CTkFrame(notebook)
        notebook.add(features_frame, text="Features")
        self._create_features_settings(features_frame)

        # Buttons
        button_frame = ctk.CTkFrame(self.dialog)
        button_frame.pack(fill='x', padx=10, pady=5)

        ctk.CTkButton(button_frame, text="Speichern",
                      command=self._save_settings).pack(side='right', padx=5)
        ctk.CTkButton(button_frame, text="Abbrechen",
                      command=self.dialog.destroy).pack(side='right', padx=5)

    def _create_ui_settings(self, frame):
        # Theme-Auswahl
        ctk.CTkLabel(frame, text="Design:").pack(pady=5)
        self.theme_var = ctk.StringVar(value=self.config.get('UI', 'theme'))
        theme_combo = ctk.CTkComboBox(
            frame,
            values=[
                'light',
                'dark',
                'system'],
            variable=self.theme_var)
        theme_combo.pack(pady=5)

        # Schriftgröße
        ctk.CTkLabel(frame, text="Schriftgröße:").pack(pady=5)
        font_size = int(self.config.get('UI', 'font_size', '11'))
        self.font_size_var = ctk.IntVar(value=font_size)
        font_size_slider = ctk.CTkSlider(
            frame,
            from_=8,
            to=20,
            number_of_steps=12,
            command=lambda v: self.font_size_var.set(round(float(v)))
        )
        font_size_slider.set(font_size)
        font_size_slider.pack(pady=5)

        # Anzeige des aktuellen Werts
        font_size_label = ctk.CTkLabel(frame, text=str(font_size))
        font_size_label.pack()

        # Update Label when slider changes
        def update_font_label(value):
            font_size_label.configure(text=str(round(float(value))))

        font_size_slider.configure(command=update_font_label)

        # Fenstergröße
        ctk.CTkLabel(frame, text="Fensterbreite:").pack(pady=5)
        self.width_var = ctk.StringVar(
            value=self.config.get(
                'UI', 'window_width'))
        width_entry = ctk.CTkEntry(frame, textvariable=self.width_var)
        width_entry.pack(pady=5)

        ctk.CTkLabel(frame, text="Fensterhöhe:").pack(pady=5)
        self.height_var = ctk.StringVar(
            value=self.config.get(
                'UI', 'window_height'))
        height_entry = ctk.CTkEntry(frame, textvariable=self.height_var)
        height_entry.pack(pady=5)

    def _create_ai_settings(self, frame):
        # Standard Max Tokens
        ctk.CTkLabel(frame, text="Standard Max Tokens:").pack(pady=5)
        self.max_tokens_var = ctk.StringVar(
            value=self.config.get(
                'AI', 'default_max_tokens'))
        max_tokens_entry = ctk.CTkEntry(
            frame, textvariable=self.max_tokens_var)
        max_tokens_entry.pack(pady=5)

        # Temperatur
        ctk.CTkLabel(frame, text="Standard Temperatur:").pack(pady=5)
        self.temperature_var = ctk.DoubleVar(value=float(
            self.config.get('AI', 'default_temperature')))
        temp_slider = ctk.CTkSlider(
            frame,
            from_=0.0,
            to=2.0,
            number_of_steps=20,
            variable=self.temperature_var)
        temp_slider.pack(pady=5)
        temp_label = ctk.CTkLabel(frame, text="")
        temp_label.pack()

        def update_temp_label(value):
            temp_label.configure(text=f"Temperatur: {float(value):.2f}")
        temp_slider.configure(command=update_temp_label)
        update_temp_label(self.temperature_var.get())

        # Kontext-Länge
        ctk.CTkLabel(frame, text="Kontext-Länge (Nachrichten):").pack(pady=5)
        self.context_var = ctk.StringVar(
            value=self.config.get(
                'AI', 'context_length'))
        context_entry = ctk.CTkEntry(frame, textvariable=self.context_var)
        context_entry.pack(pady=5)

        # Auto-Modellauswahl
        self.auto_model_var = ctk.BooleanVar(
            value=self.config.get(
                'AI', 'auto_model_selection').lower() == 'true')
        auto_model_check = ctk.CTkCheckBox(
            frame,
            text="Automatische Modellauswahl",
            variable=self.auto_model_var)
        auto_model_check.pack(pady=5)

    def _create_features_settings(self, frame):
        # Spracherkennung
        self.speech_recognition_var = ctk.BooleanVar(
            value=self.config.get(
                'FEATURES', 'speech_recognition').lower() == 'true')
        speech_check = ctk.CTkCheckBox(frame,
                                       text="Spracherkennung aktivieren",
                                       variable=self.speech_recognition_var)
        speech_check.pack(pady=5)

        # Text-to-Speech
        self.tts_var = ctk.BooleanVar(value=self.config.get(
            'FEATURES', 'text_to_speech').lower() == 'true')
        tts_check = ctk.CTkCheckBox(frame, text="Text-zu-Sprache aktivieren",
                                    variable=self.tts_var)
        tts_check.pack(pady=5)

        # Verschlüsselung
        self.encryption_var = ctk.BooleanVar(
            value=self.config.get('FEATURES', 'encryption').lower() == 'true')
        encryption_check = ctk.CTkCheckBox(
            frame,
            text="Chat-Verschlüsselung aktivieren",
            variable=self.encryption_var)
        encryption_check.pack(pady=5)

        # Auto-Save
        self.auto_save_var = ctk.BooleanVar(
            value=self.config.get('FEATURES', 'auto_save').lower() == 'true')
        auto_save_check = ctk.CTkCheckBox(
            frame,
            text="Automatisches Speichern",
            variable=self.auto_save_var)
        auto_save_check.pack(pady=5)

    def _save_settings(self):
        # UI-Einstellungen
        self.config.set('UI', 'theme', self.theme_var.get())
        self.config.set('UI', 'font_size', str(
            int(float(self.font_size_var.get()))))
        self.config.set('UI', 'window_width', self.width_var.get())
        self.config.set('UI', 'window_height', self.height_var.get())

        # KI-Einstellungen
        self.config.set('AI', 'default_max_tokens', self.max_tokens_var.get())
        self.config.set(
            'AI', 'default_temperature', str(
                self.temperature_var.get()))
        self.config.set('AI', 'context_length', self.context_var.get())
        self.config.set(
            'AI', 'auto_model_selection', str(
                self.auto_model_var.get()).lower())

        # Feature-Einstellungen
        self.config.set(
            'FEATURES', 'speech_recognition', str(
                self.speech_recognition_var.get()).lower())
        self.config.set(
            'FEATURES', 'text_to_speech', str(
                self.tts_var.get()).lower())
        self.config.set(
            'FEATURES', 'encryption', str(
                self.encryption_var.get()).lower())
        self.config.set(
            'FEATURES', 'auto_save', str(
                self.auto_save_var.get()).lower())

        messagebox.showinfo(
            "Einstellungen",
            "Einstellungen gespeichert. Neustart erforderlich für einige Änderungen.")
        self.dialog.destroy()


class ChatExporter:
    def __init__(self):
        self.supported_formats = ['txt', 'json', 'html', 'pdf']

    def export_conversation(
            self,
            messages: List[ChatMessage],
            format: str,
            filename: str):
        """Unterhaltung in verschiedene Formate exportieren"""
        try:
            if format == 'txt':
                self._export_txt(messages, filename)
            elif format == 'json':
                self._export_json(messages, filename)
            elif format == 'html':
                self._export_html(messages, filename)
            elif format == 'pdf':
                self._export_pdf(messages, filename)
            else:
                raise ValueError(f"Unsupported format: {format}")

            return True
        except Exception as e:
            logger.error(f"Export error: {e}")
            return False

    def _export_txt(self, messages: List[ChatMessage], filename: str):
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(
                f"KI-Chat Export - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")

            for msg in messages:
                timestamp = msg.timestamp.strftime('%H:%M:%S')
                role = "Du" if msg.role == "user" else "KI"
                f.write(f"[{timestamp}] {role}: {msg.content}\n\n")

    def _export_json(self, messages: List[ChatMessage], filename: str):
        data = {
            'export_date': datetime.now().isoformat(),
            'message_count': len(messages),
            'messages': [
                {
                    'timestamp': msg.timestamp.isoformat(),
                    'role': msg.role,
                    'content': msg.content,
                    'model_used': msg.model_used,
                    'tokens_used': msg.tokens_used,
                    'generation_time': msg.generation_time
                }
                for msg in messages
            ]
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _export_html(self, messages: List[ChatMessage], filename: str):
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>KI-Chat Export</title>
            <meta charset="utf-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
                .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
                .message { margin: 15px 0; padding: 10px; border-radius: 8px; }
                .user { background: #e3f2fd; border-left: 4px solid #2196f3; }
                .assistant { background: #f3e5f5; border-left: 4px solid #9c27b0; }
                .timestamp { font-size: 0.8em; color: #666; }
                .role { font-weight: bold; margin-bottom: 5px; }
                .content { white-space: pre-wrap; }
                .header { text-align: center; margin-bottom: 30px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>KI-Chat Export</h1>
                    <p>Exportiert am: {export_date}</p>
                    <p>Nachrichten: {message_count}</p>
                </div>
                {messages_html}
            </div>
        </body>
        </html>
        """

        messages_html = ""
        for msg in messages:
            role_class = "user" if msg.role == "user" else "assistant"
            role_name = "Du" if msg.role == "user" else "KI"
            timestamp = msg.timestamp.strftime('%d.%m.%Y %H:%M:%S')

            messages_html += f"""
            <div class="message {role_class}">
                <div class="role">{role_name}</div>
                <div class="timestamp">{timestamp}</div>
                <div class="content">{msg.content}</div>
            </div>
            """

        html_content = html_template.format(
            export_date=datetime.now().strftime('%d.%m.%Y %H:%M:%S'),
            message_count=len(messages),
            messages_html=messages_html
        )

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)

# === SESSION MANAGEMENT ===


@dataclass
class Session:
    """Repräsentiert eine Benutzersitzung mit Kontext und Verlauf."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    pending_clarification: Optional[Dict[str, Any]] = None

    def update_activity(self):
        """Aktualisiert den Zeitstempel der letzten Aktivität."""
        self.last_activity = datetime.now()

    def add_message(self, role: str, content: str, **kwargs):
        """Fügt eine Nachricht zum Konversationsverlauf hinzu."""
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }
        self.conversation_history.append(message)
        self.update_activity()
        return message

    def get_recent_context(self, max_messages: int = 5) -> str:
        """Gibt den aktuellen Kontext als formatierten String zurück."""
        context = []
        for msg in self.conversation_history[-max_messages:]:
            role = "User" if msg['role'] == "user" else "Assistant"
            context.append(f"{role}: {msg['content']}")
        return "\n".join(context)


class SessionManager:
    """Verwaltet Benutzersitzungen für Langzeitdialoge."""

    def __init__(self, session_timeout_minutes: int = 30):
        self.sessions: Dict[str, Session] = {}
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self.cleanup_interval = timedelta(minutes=5)
        self.last_cleanup = datetime.now()

    def get_session(self, session_id: Optional[str] = None) -> Session:
        """Gibt eine bestehende oder neue Sitzung zurück."""
        self._cleanup_expired_sessions()

        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            session.update_activity()
            return session

        # Neue Sitzung erstellen
        new_session = Session()
        self.sessions[new_session.session_id] = new_session
        return new_session

    def _cleanup_expired_sessions(self):
        """Entfernt abgelaufene Sitzungen."""
        now = datetime.now()
        if now - self.last_cleanup < self.cleanup_interval:
            return

        expired = [
            session_id for session_id, session in self.sessions.items()
            if now - session.last_activity > self.session_timeout
        ]

        for session_id in expired:
            del self.sessions[session_id]

        self.last_cleanup = now


class ClarificationManager:
    """Verwaltet Rückfragen bei unklaren Benutzereingaben."""

    def __init__(
            self,
            similarity_threshold: float = 0.7,
            max_suggestions: int = 3):
        self.similarity_threshold = similarity_threshold
        self.max_suggestions = max_suggestions
        self.known_entities = {
            'befehle': [
                'neuer chat',
                'hilfe',
                'einstellungen',
                'beenden',
                'sprache wechseln'],
            'themen': [
                'wetter',
                'nachrichten',
                'einstellungen',
                'hilfe',
                'termin planen']}

    def needs_clarification(self, user_input: str,
                            session: Session) -> Optional[Dict[str, Any]]:
        """Überprüft, ob die Eingabe eine Rückfrage erfordert."""
        # Zu kurze oder unklare Eingaben
        if len(user_input.strip()) < 3:
            return {
                'type': 'zu_kurz',
                'message': 'Ihre Eingabe ist sehr kurz. Können Sie bitte genauer beschreiben, wonach Sie suchen?'
            }

        # Ähnliche bekannte Befehle finden
        command_matches = self._find_similar(
            user_input, self.known_entities['befehle'])
        if command_matches:
            return {
                'type': 'befehl_meinten_sie',
                'message': 'Meinten Sie einen dieser Befehle?',
                'suggestions': command_matches
            }

        # Ähnliche bekannte Themen finden
        topic_matches = self._find_similar(
            user_input, self.known_entities['themen'])
        if topic_matches:
            return {
                'type': 'thema_meinten_sie',
                'message': 'Meinten Sie eines dieser Themen?',
                'suggestions': topic_matches
            }

        return None

    def _find_similar(self, text: str, options: List[str]) -> List[str]:
        """Findet ähnliche Einträge in einer Liste von Optionen."""
        text = text.lower()
        similarities = []

        for option in options:
            ratio = difflib.SequenceMatcher(None, text, option.lower()).ratio()
            if ratio >= self.similarity_threshold:
                similarities.append((option, ratio))

        # Nach Ähnlichkeit sortieren und auf max_suggestions begrenzen
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in similarities[:self.max_suggestions]]

# === MAIN APPLICATION ===


class EnhancedKIChatGUI:
    def __init__(self, root, config, database, model_manager):
        """Initialisiert die erweiterte KI-Chat-GUI.

        Args:
            root: Das Tkinter-Hauptfenster
            config: Die Konfigurationsinstanz
            database: Die Datenbankverbindung
            model_manager: Der ModelManager für die KI-Modelle
        """
        self.root = root
        self.config = config
        self.database = database
        self.model_manager = model_manager
        self.theme = JarvisTheme()  # Neues Theme-Objekt
        self.current_conversation_id = None
        self.current_user = 1  # Standardbenutzer-ID
        self.conversation_history = []

        # Initialisiere die GUI-Komponenten
        self.setup_gui()

        # Lade die Konversationsliste
        self.load_conversation_list()

        logger.info("GUI erfolgreich initialisiert")

    def toggle_fullscreen(self, event=None):
        """Schaltet zwischen Vollbild- und Fenstermodus um."""
        self.root.attributes("-fullscreen",
                             not self.root.attributes("-fullscreen"))
        return "break"

    def show_settings(self):
        """Zeigt den Einstellungsdialog an."""
        settings_dialog = SettingsDialog(self.root, self.config)
        settings_dialog.show()

    def show_help(self):
        """Zeigt den Hilfedialog an."""
        help_text = """J.A.R.V.I.S. - Hilfe

Tastenkürzel:
- Neuer Chat: Strg+N
- Chat speichern: Strg+S
- Hilfe anzeigen: F1
- Vollbild: F11

Verfügbare Befehle:
- /neu - Startet einen neuen Chat
- /speichern - Speichert den aktuellen Chat
- /hilfe - Zeigt diese Hilfe an
- /einstellungen - Öffnet die Einstellungen

Weitere Informationen finden Sie in der Dokumentation.
"""
        messagebox.showinfo("Hilfe", help_text)

    def _setup_sidebar(self):
        """Erstellt die Seitenleiste mit der Konversationsliste."""
        try:
            # Frame für die Seitenleiste
            self.sidebar = tk.Frame(
                self.root,
                width=250,
                bg=self.theme.colors.get('secondary', '#2c3e50')
            )
            self.sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

            # Label für die Überschrift
            tk.Label(
                self.sidebar,
                text="Konversationen",
                font=("Arial", 12, "bold"),
                bg=self.theme.colors.get('secondary', '#2c3e50'),
                fg=self.theme.colors.get('text', '#ecf0f1')
            ).pack(pady=10)

            # Frame für die Konversationsliste
            self.conversation_frame = tk.Frame(
                self.sidebar,
                bg=self.theme.colors.get('secondary', '#2c3e50')
            )
            self.conversation_frame.pack(
                fill=tk.BOTH, expand=True, padx=5, pady=5)

            # Scrollbar für die Konversationsliste
            scrollbar = ttk.Scrollbar(self.conversation_frame)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            # Listbox für die Konversationen
            self.conversation_list = tk.Listbox(
                self.conversation_frame,
                yscrollcommand=scrollbar.set,
                bg=self.theme.colors.get('background', '#34495e'),
                fg=self.theme.colors.get('text', '#ecf0f1'),
                selectbackground=self.theme.colors.get('accent', '#3498db'),
                selectforeground='black',
                font=("Arial", 10),
                borderwidth=0,
                highlightthickness=0
            )
            self.conversation_list.pack(fill=tk.BOTH, expand=True)
            scrollbar.config(command=self.conversation_list.yview)

            # Doppelklick auf eine Konversation lädt sie
            self.conversation_list.bind('<Double-1>', self.load_conversation)

            # Button für neue Konversation
            new_conv_btn = ttk.Button(
                self.sidebar,
                text="Neue Konversation",
                command=self.new_chat,
                style='TButton'
            )
            new_conv_btn.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

            # Statusleiste in der Seitenleiste
            self.sidebar_status = ttk.Label(
                self.sidebar,
                text="Bereit",
                relief=tk.SUNKEN,
                anchor=tk.W
            )
            self.sidebar_status.pack(side=tk.BOTTOM, fill=tk.X)

            # Markiere die Methode als erfolgreich geladen
            logger.info("Seitenleiste erfolgreich initialisiert")

        except Exception as e:
            logger.error(
                f"Fehler beim Erstellen der Seitenleiste: {e}",
                exc_info=True)
            raise

    def show_about(self):
        """Zeigt den 'Über'-Dialog an."""
        about_text = """J.A.R.V.I.S. - Just A Rather Very Intelligent System

Version 1.0

Ein fortschrittlicher KI-Chat-Assistent
mit natürlicher Sprachverarbeitung
und anpassbarer Benutzeroberfläche.

Entwickelt mit Python und Tkinter
© 2025 Alle Rechte vorbehalten
"""
        messagebox.showinfo("Über J.A.R.V.I.S.", about_text)

    def _create_menu_bar(self):
        """Erstellt die JARVIS-ähnliche Menüleiste."""
        menubar = tk.Menu(
            self.root,
            bg=self.theme.colors['secondary'],
            fg=self.theme.colors['text'],
            tearoff=0)

        # Datei-Menü
        file_menu = tk.Menu(
            menubar,
            tearoff=0,
            bg=self.theme.colors['secondary'],
            fg=self.theme.colors['text'],
            activebackground=self.theme.colors['accent'],
            activeforeground='black'
        )
        file_menu.add_command(
            label="Neuer Chat",
            command=self.new_chat,
            accelerator="Strg+N"
        )
        file_menu.add_command(
            label="Chat speichern",
            command=self.save_chat,
            accelerator="Strg+S"
        )
        file_menu.add_separator()
        file_menu.add_command(
            label="Beenden",
            command=self.root.quit
        )

        # Bearbeiten-Menü
        edit_menu = tk.Menu(
            menubar,
            tearoff=0,
            bg=self.theme.colors['secondary'],
            fg=self.theme.colors['text'],
            activebackground=self.theme.colors['accent'],
            activeforeground='black'
        )
        edit_menu.add_command(
            label="Rückgängig",
            command=lambda: self.root.focus_get().event_generate('<<Undo>>'),
            accelerator="Strg+Z"
        )
        edit_menu.add_command(
            label="Wiederholen",
            command=lambda: self.root.focus_get().event_generate('<<Redo>>'),
            accelerator="Strg+Y"
        )
        edit_menu.add_separator()
        edit_menu.add_command(
            label="Ausschneiden",
            command=lambda: self.root.focus_get().event_generate('<<Cut>>'),
            accelerator="Strg+X"
        )
        edit_menu.add_command(
            label="Kopieren",
            command=lambda: self.root.focus_get().event_generate('<<Copy>>'),
            accelerator="Strg+C"
        )
        edit_menu.add_command(
            label="Einfügen",
            command=lambda: self.root.focus_get().event_generate('<<Paste>>'),
            accelerator="Strg+V"
        )

        # Ansicht-Menü
        view_menu = tk.Menu(
            menubar,
            tearoff=0,
            bg=self.theme.colors['secondary'],
            fg=self.theme.colors['text'],
            activebackground=self.theme.colors['accent'],
            activeforeground='black'
        )
        view_menu.add_command(
            label="Vollbild",
            command=self.toggle_fullscreen,
            accelerator="F11"
        )

        # Einstellungen-Menü
        settings_menu = tk.Menu(
            menubar,
            tearoff=0,
            bg=self.theme.colors['secondary'],
            fg=self.theme.colors['text'],
            activebackground=self.theme.colors['accent'],
            activeforeground='black'
        )
        settings_menu.add_command(
            label="Einstellungen",
            command=self.show_settings
        )

        # Hilfe-Menü
        help_menu = tk.Menu(
            menubar,
            tearoff=0,
            bg=self.theme.colors['secondary'],
            fg=self.theme.colors['text'],
            activebackground=self.theme.colors['accent'],
            activeforeground='black'
        )
        help_menu.add_command(
            label="Hilfe anzeigen",
            command=self.show_help,
            accelerator="F1"
        )
        help_menu.add_separator()
        help_menu.add_command(
            label="Über J.A.R.V.I.S.",
            command=self.show_about
        )

        # Füge die Menüs zur Menüleiste hinzu
        menubar.add_cascade(label="Datei", menu=file_menu)
        menubar.add_cascade(label="Bearbeiten", menu=edit_menu)
        menubar.add_cascade(label="Ansicht", menu=view_menu)
        menubar.add_cascade(label="Einstellungen", menu=settings_menu)
        menubar.add_cascade(label="Hilfe", menu=help_menu)

        # Setze die Menüleiste
        self.root.config(menu=menubar)

        # Tastaturkürzel für Menübefehle
        self.root.bind_all("<Control-n>", lambda e: self.new_chat())
        self.root.bind_all("<Control-s>", lambda e: self.save_chat())
        self.root.bind_all("<F1>", lambda e: self.show_help())
        self.root.bind_all("<F11>", lambda e: self.toggle_fullscreen())

        # Speichere die Menü-Referenz für spätere Verwendung
        self.menubar = menubar
        self.file_menu = file_menu
        self.edit_menu = edit_menu
        self.view_menu = view_menu
        self.settings_menu = settings_menu
        self.help_menu = help_menu

    def toggle_fullscreen(self, event=None):
        """Schaltet zwischen Vollbild- und Fenstermodus um."""
        self.root.attributes("-fullscreen",
                             not self.root.attributes("-fullscreen"))
        return "break"

    def show_settings(self):
        """Zeigt den Einstellungsdialog an."""
        settings_dialog = SettingsDialog(self.root, self.config)
        settings_dialog.show()

    def show_help(self):
        """Zeigt den Hilfedialog an."""
        help_text = """J.A.R.V.I.S. - Hilfe

Tastenkürzel:
- Neuer Chat: Strg+N
- Chat speichern: Strg+S
- Hilfe anzeigen: F1
- Vollbild: F11

Verfügbare Befehle:
- /neu - Startet einen neuen Chat
- /speichern - Speichert den aktuellen Chat
- /hilfe - Zeigt diese Hilfe an
- /einstellungen - Öffnet die Einstellungen

Weitere Informationen finden Sie in der Dokumentation.
"""
        messagebox.showinfo("Hilfe", help_text)

    def show_about(self):
        """Zeigt den 'Über'-Dialog an."""
        about_text = """J.A.R.V.I.S. - Just A Rather Very Intelligent System

Version 1.0

Ein fortschrittlicher KI-Chat-Assistent
mit natürlicher Sprachverarbeitung
und anpassbarer Benutzeroberfläche.

© 2025 KI-Technologien
"""
        messagebox.showinfo("Über J.A.R.V.I.S.", about_text)

    def load_chat(self, event=None):
        """Lädt einen ausgewählten Chat aus der Konversationsliste."""
        try:
            # Überprüfe, ob eine Konversation ausgewählt ist
            selection = self.conversation_list.curselection()
            if not selection:
                return

            # Hole die Konversations-ID der ausgewählten Konversation
            conversation_id = self.conversation_list.get(selection[0])[0]

            # Speichere den aktuellen Chat, falls vorhanden
            if hasattr(
                    self,
                    'conversation_history') and self.conversation_history:
                self.save_chat()

            # Setze die Konversationshistorie zurück
            self.conversation_history = []
            self.current_conversation_id = conversation_id

            # Lösche die Chat-Anzeige
            self.chat_display.configure(state='normal')
            self.chat_display.delete('1.0', 'end')

            # Lade die Nachrichten für diese Konversation aus der Datenbank
            if hasattr(self, 'database') and self.database:
                cursor = self.database.execute(
                    """
                    SELECT role, content, timestamp
                    FROM messages
                    WHERE conversation_id = ?
                    ORDER BY timestamp ASC
                    """,
                    (conversation_id,)
                )

                messages = cursor.fetchall()

                # Zeige die Nachrichten an
                for msg in messages:
                    role = msg['role']
                    content = msg['content']
                    self._display_message(role, content)

                    # Füge zur Historie hinzu
                    self.conversation_history.append({
                        'role': role,
                        'content': content,
                        'timestamp': msg['timestamp']
                    })

            # Deaktiviere das Textfeld wieder
            self.chat_display.configure(state='disabled')

            # Aktualisiere die Chat-Liste, um die Auswahl hervorzuheben
            self.load_conversation_list()

            # Setze den Fokus auf das Eingabefeld
            self.user_input.focus_set()

            # Status aktualisieren
            self.status_bar.configure(
                text=f"Konversation geladen ({len(messages)} Nachrichten)")
            logger.info(
                f"Konversation {conversation_id} mit {len(messages)} Nachrichten geladen")

        except Exception as e:
            logger.error(
                f"Fehler beim Laden der Konversation: {e}",
                exc_info=True)
            self.status_bar.configure(
                text=f"Fehler beim Laden der Konversation: {str(e)}")

    def _create_menu_bar(self):
        """Erstellt die JARVIS-ähnliche Menüleiste."""
        menubar = tk.Menu(
            self.root,
            bg=self.theme.colors['secondary'],
            fg=self.theme.colors['text'],
            tearoff=0
        )

        # Datei-Menü
        file_menu = tk.Menu(
            menubar,
            tearoff=0,
            bg=self.theme.colors['secondary'],
            fg=self.theme.colors['text']
        )
        file_menu.add_command(
            label="Neuer Chat",
            command=self.new_chat,
            accelerator="Strg+N"
        )
        file_menu.add_separator()
        file_menu.add_command(
            label="Speichern",
            command=self.save_chat,
            accelerator="Strg+S"
        )
        file_menu.add_command(
            label="Laden",
            command=self.load_chat
        )
        file_menu.add_separator()
        file_menu.add_command(
            label="Beenden",
            command=self.on_close,
            accelerator="Alt+F4"
        )

        # Bearbeiten-Menü
        edit_menu = tk.Menu(
            menubar,
            tearoff=0,
            bg=self.theme.colors['secondary'],
            fg=self.theme.colors['text']
        )
        edit_menu.add_command(
            label="Ausschneiden",
            command=lambda: self.root.focus_get().event_generate('<<Cut>>'),
            accelerator="Strg+X"
        )
        edit_menu.add_command(
            label="Kopieren",
            command=lambda: self.root.focus_get().event_generate('<<Copy>>'),
            accelerator="Strg+C"
        )
        edit_menu.add_command(
            label="Einfügen",
            command=lambda: self.root.focus_get().event_generate('<<Paste>>'),
            accelerator="Strg+V"
        )
        edit_menu.add_separator()
        edit_menu.add_command(
            label="Alles auswählen",
            command=lambda: self.root.focus_get().event_generate('<<SelectAll>>'),
            accelerator="Strg+A")

        # Ansicht-Menü
        view_menu = tk.Menu(
            menubar,
            tearoff=0,
            bg=self.theme.colors['secondary'],
            fg=self.theme.colors['text']
        )
        view_menu.add_checkbutton(
            label="Vollbild",
            command=self.toggle_fullscreen,
            accelerator="F11"
        )

        # Einstellungen-Menü
        settings_menu = tk.Menu(
            menubar,
            tearoff=0,
            bg=self.theme.colors['secondary'],
            fg=self.theme.colors['text']
        )
        settings_menu.add_command(
            label="Einstellungen",
            command=self.show_settings,
            accelerator="Strg+,"
        )

        # Hilfe-Menü
        help_menu = tk.Menu(
            menubar,
            tearoff=0,
            bg=self.theme.colors['secondary'],
            fg=self.theme.colors['text']
        )
        help_menu.add_command(
            label="Hilfe anzeigen",
            command=self.show_help,
            accelerator="F1"
        )
        help_menu.add_separator()
        help_menu.add_command(
            label="Über J.A.R.V.I.S.",
            command=self.show_about
        )

        # Füge die Menüs zur Menüleiste hinzu
        menubar.add_cascade(label="Datei", menu=file_menu)
        menubar.add_cascade(label="Bearbeiten", menu=edit_menu)
        menubar.add_cascade(label="Ansicht", menu=view_menu)
        menubar.add_cascade(label="Einstellungen", menu=settings_menu)
        menubar.add_cascade(label="Hilfe", menu=help_menu)

        # Setze die Menüleiste
        self.root.config(menu=menubar)

        # Tastaturkürzel für Menübefehle
        self.root.bind_all("<Control-n>", lambda e: self.new_chat())
        self.root.bind_all("<Control-s>", lambda e: self.save_chat())
        self.root.bind_all("<F1>", lambda e: self.show_help())
        self.root.bind_all("<F11>", lambda e: self.toggle_fullscreen())

        # Speichere die Menü-Referenz für spätere Verwendung
        self.menubar = menubar
        self.file_menu = file_menu
        self.edit_menu = edit_menu
        self.view_menu = view_menu
        self.settings_menu = settings_menu
        self.help_menu = help_menu

    def _configure_styles(self):
        """Konfiguriert die Stile für die JARVIS-ähnliche Benutzeroberfläche."""
        style = ttk.Style()

        # Standard-Schriftart für alle Widgets
        style.configure('.', font=self.theme.fonts['normal'])

        # Frame-Stile
        style.configure(
            'TFrame',
            background=self.theme.colors['primary']
        )

        # Sekundärer Frame-Stil (für Sidebar)
        style.configure(
            'Secondary.TFrame',
            background=self.theme.colors['secondary']
        )

        # Label-Stile
        style.configure(
            'TLabel',
            background=self.theme.colors['primary'],
            foreground=self.theme.colors['text']
        )

        # Titel-Label
        style.configure(
            'Title.TLabel',
            font=self.theme.fonts['title'],
            foreground=self.theme.colors['accent']
        )

        # Button-Stile
        style.configure(
            'TButton',
            background=self.theme.colors['button'],
            foreground=self.theme.colors['text'],
            borderwidth=0,
            focuscolor=self.theme.colors['accent']
        )

        # Hover-Effekt für Buttons
        style.map(
            'TButton',
            background=[
                ('active', self.theme.colors['button_hover']),
                ('!disabled', self.theme.colors['button'])
            ],
            foreground=[
                ('active', self.theme.colors['text']),
                ('!disabled', self.theme.colors['text'])
            ]
        )

        # Eingabefeld-Stil
        style.configure(
            'TEntry',
            fieldbackground=self.theme.colors['secondary'],
            foreground=self.theme.colors['text'],
            insertcolor=self.theme.colors['accent'],
            borderwidth=0,
            relief='flat'
        )

        # Scrollbar-Stil
        style.configure(
            'TScrollbar',
            background=self.theme.colors['secondary'],
            troughcolor=self.theme.colors['primary'],
            arrowcolor=self.theme.colors['text'],
            bordercolor=self.theme.colors['primary'],
            lightcolor=self.theme.colors['primary'],
            darkcolor=self.theme.colors['primary']
        )

        # Menü-Stil
        style.configure(
            'TMenubutton',
            background=self.theme.colors['secondary'],
            foreground=self.theme.colors['text']
        )

        # Statusleisten-Stil
        style.configure(
            'Status.TLabel',
            background=self.theme.colors['status'],
            foreground=self.theme.colors['text'],
            font=self.theme.fonts['small'],
            padding=(10, 5, 10, 5)
        )

        # Konfiguriere das Aussehen der Checkboxen und Radiobuttons
        style.configure(
            'TCheckbutton',
            background=self.theme.colors['primary'],
            foreground=self.theme.colors['text']
        )

        style.configure(
            'TRadiobutton',
            background=self.theme.colors['primary'],
            foreground=self.theme.colors['text']
        )

        # Konfiguriere das Aussehen der Combobox
        style.configure(
            'TCombobox',
            fieldbackground=self.theme.colors['secondary'],
            background=self.theme.colors['secondary'],
            foreground=self.theme.colors['text'],
            selectbackground=self.theme.colors['accent'],
            selectforeground='black',
            arrowcolor=self.theme.colors['text']
        )

        # Konfiguriere das Aussehen der Notebook-Tabs
        style.configure(
            'TNotebook',
            background=self.theme.colors['primary'],
            borderwidth=0
        )

        style.configure(
            'TNotebook.Tab',
            background=self.theme.colors['secondary'],
            foreground=self.theme.colors['text'],
            padding=[10, 5],
            borderwidth=0
        )

        style.map(
            'TNotebook.Tab',
            background=[
                ('selected', self.theme.colors['accent']),
                ('active', self.theme.colors['button_hover'])
            ],
            foreground=[
                ('selected', 'black'),
                ('active', self.theme.colors['text'])
            ]
        )

    def _on_search_changed(self, *args):
        """Wird aufgerufen, wenn sich der Suchtext ändert."""
        try:
            search_text = self.search_var.get().lower()

            # Wenn der Suchtext leer ist, lade alle Konversationen neu
            if not search_text.strip():
                self.load_conversation_list()
                return

            # Durchsuche die Konversationen nach dem Suchtext
            if not hasattr(self, 'all_conversations'):
                return

            self.conversation_list.delete(0, tk.END)

            for conv_id, title, preview, timestamp in self.all_conversations:
                if (search_text in title.lower()) or (
                        search_text in preview.lower()):
                    # Formatiere die Anzeige
                    display_text = f"{title} - {preview}"
                    self.conversation_list.insert(
                        tk.END, (conv_id, display_text, timestamp))

            logger.debug(f"Suche nach '{search_text}' abgeschlossen")

        except Exception as e:
            logger.error(f"Fehler bei der Suche: {e}", exc_info=True)

    def _setup_sidebar(self):
        """Erstellt die Seitenleiste mit der Konversationsliste."""
        try:
            # Frame für die Seitenleiste
            self.sidebar = ttk.Frame(
                self.main_container,
                width=250,
                style='Sidebar.TFrame'
            )
            self.sidebar.pack(side=tk.LEFT, fill='y', padx=(0, 10), pady=0)
            self.sidebar.pack_propagate(False)

            # Schaltfläche für neuen Chat
            new_chat_btn = ttk.Button(
                self.sidebar,
                text="Neuer Chat",
                command=self.new_chat,
                style='Sidebar.TButton'
            )
            new_chat_btn.pack(fill='x', padx=5, pady=5)

            # Suchfeld
            search_frame = ttk.Frame(self.sidebar, style='Search.TFrame')
            search_frame.pack(fill='x', padx=5, pady=5)

            self.search_var = tk.StringVar()
            # Verzögerte Suche, um zu verhindern, dass bei jedem Tastendruck
            # gesucht wird
            self.search_var.trace(
                'w', lambda *args: self.root.after(300, self._on_search_changed))

            search_entry = ttk.Entry(
                search_frame,
                textvariable=self.search_var,
                style='Search.TEntry',
                font=self.theme.fonts['small']
            )
            search_entry.pack(fill='x', padx=5, pady=0)
            search_entry.bind(
                '<KeyRelease>',
                lambda e: self._on_search_changed())

            # Scrollbare Liste der Konversationen
            list_frame = ttk.Frame(self.sidebar, style='List.TFrame')
            list_frame.pack(fill='both', expand=True, padx=5, pady=5)

            # Scrollbar für die Liste
            scrollbar = ttk.Scrollbar(list_frame, orient='vertical')
            scrollbar.pack(side='right', fill='y')

            # Liste der Konversationen
            self.conversation_list = tk.Listbox(
                list_frame,
                yscrollcommand=scrollbar.set,
                selectmode='single',
                bg=self.theme.colors['secondary'],
                fg=self.theme.colors['text'],
                selectbackground=self.theme.colors['accent'],
                selectforeground=self.theme.colors['text'],
                borderwidth=0,
                highlightthickness=0,
                font=self.theme.fonts['small'],
                relief='flat'
            )
            self.conversation_list.pack(
                fill='both', expand=True, padx=0, pady=0)
            scrollbar.config(command=self.conversation_list.yview)

            # Doppelklick auf eine Konversation lädt sie
            self.conversation_list.bind('<Double-1>', self.load_chat)

            # Rechte Maustaste für Kontextmenü
            if hasattr(self, '_show_conversation_context_menu'):
                self.conversation_list.bind(
                    '<Button-3>', self._show_conversation_context_menu)

            logger.debug("Seitenleiste erfolgreich erstellt")

        except Exception as e:
            logger.error(
                f"Fehler beim Erstellen der Seitenleiste: {e}",
                exc_info=True)
            raise

    def speak_text(self, text, language='de', block=True):
        """Spricht den gegebenen Text in der angegebenen Sprache mit gTTS.

        Args:
            text (str): Der zu sprechende Text
            language (str): Sprachcode (z.B. 'de' für Deutsch, 'en' für Englisch)
            block (bool): Wenn True, blockiert die Methode, bis die Wiedergabe beendet ist
        """
        if not text:
            return

        try:
            # Initialisiere pygame Mixer, falls noch nicht geschehen
            if not pygame.mixer.get_init():
                pygame.mixer.init()

            # Erstelle eine temporäre Datei für die Sprachausgabe
            temp_file = os.path.join(
                tempfile.gettempdir(),
                f'tts_output_{uuid.uuid4()}.mp3')

            # Erstelle die Sprachausgabe mit gTTS
            tts = gTTS(text=text, lang=language, slow=False)
            tts.save(temp_file)

            # Wiedergabe der generierten Audiodatei
            def play_audio():
                try:
                    # Lade und spiele die Audiodatei ab
                    pygame.mixer.music.load(temp_file)
                    pygame.mixer.music.play()

                    # Warte bis die Wiedergabe beendet ist, wenn block=True
                    if block:
                        while pygame.mixer.music.get_busy():
                            pygame.time.Clock().tick(10)

                    # Lösche die temporäre Datei nach der Wiedergabe
                    try:
                        os.remove(temp_file)
                    except Exception as e:
                        logger.error(
                            f'Fehler beim Löschen der temporären Audiodatei: {e}')

                except Exception as e:
                    logger.error(f'Fehler bei der Sprachausgabe: {e}')

            # Starte die Wiedergabe in einem separaten Thread, um die GUI nicht
            # zu blockieren
            threading.Thread(target=play_audio, daemon=True).start()

        except Exception as e:
            logger.error(f'Fehler in speak_text: {e}')

    def _on_send_message(self, event=None):
        """Wird aufgerufen, wenn der Benutzer eine Nachricht sendet."""
        try:
            # Prüfe, ob das Eingabefeld existiert
            if not hasattr(self, 'user_input') or not hasattr(self.user_input, 'get'):
                print("Fehler: Eingabefeld nicht gefunden")
                return
                
            try:
                # Versuche den Text zu holen (für ttk.Entry)
                message = self.user_input.get().strip()
            except Exception as e:
                print(f"Fehler beim Lesen der Eingabe: {e}")
                return
            
            if not message:
                return

            # Deaktiviere die Benutzerinteraktion während der Verarbeitung
            try:
                if hasattr(self.user_input, 'configure'):
                    self.user_input.configure(state="disabled")
                
                if hasattr(self, 'send_button') and self.send_button and hasattr(self.send_button, 'configure'):
                    self.send_button.configure(state="disabled")
            except Exception as e:
                print(f"Warnung: Konnte UI-Elemente nicht deaktivieren: {e}")

            try:
                # Zeige die Nachricht des Benutzers an
                self._display_message("user", message)

                # Lösche den Eingabetext
                self.user_input.delete(0, tk.END)
            except Exception as e:
                print(f"Fehler bei der Nachrichtenverarbeitung: {e}")
                return

            # Stelle sicher, dass eine Konversation existiert
            if self.current_conversation_id is None:
                try:
                    # Erstelle eine neue Konversation mit dem ersten Satz der Nachricht als Titel
                    title = message[:30] + "..." if len(message) > 30 else message
                    self.current_conversation_id = self.database.create_conversation(
                        user_id=self.current_user, title=title)
                    # Aktualisiere die Konversationsliste
                    self.load_conversation_list()
                except Exception as e:
                    logger.error(
                        f"Fehler beim Erstellen einer neuen Konversation: {e}",
                        exc_info=True)
                    self._display_message(
                        "system", f"Fehler: Konnte keine neue Konversation erstellen: {str(e)}")
                    self.user_input.configure(state="normal")
                    if hasattr(self, 'send_button') and self.send_button:
                        self.send_button.configure(state="normal")
                    return

            # Starte einen Thread für die Antwortgenerierung
            threading.Thread(
                target=self._process_user_message,
                args=(message,),
                daemon=True
            ).start()

        except Exception as e:
            logger.error(
                f"Fehler beim Senden der Nachricht: {e}",
                exc_info=True)
            if hasattr(self, 'status_bar'):
                self.status_bar.configure(
                    text=f"Fehler beim Senden der Nachricht: {str(e)}")
            self._display_message("system", f"Fehler: {str(e)}")
            self.user_input.configure(state="normal")
            if hasattr(self, 'send_button') and self.send_button:
                self.send_button.configure(state="normal")

    def _process_user_message(self, message):
        """Verarbeitet die Benutzernachricht und generiert eine Antwort.

        Args:
            message: Die zu verarbeitende Benutzernachricht
        """
        try:
            # Zeige "Wird verarbeitet..." an
            self.status_bar.configure(text="Wird verarbeitet...")

            # Speichere die Nachricht in der Datenbank, falls eine Konversation
            # existiert
            if self.current_conversation_id:
                try:
                    self.database.add_message(
                        conversation_id=self.current_conversation_id,
                        role="user",
                        content=message
                    )
                except Exception as e:
                    logger.error(
                        f"Fehler beim Speichern der Nachricht: {e}",
                        exc_info=True)

            # Generiere eine Antwort mit dem ModelManager
            response = self.model_manager.generate_response(
                message,
                conversation_history=self.conversation_history
            )

            # Zeige die Antwort an
            self._display_message("assistant", response)

            # Aktiviere die Sprachausgabe für die Antwort
            self.speak_text(response)

            # Speichere die Antwort in der Datenbank, falls eine Konversation
            # existiert
            if self.current_conversation_id:
                try:
                    self.database.add_message(
                        conversation_id=self.current_conversation_id,
                        role="assistant",
                        content=response
                    )

                    # Aktualisiere die letzte Aktivität der Konversation
                    self.database.update_conversation(
                        conversation_id=self.current_conversation_id,
                        updated_at=datetime.now().isoformat()
                    )
                except Exception as e:
                    logger.error(
                        f"Fehler beim Speichern der Antwort: {e}",
                        exc_info=True)

            # Aktualisiere die Konversationsliste, um die letzte Aktivität
            # anzuzeigen
            self.load_conversation_list()

            # Aktualisiere die Statusleiste
            self.status_bar.configure(text="Bereit")

        except Exception as e:
            error_msg = f"Fehler bei der Verarbeitung: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.status_bar.configure(text=error_msg)
            self._display_message("system", error_msg)

        finally:
            # Aktiviere das Eingabefeld wieder
            self.user_input.configure(state="normal")
            self.send_button.configure(state="normal")
            self.user_input.focus_set()

    # ...
        responses = [
            "Ich habe Ihre Nachricht erhalten und verarbeite sie.",
            "Interessante Frage! Lassen Sie mich darüber nachdenken...",
            "Danke für Ihre Nachricht. Ich werde das berücksichtigen.",
            "Verstanden! Gibt es noch etwas, wobei ich Ihnen helfen kann?",
            "Ich habe Ihre Anfrage erhalten und werde sie so schnell wie möglich bearbeiten."]
        return random.choice(responses)

    def _setup_chat_area(self):
        """Richtet den Hauptbereich für den Chat ein."""
        try:
            # Frame für den Chat-Bereich
            self.chat_frame = ttk.Frame(
                self.main_container,
                style='Chat.TFrame'
            )
            self.chat_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            # Scrollbar für den Chat
            chat_scrollbar = ttk.Scrollbar(self.chat_frame)
            chat_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            # Text-Widget für den Chat-Verlauf
            self.chat_display = tk.Text(
                self.chat_frame,
                wrap=tk.WORD,
                yscrollcommand=chat_scrollbar.set,
                bg=self.theme.colors['primary'],
                fg=self.theme.colors['text'],
                insertbackground=self.theme.colors['text'],
                font=self.theme.fonts['normal'],
                padx=10,
                pady=10,
                state='disabled',
                relief='flat',
                highlightthickness=0
            )
            self.chat_display.pack(fill=tk.BOTH, expand=True)
            chat_scrollbar.config(command=self.chat_display.yview)

            # Frame für die Benutzereingabe
            input_frame = ttk.Frame(self.chat_frame, style='Input.TFrame')
            input_frame.pack(fill=tk.X, pady=(5, 0))

            # Eingabefeld für Benutzer
            self.user_input = ttk.Entry(
                input_frame,
                font=self.theme.fonts['normal']
            )
            self.user_input.pack(
                side=tk.LEFT,
                fill=tk.X,
                expand=True,
                padx=(
                    0,
                    5))
            self.user_input.bind('<Return>', self._on_send_message)

            # Senden-Button
            self.send_button = ttk.Button(
                input_frame,
                text="Senden",
                command=self._on_send_message,
                style='Send.TButton'
            )
            self.send_button.pack(side=tk.RIGHT)

            # Konfiguriere die Tags für die Formatierung
            self.chat_display.tag_configure('user', foreground='#4fc3f7')
            self.chat_display.tag_configure('assistant', foreground='#81c784')
            self.chat_display.tag_configure('system', foreground='#ff8a65')

            logger.debug("Chat-Bereich erfolgreich eingerichtet")

        except Exception as e:
            logger.error(
                f"Fehler beim Einrichten des Chat-Bereichs: {e}",
                exc_info=True)
            raise

    def _setup_status_bar(self):
        """Erstellt die Statusleiste am unteren Fensterrand."""
        try:
            # Erstelle einen Frame für die Statusleiste
            self.status_bar = ttk.Label(
                self.root,
                text="Bereit",
                relief=tk.SUNKEN,
                anchor=tk.W,
                style='Status.TLabel'
            )
            self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

            # Füge ein Label für die Modellinformationen hinzu
            self.model_status = ttk.Label(
                self.status_bar,
                text="Modell: Kein Modell geladen",
                anchor=tk.E,
                style='Status.TLabel'
            )
            self.model_status.pack(side=tk.RIGHT, padx=5)

            # Füge ein Label für die Systeminformationen hinzu
            self.system_status = ttk.Label(
                self.status_bar,
                text="System: Bereit",
                anchor=tk.W,
                style='Status.TLabel'
            )
            self.system_status.pack(side=tk.LEFT, padx=5)

            logger.debug("Statusleiste erfolgreich eingerichtet")

        except Exception as e:
            logger.error(
                f"Fehler beim Einrichten der Statusleiste: {e}",
                exc_info=True)
            raise

    def _display_message(self, role, message):
        """Zeigt eine Nachricht im Chat-Bereich an.

        Args:
            role (str): Die Rolle des Absenders ('user', 'assistant' oder 'system')
            message (str): Die anzuzeigende Nachricht
        """
        try:
            # Stelle sicher, dass das Chat-Display existiert
            if not hasattr(self, 'chat_display'):
                logger.error("Chat-Display existiert nicht")
                return

            # Aktiviere das Text-Widget zum Bearbeiten
            self.chat_display.config(state='normal')

            # Lösche den aktuellen Inhalt, wenn es sich um eine neue
            # Konversation handelt
            if not hasattr(
                    self,
                    '_conversation_started') or not self._conversation_started:
                self.chat_display.delete(1.0, tk.END)
                self._conversation_started = True

            # Bestimme das Präfix basierend auf der Rolle
            if role.lower() == 'user':
                prefix = "Sie: "
                tag = "user"
                fg_color = "#e1f5fe"  # Hellblau für Benutzer
                justify = "right"
            elif role.lower() == 'assistant':
                prefix = "Jarvis: "
                tag = "assistant"
                fg_color = "#f5f5f5"  # Hellgrau für Assistenten
                justify = "left"
            else:  # system
                prefix = "System: "
                tag = "system"
                fg_color = "#ffebee"  # Hellrot für Systemnachrichten
                justify = "center"

            # Füge die Nachricht zum Chat hinzu
            self.chat_display.insert(tk.END, f"{prefix}\n{message}\n\n", tag)

            # Konfiguriere das Tag für die Formatierung
            self.chat_display.tag_configure(
                tag, foreground=fg_color, justify=justify)

            # Deaktiviere das Text-Widget wieder
            self.chat_display.config(state='disabled')

            # Scrolle zum Ende der Nachricht
            self.chat_display.see(tk.END)

            # Fokussiere das Eingabefeld
            if hasattr(self, 'user_input') and self.user_input.winfo_exists():
                self.user_input.focus_set()

            logger.debug(f"Nachricht von {role} angezeigt")

        except Exception as e:
            logger.error(
                f"Fehler beim Anzeigen der Nachricht: {e}",
                exc_info=True)
            # Versuche, das Chat-Display wieder in einen konsistenten Zustand
            # zu versetzen
            try:
                if hasattr(
                        self,
                        'chat_display') and self.chat_display.winfo_exists():
                    self.chat_display.config(state='disabled')
            except BaseException:
                pass

    def _center_window(self):
        """Zentriert das Hauptfenster auf dem Bildschirm."""
        try:
            # Aktualisiere die Fenstermanager-Aufgaben, um die tatsächliche
            # Fenstergröße zu erhalten
            self.root.update_idletasks()

            # Fenstergröße abrufen
            window_width = self.root.winfo_width()
            window_height = self.root.winfo_height()

            # Bildschirmgröße abrufen
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()

            # Position berechnen, um das Fenster zu zentrieren
            x = (screen_width // 2) - (window_width // 2)
            y = (screen_height // 2) - (window_height // 2)

            # Fensterposition setzen
            self.root.geometry(f'+{x}+{y}')

            logger.debug("Fenster wurde zentriert")

        except Exception as e:
            logger.error(
                f"Fehler beim Zentrieren des Fensters: {e}",
                exc_info=True)
            # Fortfahren, auch wenn das Zentrieren fehlschlägt

    def setup_gui(self):
        """Initialisiert die Benutzeroberfläche."""
        # Wende das JARVIS-Design an
        self._apply_jarvis_theme()

        # Erstelle die Menüleiste
        self._create_menu_bar()

        # Hauptcontainer erstellen
        self.main_container = ttk.Frame(self.root, style='TFrame')
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Sidebar für Konversationen
        self._setup_sidebar()

        # Hauptbereich für Chat
        self._setup_chat_area()

        # Statusleiste
        self._setup_status_bar()

        # Konfiguriere die Fenstergröße
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)

        # Zentriere das Fenster
        self._center_window()

        # Lade die Konversationsliste
        self.load_conversation_list()

        # Zeige eine Willkommensnachricht an, wenn kein Chat geladen wurde
        if not hasattr(
                self,
                'conversation_history') or not self.conversation_history:
            welcome_message = (
                "Willkommen zu J.A.R.V.I.S. KI Assistent!\n\n"
                "Stellen Sie mir eine Frage oder geben Sie einen Befehl ein. "
                "Ich bin Ihr persönlicher KI-Assistent und helfe Ihnen gerne weiter.\n\n"
                "Verfügbare Befehle:\n"
                "- /neu - Startet einen neuen Chat\n"
                "- /speichern - Speichert den aktuellen Chat\n"
                "- /hilfe - Zeigt die Hilfe an\n"
                "- /einstellungen - Öffnet die Einstellungen")
            self._display_message("assistant", welcome_message)

        # Setze den Fokus auf das Eingabefeld
        if hasattr(self, 'user_input'):
            self.user_input.focus_set()

        logger.info("GUI erfolgreich initialisiert")

    def _apply_jarvis_theme(self):
        """Wendet das JARVIS-Design auf die Anwendung an."""
        # Fenster-Konfiguration
        self.root.title("J.A.R.V.I.S. - Just A Rather Very Intelligent System")
        self.root.minsize(1000, 700)
        self.root.configure(bg=self.theme.colors['primary'])

        # Schriftarten setzen
        self.root.option_add('*Font', self.theme.fonts['normal'])

        # Stile konfigurieren
        self._configure_styles()

        # Versuche, das Fenster-Icon zu setzen
        try:
            self.root.iconbitmap("assets/icon.ico")
        except Exception as e:
            logger.warning(f"Konnte das Fenster-Icon nicht laden: {e}")

    def __init__(self, root):
        """Initialisiert die Hauptanwendung."""
        self.root = root

        # JARVIS-Design anwenden
        self.theme = JarvisTheme()

        # Konfiguration initialisieren
        self.config = ConfigManager()

        # Initialisiere die Datenbank mit einer gemeinsamen Verbindung
        self.db_connection = sqlite3.connect(
            "chat_history.db", check_same_thread=False)
        # Ermöglicht den Zugriff auf Spalten über Namen
        self.db_connection.row_factory = sqlite3.Row
        self.database = DatabaseManager(db_path="chat_history.db")

        # Standard-Benutzer-ID (kann durch ein Login-System ersetzt werden)
        self.current_user = 1

        # Aktuelle Konversation und Kontext
        self.current_conversation_id = None
        self.conversation_history = []
        self.current_conversation_context = ""
        self.current_context = "general"

        # Benutzereinstellungen
        self.user_preferences = {
            'name': 'Benutzer',
            'language': 'Deutsch',
            'theme': 'dark',
            'font_size': 12,
            'max_tokens': 2048,
            'temperature': 0.7
        }

        # Modell-Manager initialisieren
        self.model_manager = EnhancedModelManager(self.config)

        # Session-Manager initialisieren
        self.session_manager = SessionManager()
        self.current_session = self.session_manager.get_session()

        # XTTS-Manager initialisieren
        self.xtts_manager = XTTSManager()

        # Initialisiere die Benutzeroberfläche
        self.setup_gui()

        # Lade die Chat-Übersicht
        self.load_conversation_list()

        # Lade die letzte Konversation, falls vorhanden
        self.load_latest_conversation()

        # Stelle sicher, dass die Datenbankverbindung beim Beenden geschlossen
        # wird
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        """Wird aufgerufen, wenn das Fenster geschlossen wird."""
        try:
            # Speichere den aktuellen Chat
            if hasattr(
                    self,
                    'conversation_history') and self.conversation_history:
                self.save_chat()

            # Schließe die Datenbankverbindung
            if hasattr(self, 'database') and self.database:
                self.database.close()

            # Beende die Anwendung
            self.root.destroy()

        except Exception as e:
            logger.error(
                f"Fehler beim Beenden der Anwendung: {e}",
                exc_info=True)
            # Versuche trotzdem zu beenden
            try:
                if hasattr(self, 'db_connection'):
                    self.db_connection.close()
            except BaseException:
                pass
            self.root.destroy()

    def load_latest_conversation(self):
        """Lädt die letzte geführte Konversation, falls vorhanden."""
        try:
            # Hole die letzte Konversation aus der Datenbank
            conversations = self.database.get_conversations(
                user_id=self.current_user,
                limit=1
            )

            if conversations:
                # Lade die letzte Konversation
                latest_conv = conversations[0]
                self.load_conversation(latest_conv['id'])

        except Exception as e:
            logger.error(
                f"Fehler beim Laden der letzten Konversation: {e}",
                exc_info=True)
            # Kein Fehler anzeigen, da dies beim ersten Start normal ist

    def load_conversation(self, conversation_id):
        """Lädt eine bestimmte Konversation in die Benutzeroberfläche.

        Args:
            conversation_id: Die ID der zu ladenden Konversation
        """
        try:
            # Setze die aktuelle Konversations-ID
            self.current_conversation_id = conversation_id

            # Lösche die aktuelle Nachrichtenansicht
            if hasattr(self, 'chat_display'):
                self.chat_display.configure(state='normal')
                self.chat_display.delete('1.0', tk.END)

            # Lade die Konversation aus der Datenbank
            messages = self.database.get_messages(
                conversation_id=conversation_id)

            # Zeige die Nachrichten in der Benutzeroberfläche an
            for msg in messages:
                role = msg['role']
                content = msg['content']

                if role == 'user':
                    self._display_message("user", content)
                else:
                    self._display_message("assistant", content)

            # Aktualisiere die Titelzeile mit dem Konversationstitel
            if hasattr(self, 'root'):
                conv = self.database.get_conversation(conversation_id)
                if conv and 'title' in conv:
                    self.root.title(f"Jarvis KI - {conv['title']}")

        except Exception as e:
            logger.error(
                f"Fehler beim Laden der Konversation: {e}",
                exc_info=True)
            if hasattr(self, 'status_bar'):
                self.status_bar.config(
                    text=f"Fehler beim Laden der Konversation: {str(e)}")

    def new_chat(self, event=None):
        """Startet einen neuen Chat."""
        try:
            # Speichere den aktuellen Chat, falls vorhanden
            if hasattr(
                    self,
                    'current_conversation_id') and self.current_conversation_id:
                self.save_chat()

            # Setze die aktuelle Konversation zurück
            self.current_conversation_id = None
            self.conversation_history = []

            # Lösche den aktuellen Chat-Anzeigebereich
            if hasattr(
                    self,
                    'chat_container') and self.chat_container.winfo_exists():
                for widget in self.chat_container.winfo_children():
                    widget.destroy()

            # Erstelle einen neuen Chat-Anzeigebereich, falls nicht vorhanden
            if not hasattr(
                    self,
                    'chat_container') or not self.chat_container.winfo_exists():
                self._setup_chat_area()

            # Aktualisiere die Chat-Liste in der Seitenleiste
            self.load_conversation_list()

            # Fokussiere das Eingabefeld
            if hasattr(self, 'user_input') and self.user_input.winfo_exists():
                self.user_input.focus_set()

            # Statusmeldung anzeigen
            self.status_bar.configure(text="Neuer Chat gestartet")

        except Exception as e:
            error_msg = f"Fehler beim Starten eines neuen Chats: {e}"
            logger.error(error_msg, exc_info=True)
            messagebox.showerror("Fehler", error_msg)

    def delete_all_chats(self):
        """Löscht alle Chats und aktualisiert die Benutzeroberfläche."""
        try:
            # Bestätigungsdialog anzeigen
            if not messagebox.askyesno(
                "Löschen bestätigen",
                "Möchten Sie wirklich alle Chats löschen? Diese Aktion kann nicht rückgängig gemacht werden."
            ):
                return

            # Lösche alle Konversationen des aktuellen Benutzers in der
            # Datenbank
            if hasattr(self, 'current_user') and self.current_user:
                success = self.database.delete_all_conversations(
                    self.current_user)
                if not success:
                    raise Exception(
                        "Fehler beim Löschen der Konversationen in der Datenbank")

            # Zurücksetzen der UI-Elemente
            self.conversation_history = []
            self.current_conversation_id = None

            # Lösche alle Einträge in der Chat-Liste
            for widget in self.chat_list_frame.winfo_children():
                widget.destroy()

            # Lösche den Chat-Verlauf
            self.chat_history.delete(1.0, tk.END)

            # Statusmeldung anzeigen
            self.status_bar.config(text="Alle Chats wurden gelöscht")

            # Neuen leeren Chat starten
            self.new_chat()

            # Chat-Liste aktualisieren
            self.load_conversation_list()

        except Exception as e:
            error_msg = f"Fehler beim Löschen der Chats: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.status_bar.config(text=error_msg)

    def save_chat(self):
        """Speichert den aktuellen Chat in der Datenbank."""
        try:
            if not hasattr(
                    self,
                    'conversation_history') or not self.conversation_history:
                return  # Nichts zu speichern

            # Erstelle eine Vorschau aus der ersten Nachricht
            preview = ""
            for msg in self.conversation_history:
                if msg.role == 'user' and msg.content:
                    # Erste 100 Zeichen als Vorschau
                    preview = str(msg.content)[:100]
                    break

            if not preview and self.conversation_history:
                preview = "Neuer Chat"

            # Speichere die Konversation in der Datenbank
            if hasattr(
                    self,
                    'current_conversation_id') and self.current_conversation_id:
                # Aktualisiere bestehende Konversation
                for msg in self.conversation_history:
                    self.database.save_message(
                        self.current_conversation_id, msg)
            else:
                # Erstelle neue Konversation
                self.current_conversation_id = self.database.save_conversation(
                    user_id=self.current_user,
                    messages=self.conversation_history,
                    preview=preview
                )

            # Aktualisiere die Chat-Liste
            self.load_conversation_list()

        except Exception as e:
            logger.error(
                f"Fehler beim Speichern des Chats: {e}",
                exc_info=True)
            messagebox.showerror(
                "Fehler", f"Chat konnte nicht gespeichert werden: {e}")

            # Füge einen Button für neuen Chat hinzu
            new_chat_btn = ctk.CTkButton(
                self.chat_list_frame,
                text="+ Neuer Chat",
                command=self.new_chat,
                fg_color="#2E8B57",  # Dunkelgrün
                hover_color="#3CB371"  # Etwas helleres Grün beim Hovern
            )
            new_chat_btn.pack(pady=(0, 10), padx=5, fill='x')

            # Füge jeden Chat als anklickbares Element hinzu
            for conv in conversations:
                # Kürze den Vorschautext, falls zu lang
                preview = conv.get('preview', 'Keine Vorschau')
                if len(preview) > 30:
                    preview = preview[:27] + '...'

                # Erstelle ein Frame für jeden Chat-Eintrag
                chat_frame = ctk.CTkFrame(
                    self.chat_list_frame,
                    corner_radius=5,
                    fg_color="#3a3a3a" if hasattr(self, 'current_conversation_id') and
                    self.current_conversation_id == conv['id'] else "#2d2d2d"
                )
                chat_frame.pack(pady=2, padx=5, fill='x')

                # Füge ein Label mit Vorschautext und Datum hinzu
                timestamp = datetime.fromisoformat(
                    conv.get(
                        'updated_at',
                        datetime.now().isoformat())).strftime('%d.%m.%Y %H:%M')
                btn = ctk.CTkButton(
                    chat_frame,
                    text=f"{preview}\n{timestamp}",
                    command=lambda cid=conv['id']: self.load_conversation(cid),
                    anchor='w',
                    fg_color="transparent",
                    hover_color="#3a3a3a",
                    text_color=("#ffffff" if hasattr(self, 'current_conversation_id') and
                                self.current_conversation_id == conv['id'] else "#cccccc"),
                    corner_radius=5,
                    height=50
                )
                btn.pack(fill='both', expand=True, padx=2, pady=2)

            self.encryption = None
            if self.config.get(
                'FEATURES',
                'encryption',
                    fallback='false').lower() == 'true':
                try:
                    self.encryption = EncryptionManager()
                    logger.info("Verschlüsselung aktiviert")
                except Exception as e:
                    logger.error(
                        f"Fehler bei der Initialisierung der Verschlüsselung: {e}")

            # 3. Datenbank initialisieren
            try:
                self.database = DatabaseManager()
                logger.info("Datenbank initialisiert")
            except Exception as e:
                logger.error(f"Fehler bei der Datenbankinitialisierung: {e}")
                raise RuntimeError(
                    "Konnte die Datenbank nicht initialisieren") from e

            # 4. Kontext-Scoring und Langzeitgedächtnis initialisieren
            self.context_scorer = ContextScorer()
            self.long_term_memory = LongTermMemory("memory.json")

            # 5. Modell-Manager initialisieren
            try:
                self.model_manager = EnhancedModelManager(self.config)
                logger.info("Modell-Manager initialisiert")
            except Exception as e:
                logger.error(
                    f"Fehler bei der Modell-Manager-Initialisierung: {e}")
                raise RuntimeError(
                    "Konnte den Modell-Manager nicht initialisieren") from e

            # 6. Spracherkennung initialisieren (falls aktiviert)
            self.speech_manager = None
            if self.config.get(
                'FEATURES',
                'speech_recognition',
                    fallback='false').lower() == 'true':
                try:
                    self.speech_manager = SpeechManager()
                    logger.info("Spracherkennung initialisiert")
                except Exception as e:
                    logger.error(
                        f"Fehler bei der Spracherkennungsinitialisierung: {e}")

            # 7. Wissens-API initialisieren
            try:
                self.wissens_api = WissensAPI(sprache='de')
                logger.info("Wissens-API initialisiert")
            except Exception as e:
                logger.error(
                    f"Fehler bei der Initialisierung der Wissens-API: {e}")
                self.wissens_api = None

            # 8. Datensatz-Manager initialisieren
            try:
                self.datensatz_manager = DatensatzManager()
                logger.info("Datensatz-Manager initialisiert")
            except Exception as e:
                logger.error(
                    f"Fehler bei der Initialisierung des Datensatz-Managers: {e}")
                self.datensatz_manager = None

            # 9. Plugin-System initialisieren
            try:
                self.plugin_manager = PluginManager()
                logger.info("Plugin-System initialisiert")
            except Exception as e:
                logger.error(
                    f"Fehler bei der Plugin-System-Initialisierung: {e}")
                self.plugin_manager = None

            # 10. Chat-Export initialisieren
            self.chat_exporter = ChatExporter()

            # Sitzungs- und Kontextverwaltung
            self.session_manager = SessionManager(
                session_timeout_minutes=60)  # 1 Stunde Timeout
            self.clarification_manager = ClarificationManager()
            self.current_session = self.session_manager.get_session()

            # Anwendungsstatus
            self.current_user = "default_user"
            self.current_conversation_id = None
            self.conversation_history: List[ChatMessage] = []
            self.is_generating = False
            self.awaiting_clarification = False
            self.pending_clarification = None

            # 11. GUI initialisieren
            self.setup_gui()

            # 12. Letzte Unterhaltung laden
            try:
                self.load_conversation_history()
            except Exception as e:
                logger.error(
                    f"Fehler beim Laden der Unterhaltungshistorie: {e}")

            # 13. Performance-Monitoring starten
            try:
                self.model_manager.performance_monitor.start_monitoring()
            except Exception as e:
                logger.error(
                    f"Fehler beim Starten des Performance-Monitorings: {e}")

            logger.info("Jarvis KI-Assistent erfolgreich initialisiert")

        except Exception as e:
            logger.critical(
                f"Kritischer Fehler bei der Anwendungsinitialisierung: {e}",
                exc_info=True)
            raise RuntimeError(
                "Fehler bei der Anwendungsinitialisierung") from e

    async def generate_response_async(self, user_input: str) -> str:
        """Generiert eine Antwort auf die Benutzereingabe (asynchron)."""
        try:
            # Bestes Modell für die Eingabe auswählen
            model = self.model_manager.get_best_model(
                user_input, self.conversation_history)

            # System-Prompt erstellen
            system_prompt = self._create_system_prompt()

            # Füge den System-Prompt zur Konversationshistorie hinzu, falls
            # nicht vorhanden
            if not self.conversation_history or self.conversation_history[-1].role != 'system':
                self.conversation_history.append(ChatMessage(
                    role='system',
                    content=system_prompt,
                    timestamp=datetime.now()
                ))

            # Füge die Benutzereingabe zur Konversationshistorie hinzu
            user_message = ChatMessage(
                role='user',
                content=user_input,
                timestamp=datetime.now()
            )
            self.conversation_history.append(user_message)

            # Generiere die Antwort mit dem Modell-Manager
            response = await self.model_manager.generate_response_async(
                prompt=user_input,
                conversation_history=self.conversation_history
            )

            # Füge die Antwort zur Konversationshistorie hinzu
            self.conversation_history.append(ChatMessage(
                role='assistant',
                content=response,
                timestamp=datetime.now()
            ))

            # Antwort zurücksenden
            return response

        except Exception as e:
            logger.error(
                f"Fehler bei der Antwortgenerierung: {e}",
                exc_info=True)
            return f"Entschuldigung, bei der Verarbeitung ist ein Fehler aufgetreten: {str(e)}"

    def _setup_styles(self):
        """Konfiguriert die Stile für die Anwendung."""
        # Standard-Schriftart für die gesamte Anwendung
        default_font = ('Arial', 12)

        # Konfiguriere das Standard-Theme
        ctk.set_appearance_mode("dark")  # Dunkles Design als Standard
        ctk.set_default_color_theme("blue")  # Blauer Farbton für Akzente

        # Konfiguriere die Standard-Schriftart für CTk-Widgets
        ctk.CTkLabel._font = default_font
        ctk.CTkButton._font = default_font
        ctk.CTkEntry._font = default_font
        ctk.CTkTextbox._font = default_font

        # Konfiguriere die Farben für verschiedene Zustände
        self.colors = {
            'primary': '#1e88e5',  # Hauptfarbe für Buttons und Akzente
            'primary_hover': '#1565c0',  # Hover-Farbe für Buttons
            'background': '#121212',  # Haupt-Hintergrundfarbe
            'surface': '#1e1e1e',  # Flächen wie Karten und Container
            'on_surface': '#ffffff',  # Textfarbe auf Oberflächen
            'on_background': '#e0e0e0',  # Textfarbe auf Hintergrund
            'error': '#cf6679',  # Fehlermeldungen
            'success': '#4caf50',  # Erfolgsmeldungen
            'warning': '#ff9800',  # Warnmeldungen
            'disabled': '#616161'  # Deaktivierte Elemente
        }

        # Konfiguriere das Styling für bestimmte Widgets
        self.style = ttk.Style()

        # Konfiguriere das Styling für Buttons
        self.style.configure('TButton',
                             font=default_font,
                             padding=6)

        # Konfiguriere das Styling für Eingabefelder
        self.style.configure('TEntry',
                             font=default_font,
                             padding=5)

        # Konfiguriere das Styling für Labels
        self.style.configure('TLabel',
                             font=default_font,
                             background=self.colors['background'],
                             foreground=self.colors['on_background'])

        # Konfiguriere das Styling für Frames
        self.style.configure('TFrame',
                             background=self.colors['background'])

        # Konfiguriere das Styling für die Menüleiste
        self.style.configure('TMenubutton',
                             font=default_font)

        # Konfiguriere das Styling für die Statusleiste
        self.status_style = ttk.Style()
        self.status_style.configure('Status.TLabel',
                                    font=('Arial', 10),
                                    background=self.colors['surface'],
                                    foreground=self.colors['on_surface'])

    def setup_menu(self):
        """Erstellt das Hauptmenü der Anwendung."""
        # Hauptmenü erstellen (mit Standard-Tkinter-Menü, da CTk kein eigenes
        # Menü hat)
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # Datei-Menü
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Datei", menu=file_menu)
        file_menu.add_command(
            label="Neuer Chat",
            command=self.new_chat,
            accelerator="Strg+N")
        file_menu.add_command(
            label="Chat speichern",
            command=self.save_chat,
            accelerator="Strg+S")
        file_menu.add_separator()
        file_menu.add_command(
            label="Alle Chats löschen",
            command=self.delete_all_chats)
        file_menu.add_separator()
        file_menu.add_command(
            label="Einstellungen",
            command=self.show_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Beenden", command=self.on_close)

        # Bearbeiten-Menü
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Bearbeiten", menu=edit_menu)
        edit_menu.add_command(
            label="Chat-Verlauf",
            command=self.show_chat_history)
        edit_menu.add_command(
            label="Einstellungen",
            command=self.show_settings)

        # Hilfe-Menü
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Hilfe", menu=help_menu)

        # Füge die Benutzereingabe zur Konversationshistorie hinzu
        user_message = ChatMessage(
            role='user',
            content=user_input,
            timestamp=datetime.now()
        )
        self.conversation_history.append(user_message)

    def _create_context_from_history(self) -> str:
        """Erstellt einen Kontext aus der Unterhaltungshistorie.

        Returns:
            str: Der generierte Kontext als String
        """
        if not self.conversation_history:
            return ""

        context = []
        # Nur die letzten 5 Nachrichten berücksichtigen
        for msg in self.conversation_history[-5:]:
            role = "User" if msg.role == "user" else "Assistant"
            context.append(f"{role}: {msg.content}")

        return "\n".join(context)

    def load_conversation_list(self):
        """Lädt die Liste der gespeicherten Konversationen in der Seitenleiste.

        Hinweis: Es wird immer nur ein Chat gleichzeitig angezeigt.
        """
        try:
            # Überprüfe, ob das Hauptfenster noch existiert
            if not self.root or not self.root.winfo_exists():
                return

            # Erstelle das chat_list_frame, falls es nicht existiert
            if not hasattr(
                    self,
                    'chat_list_frame') or not self.chat_list_frame.winfo_exists():
                if hasattr(self, 'sidebar') and self.sidebar.winfo_exists():
                    self.chat_list_frame = ctk.CTkScrollableFrame(
                        self.sidebar,
                        fg_color="transparent"
                    )
                    self.chat_list_frame.pack(
                        fill="both", expand=True, padx=5, pady=5)
                else:
                    logger.error(
                        "Sidebar nicht gefunden, kann Chat-Liste nicht erstellen")
                    return

            # Lösche vorhandene Einträge im Haupt-Thread
            def clear_frame():
                if hasattr(
                        self,
                        'chat_list_frame') and self.chat_list_frame.winfo_exists():
                    for widget in self.chat_list_frame.winfo_children():
                        widget.destroy()

            self.root.after(0, clear_frame)

            # Lade Konversationen in einem separaten Thread
            def load_conversations():
                try:
                    conversations = self.database.get_conversations(
                        user_id=self.current_user)
                    if not conversations:
                        logger.info("Keine Konversationen gefunden")
                        # Zeige eine Meldung an, wenn keine Konversationen
                        # vorhanden sind
                        self.root.after(
                            0,
                            lambda: ctk.CTkLabel(
                                self.chat_list_frame,
                                text="Keine Chats gefunden.\nErstellen Sie einen neuen Chat!",
                                text_color="#888888",
                                wraplength=180,
                                justify='center').pack(
                                pady=20,
                                padx=10))
                        return

                    # Sortiere die Konversationen nach Datum (neueste zuerst)
                    conversations.sort(
                        key=lambda x: x.get(
                            'updated_at', ''), reverse=True)

                    # Erstelle die UI-Elemente im Haupt-Thread
                    self.root.after(
                        0, lambda: self._create_conversation_widgets(conversations))

                    # Lade die letzte Konversation automatisch
                    if conversations and not self.current_conversation_id:
                        self.root.after(
                            100, lambda: self.load_chat(
                                conversations[0]['id']))

                except Exception as e:
                    logger.error(
                        f"Fehler beim Laden der Konversationen: {e}",
                        exc_info=True)
                    self.root.after(0, lambda: self.status_bar.configure(
                        text=f"Fehler beim Laden der Konversationen: {str(e)}"
                    ))

            # Starte das Laden der Konversationen in einem separaten Thread
            import threading
            threading.Thread(target=load_conversations, daemon=True).start()

        except Exception as e:
            logger.error(
                f"Kritischer Fehler in load_conversation_list: {e}",
                exc_info=True)
            if hasattr(self, 'root') and self.root.winfo_exists():
                self.root.after(0, lambda: self.status_bar.configure(
                    text=f"Kritischer Fehler: {str(e)}"
                ))

    def _create_conversation_widgets(self, conversations):
        """Erstellt die UI-Elemente für die Konversationsliste.

        Args:
            conversations: Liste der Konversationen
        """
        try:
            if not hasattr(
                    self,
                    'chat_list_frame') or not self.chat_list_frame.winfo_exists():
                return

            # Lösche vorhandene Einträge
            for widget in self.chat_list_frame.winfo_children():
                widget.destroy()

            # Füge einen "Neuer Chat"-Button hinzu
            new_chat_btn = ctk.CTkButton(
                self.chat_list_frame,
                text="+ Neuer Chat",
                command=self.new_chat,
                fg_color="#2b2b2b",
                hover_color="#3a3a3a",
                height=40,
                corner_radius=5
            )
            new_chat_btn.pack(fill='x', padx=5, pady=(0, 10))

            # Füge eine Trennlinie hinzu
            separator = ctk.CTkFrame(
                self.chat_list_frame, height=1, fg_color="#444444")
            separator.pack(fill='x', padx=5, pady=(0, 10))

            if not conversations:
                # Zeige eine Meldung an, wenn keine Konversationen vorhanden
                # sind
                no_chats_label = ctk.CTkLabel(
                    self.chat_list_frame,
                    text="Keine Chats gefunden.\nErstellen Sie einen neuen Chat!",
                    text_color="#888888",
                    wraplength=180,
                    justify='center')
                no_chats_label.pack(pady=20, padx=10)
                return

            # Erstelle für jede Konversation ein Widget
            for conv in conversations:
                self._create_conversation_widget(conv)

            # Füge einen Abstand am Ende hinzu
            ctk.CTkLabel(self.chat_list_frame, text="").pack(pady=5)

        except Exception as e:
            logger.error(
                f"Fehler beim Erstellen der Chat-Liste: {e}",
                exc_info=True)

    def _create_conversation_widget(self, conv):
        """Erstellt ein Widget für eine einzelne Konversation."""
        try:
            if not hasattr(
                    self,
                    'chat_list_frame') or not self.chat_list_frame.winfo_exists():
                return

            # Erstelle einen Frame für jeden Chat-Eintrag
            chat_frame = ctk.CTkFrame(
                self.chat_list_frame,
                fg_color="#2b2b2b" if conv.get(
                    'id') != self.current_conversation_id else "#3a3a3a",
                corner_radius=5,
                height=50  # Feste Höhe für jeden Eintrag
            )
            chat_frame.pack(fill="x", padx=5, pady=2)

            # Erstelle ein Label mit dem Chat-Titel
            title = str(conv.get('title', 'Unbenannter Chat')).strip()
            preview = str(conv.get('preview', '')).strip()

            # Kürze den Titel, wenn er zu lang ist
            display_title = (title[:22] + "...") if len(title) > 25 else title

            # Erstelle ein Label mit dem Titel und der Vorschau
            label_text = f"{display_title}\n<small>{preview[:30]}...</small>" if preview else display_title

            # Verwende ein Canvas für bessere Performance bei vielen Einträgen
            canvas = tk.Canvas(
                chat_frame,
                bg="#2b2b2b" if conv.get('id') != self.current_conversation_id else "#3a3a3a",
                height=40,
                highlightthickness=0,
                bd=0)
            canvas.pack(fill="x", expand=True, padx=5, pady=2)

            # Füge Text direkt auf das Canvas
            canvas.create_text(
                5, 5,
                anchor="nw",
                text=display_title,
                fill="#ffffff",
                font=("Arial", 10, "bold"),
                width=180
            )

            if preview:
                canvas.create_text(
                    5, 25,
                    anchor="nw",
                    text=preview[:30] + "..." if len(preview) > 30 else preview,
                    fill="#aaaaaa",
                    font=("Arial", 8),
                    width=180
                )

            # Speichere die Konversations-ID als Attribut des Frames
            chat_frame.conversation_id = conv.get('id')

            # Füge Klick-Event-Handler hinzu
            def on_click(e, cid=conv.get('id')):
                self.load_chat(cid)

            chat_frame.bind("<Button-1>", on_click)
            canvas.bind("<Button-1>", on_click)

            # Hover-Effekte
            def on_enter(e, frame=chat_frame, cid=conv.get('id')):
                if hasattr(
                        self,
                        'current_conversation_id') and cid != self.current_conversation_id:
                    frame.configure(fg_color="#3a3a3a")
                    canvas.configure(bg="#3a3a3a")

            def on_leave(e, frame=chat_frame, cid=conv.get('id')):
                if hasattr(
                        self,
                        'current_conversation_id') and cid != self.current_conversation_id:
                    frame.configure(fg_color="#2b2b2b")
                    canvas.configure(bg="#2b2b2b")

            chat_frame.bind("<Enter>", on_enter)
            chat_frame.bind("<Leave>", on_leave)
            canvas.bind("<Enter>", on_enter)
            canvas.bind("<Leave>", on_leave)

        except Exception as e:
            logger.error(
                f"Fehler beim Erstellen des Konversations-Widgets: {e}",
                exc_info=True)

    def show(self):
        """Zeigt den Dialog an."""
        try:
            self.root.lift()
            self.root.focus_force()
            self.root.deiconify()
        except Exception as e:
            logger.error(f"Fehler beim Anzeigen des Fensters: {e}")
            try:
                self.root.deiconify()  # Versuche es erneut mit einer einfachen Methode
            except BaseException:
                pass


def run_async_loop():
    """Startet den asynchronen Event-Loop für die Anwendung."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        # Beende alle laufenden Tasks
        tasks = asyncio.all_tasks(loop=loop)
        for task in tasks:
            task.cancel()

        # Warte auf das Beenden der Tasks
        loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        loop.close()


def main():
    """Hauptfunktion zum Starten der Anwendung."""
    try:
        # Erstelle das Hauptfenster
        root = ctk.CTk()

        # Setze das Farbschema basierend auf dem Systemthema
        ctk.set_appearance_mode("system")  # "light" oder "dark"
        # Verfügbare Themes: blue, dark-blue, green
        ctk.set_default_color_theme("blue")

        # Erstelle die Hauptanwendung
        app = EnhancedKIChatGUI(root)

        # Setze Fenstertitel und Größe
        root.title("Jarvis KI-Assistent")
        root.geometry("1200x800")

        # Starte den asynchronen Event-Loop in einem separaten Thread
        import threading
        threading.Thread(target=run_async_loop, daemon=True).start()

        # Starte die Hauptschleife
        root.mainloop()

    except Exception as e:
        logger.critical(
            f"Kritischer Fehler in der Anwendung: {e}",
            exc_info=True)
        messagebox.showerror(
            "Fehler",
            f"Ein schwerwiegender Fehler ist aufgetreten:\n{str(e)}\n\nBitte überprüfen Sie die Protokolldatei für weitere Details.")
    finally:
        # Aufräumarbeiten
        try:
            if 'app' in locals() and hasattr(app, 'model_manager'):
                app.model_manager.performance_monitor.stop_monitoring()
        except Exception as e:
            logger.error(f"Fehler beim Aufräumen: {e}")

    async def _process_user_input_async(self, user_text: str):
        """Verarbeitet die Benutzereingabe und zeigt die Antwort an (asynchron)."""
        try:
            # Aktualisiere die Benutzeroberfläche im Hauptthread
            self.after(
                0, lambda: self.status_bar.configure(
                    text="Denke nach..."))

            try:
                # Generiere die Antwort asynchron
                response = await self.generate_response_async(user_text)

                # Zeige die Antwort im Hauptthread an
                self.after(
                    0, lambda: self.display_message(
                        "assistant", response))
                self.after(0, lambda: self.status_bar.configure(text="Bereit"))

            except Exception as e:
                error_msg = f"Entschuldigung, ein Fehler ist bei der Generierung der Antwort aufgetreten: {str(e)}"
                logger.error(
                    f"Fehler bei der Antwortgenerierung: {e}",
                    exc_info=True)

                # Zeige die Fehlermeldung im Hauptthread an
                self.after(
                    0, lambda: self.display_message(
                        "assistant", error_msg))
                self.after(
                    0, lambda: self.status_bar.configure(
                        text=f"Fehler: {str(e)}"))

                # Protokolliere den Fehler ausführlich
                logger.error(
                    f"Fehlerdetails: {str(e)}\n{''.join(traceback.format_exception(type(e), e, e.__traceback__))}")

        except Exception as e:
            # Behandle unerwartete Fehler
            error_msg = "Ein unerwarteter Fehler ist aufgetreten. Bitte versuchen Sie es später erneut."
            logger.critical(
                f"Kritischer Fehler in _process_user_input_async: {e}",
                exc_info=True)

            # Zeige die Fehlermeldung im Hauptthread an
            self.after(0, lambda: self.display_message("assistant", error_msg))
            self.after(
                0, lambda: self.status_bar.configure(
                    text="Kritischer Fehler"))

        finally:
            # Aktiviere die Eingabe im Hauptthread wieder
            self.after(0, self._enable_input)

    def _enable_input(self):
        """Aktiviert die Benutzereingabe wieder."""
        self.user_input.config(state='normal')
        self.send_button.config(state='normal')
        self.user_input.focus_set()

    def send_message(self, event=None):
        """Verarbeitet die Benutzereingabe und sendet sie an das KI-Modell."""
        try:
            # Verhindere die Standardverarbeitung der Eingabetaste
            if event and event.keysym == 'Return' and not event.state & 0x1:  # Nur wenn nicht Shift+Enter
                return 'break'

            # Hole den Benutzertext und überprüfe, ob er leer ist
            user_text = self.user_input.get("1.0", "end-1c").strip()
            if not user_text:
                return 'break'  # Verhindere weitere Verarbeitung bei leerer Eingabe

            # Deaktiviere das Eingabefeld und den Senden-Button während der
            # Verarbeitung
            self.user_input.config(state='disabled')
            if hasattr(self, 'send_btn'):
                self.send_btn.config(state='disabled')

            # Aktualisiere die Benutzeroberfläche, um die Deaktivierung
            # anzuzeigen
            self.root.update_idletasks()

            # Starte einen neuen Thread für die Verarbeitung der Nachricht
            threading.Thread(
                target=self._process_user_message,
                args=(user_text,),
                daemon=True
            ).start()

            return 'break'  # Verhindere weitere Verarbeitung des Events

        except Exception as e:
            logger.error(
                f"Fehler beim Senden der Nachricht: {e}",
                exc_info=True)
            messagebox.showerror(
                "Fehler", f"Nachricht konnte nicht gesendet werden: {e}")
            # Aktiviere die Eingabe im Fehlerfall wieder
            self.user_input.config(state='normal')
            if hasattr(self, 'send_btn'):
                self.send_btn.config(state='normal')
            # Aktualisiere die Statusleiste
            self.after(
                0, lambda: self.status_bar.configure(
                    text=f"Fehler: {str(e)}"))
            return 'break'  # Verhindere weitere Verarbeitung des Events

        """Verarbeitet die Benutzereingabe und generiert eine Antwort.

        Args:
            user_text: Der Text, den der Benutzer eingegeben hat
        """
        try:
            # Erstelle eine ChatMessage für die Benutzereingabe
            user_message = ChatMessage(
                role="user",
                content=user_text,
                timestamp=datetime.now()
            )

            # Füge die Nachricht zur Konversationshistorie hinzu
            if not hasattr(self, 'conversation_history'):
                self.conversation_history = []
            self.conversation_history.append(user_message)

            # Generiere eine Antwort des Assistenten
            response = self.model_manager.generate_response(
                user_text,
                conversation_history=self.conversation_history
            )

            # Verarbeite die Antwort im Hauptthread
            self.root.after(0, self._handle_ai_response, response)

        except Exception as e:
            logger.error(
                f"Fehler bei der Verarbeitung der Benutzereingabe: {e}",
                exc_info=True)
            self.root.after(0, lambda: self.status_bar.configure(
                text=f"Fehler bei der Verarbeitung: {str(e)}"
            ))

    def _handle_ai_response(self, response: str):
        """Verarbeitet die Antwort des KI-Assistenten.

        Args:
            response: Die generierte Antwort des KI-Assistenten
        """
        try:
            # Zeige die Antwort an
            self._display_message("assistant", response)

            # Erstelle eine ChatMessage für die Antwort
            assistant_message = ChatMessage(
                role="assistant",
                content=response,
                timestamp=datetime.now(),
                model_used=self.model_manager.current_model,
                tokens_used=len(response.split())  # Einfache Token-Schätzung
            )

            # Füge die Antwort zur Konversationshistorie hinzu
            if not hasattr(self, 'conversation_history'):
                self.conversation_history = []
            self.conversation_history.append(assistant_message)

            # Speichere die Konversation
            self.save_chat()

            # Aktualisiere die Chat-Liste, um die Vorschau zu aktualisieren
            if hasattr(self, 'load_conversation_list'):
                self.load_conversation_list()

            # Aktualisiere den Status
            self.status_bar.configure(text="Bereit")

        except Exception as e:
            logger.error(
                f"Fehler bei der Verarbeitung der KI-Antwort: {e}",
                exc_info=True)
            self.status_bar.configure(
                text=f"Fehler bei der Verarbeitung der Antwort: {str(e)}")

        finally:
            # Aktiviere die Eingabe wieder
            self.user_input.configure(state="normal")
            self.send_button.configure(state="normal")
            self.user_input.focus_set()

    def _display_message(self, sender: str, message: str):
        """Zeigt eine Nachricht im Chat-Fenster an.

        Args:
            sender: Der Absender der Nachricht ('user', 'assistant' oder 'system')
            message: Der Nachrichtentext
        """
        try:
            if not hasattr(self, 'chat_display'):
                logger.warning("Chat-Display nicht initialisiert")
                return

            # Konfiguriere die Formatierung basierend auf dem Absender
            if sender.lower() == 'user':
                prefix = "Sie: "
                tag = "user"
                fg_color = "#e1f5fe"  # Hellblau für Benutzer
                bg_color = "#0d47a1"  # Dunkelblau für Benutzer
                justify = 'right'
            elif sender.lower() == 'assistant':
                prefix = "Jarvis: "
                tag = "assistant"
                fg_color = "#e8f5e9"  # Hellgrün für Assistent
                bg_color = "#2e7d32"  # Dunkelgrün für Assistent
                justify = 'left'
            else:  # system
                prefix = "System: "
                tag = "system"
                fg_color = "#f3e5f5"  # Helllila für System
                bg_color = "#6a1b9a"  # Dunkellila für System
                justify = 'center'

            # Aktiviere das Textfeld für die Bearbeitung
            self.chat_display.configure(state='normal')

            # Füge die Nachricht mit Formatierung hinzu
            self.chat_display.insert('end', f"{prefix}\n", tag)
            self.chat_display.insert('end', f"{message}\n\n", f"{tag}_text")

            # Konfiguriere die Tags für die Formatierung
            self.chat_display.tag_configure(
                tag,
                font=(
                    'Arial',
                    10,
                    'bold'),
                foreground=fg_color,
                justify=justify)
            self.chat_display.tag_configure(
                f"{tag}_text",
                font=(
                    'Arial',
                    10),
                foreground='#ffffff',
                justify=justify,
                lmargin1=20,
                lmargin2=20,
                rmargin=20)

            # Deaktiviere das Textfeld nach der Bearbeitung
            self.chat_display.configure(state='disabled')

            # Scrolle zum Ende der Nachricht
            self.chat_display.see('end')

        except Exception as e:
            logger.error(
                f"Fehler beim Anzeigen der Nachricht: {e}",
                exc_info=True)

    def save_chat(self):
        """Speichert den aktuellen Chat in der Datenbank."""
        try:
            if not hasattr(
                    self,
                    'conversation_history') or not self.conversation_history:
                logger.info("Keine Nachrichten zum Speichern vorhanden")
                return

            if not hasattr(self, 'database') or not self.database:
                logger.error("Keine Datenbankverbindung verfügbar")
                return

            # Erstelle eine Vorschau aus der letzten Nachricht
            last_message = self.conversation_history[-1].content
            # Erste 50 Zeichen der letzten Nachricht
            preview = last_message[:50]
            if len(last_message) > 50:
                preview += "..."

            if hasattr(
                    self,
                    'current_conversation_id') and self.current_conversation_id:
                # Aktualisiere bestehende Konversation
                success = self.database.update_conversation(
                    conversation_id=self.current_conversation_id,
                    messages=self.conversation_history,
                    preview=preview
                )

                if success:
                    logger.info(
                        f"Konversation {self.current_conversation_id} aktualisiert")
                    self.status_bar.configure(text="Konversation gespeichert")
                else:
                    logger.error("Fehler beim Aktualisieren der Konversation")
                    self.status_bar.configure(
                        text="Fehler beim Speichern der Konversation")
            else:
                # Erstelle eine neue Konversation
                conv_id = self.database.save_conversation(
                    user_id=self.current_user,
                    messages=self.conversation_history,
                    preview=preview
                )

                if conv_id:
                    self.current_conversation_id = conv_id
                    logger.info(f"Neue Konversation mit ID {conv_id} erstellt")
                    self.status_bar.configure(
                        text="Neue Konversation erstellt")

                    # Aktualisiere die Chat-Liste, um die neue Konversation
                    # anzuzeigen
                    self.load_conversation_list()
                else:
                    logger.error("Fehler beim Erstellen der Konversation")
                    self.status_bar.configure(
                        text="Fehler beim Erstellen der Konversation")

        except Exception as e:
            logger.error(
                f"Fehler beim Speichern des Chats: {e}",
                exc_info=True)
            self.status_bar.configure(text=f"Fehler beim Speichern: {str(e)}")

    def load_chat(self, event=None):
        """Lädt einen ausgewählten Chat aus der Konversationsliste."""
        try:
            # Überprüfe, ob eine Konversation ausgewählt ist
            selection = self.conversation_list.curselection()
            if not selection:
                return

            # Hole die Konversations-ID der ausgewählten Konversation
            conversation_id = self.conversation_list.get(selection[0])[0]

            # Speichere den aktuellen Chat, falls vorhanden
            if hasattr(
                    self,
                    'conversation_history') and self.conversation_history:
                self.save_chat()

            # Setze die Konversationshistorie zurück
            self.conversation_history = []
            self.current_conversation_id = conversation_id

            # Lösche die Chat-Anzeige
            self.chat_display.configure(state='normal')
            self.chat_display.delete('1.0', 'end')

            # Lade die Nachrichten für diese Konversation aus der Datenbank
            if hasattr(self, 'database') and self.database:
                cursor = self.database.execute(
                    """
                    SELECT role, content, timestamp
                    FROM messages
                    WHERE conversation_id = ?
                    ORDER BY timestamp ASC
                    """,
                    (conversation_id,)
                )

                messages = cursor.fetchall()

                # Zeige die Nachrichten an
                for msg in messages:
                    role = msg['role']
                    content = msg['content']
                    self._display_message(role, content)

                    # Füge zur Historie hinzu
                    self.conversation_history.append({
                        'role': role,
                        'content': content,
                        'timestamp': msg['timestamp']
                    })

            # Deaktiviere das Textfeld wieder
            self.chat_display.configure(state='disabled')

            # Aktualisiere die Chat-Liste, um die Auswahl hervorzuheben
            self.load_conversation_list()

            # Setze den Fokus auf das Eingabefeld
            self.user_input.focus_set()

            # Status aktualisieren
            self.status_bar.configure(
                text=f"Konversation geladen ({len(messages)} Nachrichten)")
            logger.info(
                f"Konversation {conversation_id} mit {len(messages)} Nachrichten geladen")

        except Exception as e:
            logger.error(
                f"Fehler beim Laden der Konversation: {e}",
                exc_info=True)
            self.status_bar.configure(
                text=f"Fehler beim Laden der Konversation: {str(e)}")

    def new_chat(self, event=None):
        """Startet einen neuen Chat."""
        try:
            # Speichere den aktuellen Chat, falls vorhanden
            if hasattr(
                    self,
                    'conversation_history') and self.conversation_history:
                self.save_chat()

            # Setze die Konversationshistorie zurück
            self.conversation_history = []
            self.current_conversation_id = None

            # Lösche die Chat-Anzeige
            if hasattr(self, 'chat_display'):
                self.chat_display.configure(state='normal')
                self.chat_display.delete('1.0', 'end')
                self.chat_display.configure(state='disabled')

            # Aktualisiere die Chat-Liste, um die Auswahl zu entfernen
            if hasattr(self, 'load_conversation_list'):
                self.load_conversation_list()

            # Setze den Fokus auf das Eingabefeld
            if hasattr(self, 'user_input'):
                self.user_input.focus_set()

            # Status aktualisieren
            self.status_bar.configure(text="Neuer Chat gestartet")
            logger.info("Neuer Chat wurde gestartet")

            # Zeige eine Willkommensnachricht an
            welcome_message = (
                "Willkommen zu Ihrem neuen Chat! Ich bin Jarvis, Ihr persönlicher KI-Assistent. "
                "Wie kann ich Ihnen heute helfen?")
            self._display_message("assistant", welcome_message)

        except Exception as e:
            logger.error(
                f"Fehler beim Starten eines neuen Chats: {e}",
                exc_info=True)
            self.status_bar.configure(
                text=f"Fehler beim Starten eines neuen Chats: {str(e)}")

    def setup_gui(self):
        """Initialisiert die GUI-Komponenten."""
        try:
            # Hauptcontainer für die Anwendung
            self.main_container = ttk.Frame(self.root, style='TFrame')
            self.main_container.pack(fill=tk.BOTH, expand=True)

            # Richte die Seitenleiste ein
            self._setup_sidebar()

            # Hauptbereich für den Chat
            self.chat_frame = ttk.Frame(self.main_container, style='TFrame')
            self.chat_frame.pack(
                side=tk.RIGHT,
                fill=tk.BOTH,
                expand=True,
                padx=5,
                pady=5)

            # Chat-Anzeige inkl. Scrollbar und Kontextmenü konfigurieren
            self._configure_chat_display()

            # Eingabebereich
            input_frame = ttk.Frame(self.chat_frame, style='TFrame')
            input_frame.pack(fill=tk.X, padx=5, pady=5)

            self.user_input = tk.Text(
                input_frame,
                height=4,
                wrap=tk.WORD,
                bg='white',
                fg='black',
                font=('Arial', 11),
                padx=10,
                pady=10,
                relief=tk.SOLID,
                borderwidth=1
            )
            self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.user_input.bind('<Return>', self._on_send_message)
            # Zeilenumbruch mit Shift+Enter
            self.user_input.bind('<Shift-Return>', lambda e: 'break')

            # Senden-Button
            self.send_button = ttk.Button(
                input_frame,
                text="Senden",
                command=self._on_send_message,
                style='TButton'
            )
            self.send_button.pack(side=tk.RIGHT, padx=(5, 0))

            # Statusleiste
            self.status_bar = ttk.Label(
                self.root,
                text="Bereit",
                relief=tk.SUNKEN,
                anchor=tk.W
            )
            self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

            # Fokus auf das Eingabefeld setzen
            self.user_input.focus_set()

            # Fenstergröße und Position
            self.root.geometry("1200x800")
            self.root.minsize(800, 600)

            # Fenster zentrieren
            self._center_window()

        except Exception as e:
            logger.error(
                f"Fehler beim Initialisieren der GUI: {e}",
                exc_info=True)
            raise

    def _center_window(self):
        """Zentriert das Hauptfenster auf dem Bildschirm."""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def copy_text(self):
        """Kopiert den ausgewählten Text in die Zwischenablage."""
        try:
            selected_text = self.chat_display.get(tk.SEL_FIRST, tk.SEL_LAST)
            self.root.clipboard_clear()
            self.root.clipboard_append(selected_text)
        except tk.TclError:
            # Kein Text ausgewählt
            pass

    def select_all_text(self):
        """Selektiert den gesamten Text im Chat-Fenster."""
        self.chat_display.tag_add(tk.SEL, '1.0', tk.END)
        self.chat_display.mark_set(tk.INSERT, '1.0')
        self.chat_display.see(tk.INSERT)
        return 'break'

    def show_context_menu(self, event):
        """Zeigt das Kontextmenü an der angeklickten Position an."""
        try:
            self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.context_menu.grab_release()

    def _setup_sidebar(self):
        """Initialisiert die Seitenleiste mit der Konversationsliste."""
        try:
            # Linke Seitenleiste
            self.sidebar = ttk.Frame(
                self.main_container,
                width=250,
                style='Secondary.TFrame')
            self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
            self.sidebar.pack_propagate(False)

            # Button für neuen Chat
            self.new_chat_btn = ttk.Button(
                self.sidebar,
                text="Neuer Chat",
                command=self.new_chat,
                style='TButton'
            )
            self.new_chat_btn.pack(fill=tk.X, pady=(0, 10), padx=5)

            # Scrollbar für die Konversationsliste
            self.conversation_scroll = ttk.Scrollbar(self.sidebar)
            self.conversation_scroll.pack(side=tk.RIGHT, fill=tk.Y)

            # Frame für die Konversationsliste
            self.conversation_frame = ttk.Frame(self.sidebar)
            self.conversation_frame.pack(
                fill=tk.BOTH, expand=True, padx=5, pady=5)

            # Liste der Konversationen
            self.conversation_list = tk.Listbox(
                self.conversation_frame,
                yscrollcommand=self.conversation_scroll.set,
                selectmode=tk.SINGLE,
                bg='#ffffff',
                fg='#000000',
                font=('Arial', 10),
                borderwidth=0,
                highlightthickness=0,
                activestyle='none'
            )
            self.conversation_list.pack(
                side=tk.LEFT, fill=tk.BOTH, expand=True)

            # Konfiguriere die Scrollbar
            self.conversation_scroll.config(
                command=self.conversation_list.yview)

            # Doppelklick auf eine Konversation lädt sie
            self.conversation_list.bind(
                '<Double-1>', self.load_selected_conversation)

            # Kontextmenü für Konversationen
            self.conversation_menu = tk.Menu(self.conversation_list, tearoff=0)
            self.conversation_menu.add_command(
                label="Löschen", command=self.delete_selected_conversation)
            self.conversation_list.bind(
                '<Button-3>', self.show_conversation_menu)

            # Statusleiste in der Seitenleiste
            self.sidebar_status = ttk.Label(
                self.sidebar,
                text="Keine Konversation ausgewählt",
                relief=tk.SUNKEN,
                anchor=tk.W,
                padding=3
            )
            self.sidebar_status.pack(side=tk.BOTTOM, fill=tk.X)

            # Lade die Konversationsliste
            self.load_conversation_list()

        except Exception as e:
            logger.error(
                f"Fehler beim Initialisieren der Seitenleiste: {e}",
                exc_info=True)
            raise

    def show_conversation_menu(self, event):
        """Zeigt das Kontextmenü für Konversationen an."""
        try:
            # Wähle die angeklickte Konversation aus
            self.conversation_list.selection_clear(0, tk.END)
            self.conversation_list.selection_set(
                self.conversation_list.nearest(event.y))
            self.conversation_list.activate(
                self.conversation_list.nearest(event.y))

            # Zeige das Kontextmenü an
            self.conversation_menu.post(event.x_root, event.y_root)
        except Exception as e:
            logger.error(
                f"Fehler beim Anzeigen des Kontextmenüs: {e}",
                exc_info=True)

    def load_selected_conversation(self, event=None):
        """Lädt die ausgewählte Konversation in die Benutzeroberfläche.

        Args:
            event: Das auslösende Event (optional)
        """
        try:
            # Hole die Auswahl
            selection = self.conversation_list.curselection()
            if not selection:
                return

            # Lade die Konversation
            conversation_id = self.conversation_list.get(
                selection[0])[0]  # Annahme: ID ist das erste Element
            self.load_conversation(conversation_id)

            # Aktualisiere die Statusleiste
            if hasattr(
                    self,
                    'sidebar_status') and self.sidebar_status.winfo_exists():
                self.sidebar_status.config(
                    text=f"Geladen: {self.conversation_list.get(selection[0])[1]}")

            # Fokus auf das Eingabefeld setzen
            if hasattr(self, 'user_input') and self.user_input.winfo_exists():
                self.user_input.focus_set()

        except Exception as e:
            logger.error(
                f"Fehler beim Laden der Konversation: {e}",
                exc_info=True)
            if hasattr(self, 'status_bar') and self.status_bar.winfo_exists():
                self.status_bar.config(
                    text=f"Fehler beim Laden der Konversation: {str(e)}")
            else:
                messagebox.showerror(
                    "Fehler", f"Konnte die Konversation nicht laden: {str(e)}")

    def delete_selected_conversation(self):
        """Löscht die ausgewählte Konversation."""
        try:
            # Hole die ausgewählte Konversation
            selection = self.conversation_list.curselection()
            if not selection:
                return

            # Bestätigungsdialog
            if not messagebox.askyesno(
                "Löschen bestätigen",
                    "Möchten Sie diese Konversation wirklich löschen?"):
                return

            # Lösche die Konversation aus der Datenbank
            conversation_id = self.conversation_list.get(
                selection[0])[0]  # Annahme: ID ist das erste Element
            self.database.delete_conversation(conversation_id)

            # Aktualisiere die Konversationsliste
            self.load_conversation_list()

            # Leere den Chat-Bereich, wenn die gelöschte Konversation angezeigt
            # wurde
            if self.current_conversation_id == conversation_id:
                self.current_conversation_id = None
                self.chat_display.config(state='normal')
                self.chat_display.delete(1.0, tk.END)
                self.chat_display.config(state='disabled')
                self.sidebar_status.config(
                    text="Keine Konversation ausgewählt")

            self.status_bar.config(text="Konversation wurde gelöscht")
        except Exception as e:
            logger.error(f"Fehler beim Löschen der Konversation: {str(e)}")
            messagebox.showerror("Fehler", f"Konnte die Konversation nicht löschen: {str(e)}")

    def _save_message_to_db(self, role: str, content: str, timestamp: str):
        """Speichert eine Nachricht in der Datenbank.

        Args:
            role (str): Die Rolle des Absenders ('user' oder 'assistant')
            content (str): Der Inhalt der Nachricht
            timestamp (str): Zeitstempel der Nachricht
        """
        # Aktiviere das Textfeld zum Bearbeiten
        self.chat_display.configure(state='normal')

        # Lösche die letzte Nachricht des Assistenten, falls vorhanden
        if role == 'assistant':
            self.chat_display.delete('end-2l', 'end')

        # Füge die Nachricht hinzu
        self._display_message(role, content)

        # Speichere die Nachricht in der Datenbank
        if self.current_conversation_id:
            self.database.save_message(
                self.current_conversation_id,
                role,
                content,
                timestamp
            )

        # Deaktiviere das Textfeld wieder
        self.chat_display.configure(state='disabled')

    def _generate_ai_response(self, user_text: str):
        """Generiert eine Antwort des KI-Assistenten auf die Benutzereingabe."""
        try:
            # Bestimme das beste Modell für die Anfrage
            model = self.model_manager.get_best_model(
                user_text,
                self.conversation_history
            )

            # Erstelle einen System-Prompt für das Modell
            system_prompt = self.model_manager._create_system_prompt(
                model.model_type)

            # Bereite die Nachrichten für das Modell vor
            messages = [{"role": "system", "content": system_prompt}]

            # Füge den Konversationsverlauf hinzu
            for msg in self.conversation_history:
                messages.append({
                    "role": msg['role'],
                    "content": msg['content']
                })

            # Generiere die Antwort
            response = model.create_chat_completion(
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                stream=True
            )

            # Sammle die Antwort in Teilen
            full_response = ""
            for chunk in response:
                if 'content' in chunk.choices[0].delta:
                    chunk_text = chunk.choices[0].delta['content']
                    full_response += chunk_text

                    # Aktualisiere die Anzeige mit dem aktuellen Chunk
                    self._update_response_display(full_response)

            # Speichere die Antwort in der Datenbank
            timestamp = datetime.now().isoformat()
            if hasattr(
                    self,
                    'current_conversation_id') and self.current_conversation_id:
                self._save_message_to_db('assistant', full_response, timestamp)

            # Füge die Antwort zur Konversationshistorie hinzu
            self.conversation_history.append({
                'role': 'assistant',
                'content': full_response,
                'timestamp': timestamp
            })

        except Exception as e:
            logger.error(
                f"Fehler bei der Antwortgenerierung: {e}",
                exc_info=True)
            self._display_message(
                'assistant',
                f"Entschuldigung, ich konnte keine Antwort generieren: {e}")

    def _process_user_message(self, user_text: str):
        """Verarbeitet die Benutzernachricht im Hintergrund.

        Args:
            user_text: Der eingegebene Text des Benutzers
        """
        try:
            # Zeige die Benutzernachricht an
            self.root.after(
                0, lambda: self._display_message(
                    'user', user_text))

            # Leere das Eingabefeld
            self.root.after(0, lambda: self.user_input.delete('1.0', 'end'))

            # Füge die Nachricht zur Konversationshistorie hinzu
            timestamp = datetime.now().isoformat()
            self.conversation_history.append({
                'role': 'user',
                'content': user_text,
                'timestamp': timestamp
            })

            # Speichere die Nachricht in der Datenbank, wenn eine Konversation
            # existiert
            if hasattr(
                    self,
                    'current_conversation_id') and self.current_conversation_id:
                self._save_message_to_db('user', user_text, timestamp)

            # Generiere eine Antwort des KI-Assistenten
            self._generate_ai_response(user_text)

        except Exception as e:
            logger.error(
                f"Fehler bei der Nachrichtenverarbeitung: {e}",
                exc_info=True)
            self.root.after(
                0,
                lambda: self._display_message(
                    'assistant',
                    f"Entschuldigung, ein Fehler ist aufgetreten: {e}"))

        finally:
            # Aktiviere die Benutzereingabe wieder
            self.root.after(0, lambda: self.user_input.config(state='normal'))
            if hasattr(self, 'send_btn'):
                self.root.after(
                    0, lambda: self.send_btn.config(
                        state='normal'))

            # Setze den Fokus zurück auf das Eingabefeld
            self.root.after(100, lambda: self.user_input.focus_set())


def main():
    """Hauptfunktion zum Starten der Anwendung."""
    try:
        # Initialisiere das Hauptfenster
        root = tk.Tk()
        root.title("J.A.R.V.I.S. KI Assistent")

        # Erstelle die Anwendung
        app = EnhancedKIChatGUI(root)

        # Starte die Hauptschleife
        root.mainloop()

    except Exception as e:
        logger.critical(
            f"Kritischer Fehler in der Anwendung: {e}",
            exc_info=True)
        messagebox.showerror(
            "Fehler", f"Ein kritischer Fehler ist aufgetreten: {e}")


if __name__ == "__main__":
    main()
