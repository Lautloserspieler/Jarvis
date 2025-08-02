import logging
import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Callable, Dict, Any, List

import speech_recognition as sr
import pyttsx3

logger = logging.getLogger(__name__)

class SpeechRecognitionState(Enum):
    """Zustände der Spracherkennung."""
    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()
    ERROR = auto()

class TTSState(Enum):
    """Zustände der Sprachausgabe."""
    IDLE = auto()
    SPEAKING = auto()
    PAUSED = auto()
    ERROR = auto()

@dataclass
class VoiceCommand:
    """Repräsentiert einen erkannten Sprachbefehl."""
    text: str
    confidence: float
    timestamp: float
    metadata: Dict[str, Any] = None

class SpeechManager:
    """Verwaltet die Sprachsteuerung für den KI-Assistenten."""
    
    def __init__(self, config):
        """Initialisiert den SpeechManager.
        
        Args:
            config: Eine ConfigManager-Instanz
        """
        self.config = config
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300  # Anpassbar über Konfiguration
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        self.recognizer.phrase_threshold = 0.3
        
        # Warteschlange für Sprachbefehle
        self.command_queue = queue.Queue()
        
        # Statusvariablen
        self.recognition_state = SpeechRecognitionState.IDLE
        self.tts_state = TTSState.IDLE
        self.is_listening = False
        self.stop_event = threading.Event()
        
        # Initialisiere die Sprachausgabe
        self.engine = self._init_tts_engine()
        
        # Callbacks
        self.on_command_callbacks = []
        self.on_wake_word_callbacks = []
        self.on_speech_start_callbacks = []
        self.on_speech_end_callbacks = []
        
        # Starte den Hintergrund-Thread für die Spracherkennung
        self.recognition_thread = threading.Thread(
            target=self._recognition_worker,
            daemon=True
        )
        self.recognition_thread.start()
        
        logger.info("SpeechManager initialisiert")
    
    def _init_tts_engine(self):
        """Initialisiert die Text-to-Speech-Engine."""
        try:
            engine = pyttsx3.init()
            
            # Konfiguriere die Stimme
            voices = engine.getProperty('voices')
            if voices:
                # Versuche, eine deutsche Stimme zu finden
                german_voices = [v for v in voices if 'german' in v.languages[0].lower()]
                if german_voices:
                    engine.setProperty('voice', german_voices[0].id)
                else:
                    engine.setProperty('voice', voices[0].id)
            
            # Setze die Sprechgeschwindigkeit (Standard: 200 WPM)
            engine.setProperty('rate', 150)
            
            # Setze die Lautstärke (0.0 bis 1.0)
            engine.setProperty('volume', 0.9)
            
            # Event-Handler für die Sprachausgabe
            def on_start(name):
                self.tts_state = TTSState.SPEAKING
                self._trigger_callbacks(self.on_speech_start_callbacks)
            
            def on_end(name, completed):
                self.tts_state = TTSState.IDLE
                self._trigger_callbacks(self.on_speech_end_callbacks, completed=completed)
            
            # Registriere die Event-Handler
            engine.connect('started-utterance', on_start)
            engine.connect('finished-utterance', on_end)
            
            logger.info("TTS-Engine initialisiert")
            return engine
            
        except Exception as e:
            logger.error(f"Fehler bei der Initialisierung der TTS-Engine: {e}")
            self.tts_state = TTSState.ERROR
            return None
    
    def _trigger_callbacks(self, callbacks, *args, **kwargs):
        """Ruft alle registrierten Callbacks auf."""
        for callback in callbacks:
            try:
                if callable(callback):
                    callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Fehler in Callback: {e}")
    
    def start_listening(self):
        """Startet die kontinuierliche Spracherkennung."""
        if self.is_listening:
            return
            
        self.is_listening = True
        self.stop_event.clear()
        
        # Starte den Hintergrund-Thread für die Spracherkennung
        if not self.recognition_thread.is_alive():
            self.recognition_thread = threading.Thread(
                target=self._recognition_worker,
                daemon=True
            )
            self.recognition_thread.start()
        
        logger.info("Spracherkennung gestartet")
    
    def stop_listening(self):
        """Stoppt die kontinuierliche Spracherkennung."""
        self.is_listening = False
        self.stop_event.set()
        logger.info("Spracherkennung gestoppt")
    
    def _recognition_worker(self):
        """Hintergrund-Thread für die kontinuierliche Spracherkennung."""
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            while not self.stop_event.is_set() and self.is_listening:
                try:
                    self.recognition_state = SpeechRecognitionState.LISTENING
                    logger.debug("Höre zu...")
                    
                    # Höre auf Audio-Eingabe mit Timeout für reaktive Unterbrechung
                    audio = self.recognizer.listen(
                        source, 
                        timeout=None, 
                        phrase_time_limit=5
                    )
                    
                    self.recognition_state = SpeechRecognitionState.PROCESSING
                    logger.debug("Verarbeite Audio...")
                    
                    # Verwende Google Web Speech API für die Spracherkennung
                    try:
                        text = self.recognizer.recognize_google(
                            audio, 
                            language="de-DE"
                        )
                        
                        if text:
                            confidence = 0.9  # Google gibt keine Konfidenz zurück, daher schätzen wir
                            command = VoiceCommand(
                                text=text,
                                confidence=confidence,
                                timestamp=time.time()
                            )
                            
                            # Füge den Befehl zur Warteschlange hinzu
                            self.command_queue.put(command)
                            
                            # Trigger Callbacks
                            self._trigger_callbacks(self.on_command_callbacks, command)
                            
                            logger.info(f"Erkannter Befehl: {text} (Confidence: {confidence:.2f})")
                    
                    except sr.UnknownValueError:
                        logger.debug("Sprache konnte nicht erkannt werden")
                    except sr.RequestError as e:
                        logger.error(f"Fehler bei der Spracherkennung: {e}")
                        self.recognition_state = SpeechRecognitionState.ERROR
                    
                except Exception as e:
                    logger.error(f"Fehler im Spracherkennungs-Thread: {e}")
                    self.recognition_state = SpeechRecognitionState.ERROR
                    time.sleep(1)  # Kurze Pause bei Fehlern
                
                finally:
                    self.recognition_state = SpeechRecognitionState.IDLE
    
    def speak(self, text: str, block: bool = False):
        """Spricht den gegebenen Text aus.
        
        Args:
            text: Der zu sprechende Text
            block: Wenn True, blockiert die Methode, bis die Ausgabe abgeschlossen ist
        """
        if not self.engine:
            logger.error("TTS-Engine nicht initialisiert")
            return
        
        try:
            # Starte die Sprachausgabe in einem separaten Thread
            self.engine.say(text)
            
            if block:
                self.engine.runAndWait()
            else:
                # Starte die Sprachausgabe in einem separaten Thread
                threading.Thread(
                    target=self.engine.runAndWait,
                    daemon=True
                ).start()
                
        except Exception as e:
            logger.error(f"Fehler bei der Sprachausgabe: {e}")
            self.tts_state = TTSState.ERROR
    
    def stop_speaking(self):
        """Bricht die aktuelle Sprachausgabe ab."""
        if self.engine:
            try:
                self.engine.stop()
                self.tts_state = TTSState.IDLE
            except Exception as e:
                logger.error(f"Fehler beim Stoppen der Sprachausgabe: {e}")
    
    def get_next_command(self, timeout: float = None) -> Optional[VoiceCommand]:
        """Holt den nächsten erkannten Sprachbefehl aus der Warteschlange.
        
        Args:
            timeout: Maximale Wartezeit in Sekunden
            
        Returns:
            Der nächste VoiceCommand oder None bei Timeout
        """
        try:
            return self.command_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def register_command_callback(self, callback: Callable[[VoiceCommand], None]):
        """Registriert einen Callback für erkannte Sprachbefehle.
        
        Args:
            callback: Die aufzurufende Funktion
        """
        if callable(callback) and callback not in self.on_command_callbacks:
            self.on_command_callbacks.append(callback)
    
    def unregister_command_callback(self, callback: Callable[[VoiceCommand], None]):
        """Entfernt einen registrierten Callback.
        
        Args:
            callback: Die zu entfernende Funktion
        """
        if callback in self.on_command_callbacks:
            self.on_command_callbacks.remove(callback)
    
    def register_wake_word_callback(self, callback: Callable[[], None]):
        """Registriert einen Callback für das Erkennen des Wake-Words.
        
        Args:
            callback: Die aufzurufende Funktion
        """
        if callable(callback) and callback not in self.on_wake_word_callbacks:
            self.on_wake_word_callbacks.append(callback)
    
    def unregister_wake_word_callback(self, callback: Callable[[], None]):
        """Entfernt einen registrierten Wake-Word-Callback.
        
        Args:
            callback: Die zu entfernende Funktion
        """
        if callback in self.on_wake_word_callbacks:
            self.on_wake_word_callbacks.remove(callback)
    
    def set_voice_properties(self, rate: int = None, volume: float = None, voice_id: str = None):
        """Legt die Eigenschaften der Sprachausgabe fest.
        
        Args:
            rate: Sprechgeschwindigkeit in Wörtern pro Minute
            volume: Lautstärke (0.0 bis 1.0)
            voice_id: ID der zu verwendenden Stimme
        """
        if not self.engine:
            return
        
        try:
            if rate is not None:
                self.engine.setProperty('rate', rate)
            
            if volume is not None:
                self.engine.setProperty('volume', max(0.0, min(1.0, volume)))
            
            if voice_id is not None:
                voices = self.engine.getProperty('voices')
                if any(v.id == voice_id for v in voices):
                    self.engine.setProperty('voice', voice_id)
                else:
                    logger.warning(f"Stimme mit ID {voice_id} nicht gefunden")
        
        except Exception as e:
            logger.error(f"Fehler beim Setzen der Stimmeigenschaften: {e}")
    
    def get_available_voices(self) -> List[Dict[str, Any]]:
        """Gibt eine Liste der verfügbaren Stimmen zurück.
        
        Returns:
            Eine Liste von Dictionaries mit Stimminformationen
        """
        if not self.engine:
            return []
        
        voices = []
        for voice in self.engine.getProperty('voices'):
            voices.append({
                'id': voice.id,
                'name': voice.name,
                'languages': voice.languages,
                'gender': 'male' if 'male' in voice.name.lower() else 'female' if 'female' in voice.name.lower() else 'unknown'
            })
        
        return voices
    
    def is_speaking(self) -> bool:
        """Überprüft, ob gerade gesprochen wird.
        
        Returns:
            True, wenn die Sprachausgabe aktiv ist, sonst False
        """
        return self.tts_state == TTSState.SPEAKING
    
    def cleanup(self):
        """Bereinigt Ressourcen und stoppt alle laufenden Prozesse."""
        self.stop_listening()
        self.stop_speaking()
        
        # Warte auf das Ende des Erkennungs-Threads
        if self.recognition_thread.is_alive():
            self.recognition_thread.join(timeout=2.0)
        
        # Stoppe die TTS-Engine
        if self.engine:
            try:
                self.engine.stop()
                # pyttsx3 hat keine explizite cleanup-Methode
                del self.engine
                self.engine = None
            except Exception as e:
                logger.error(f"Fehler beim Aufräumen der TTS-Engine: {e}")
        
        logger.info("SpeechManager wurde aufgeräumt")
