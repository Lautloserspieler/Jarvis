import os
import logging
import json
import threading
import queue
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time

import numpy as np
from llama_cpp import Llama

# Lokale Module
from config_manager import ConfigManager
from context_scorer import ContextScorer
from performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)

class ModelType(str, Enum):
    """Enum für verschiedene Modelltypen."""
    CODE = "code"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    GENERAL = "general"
    EMOTIONAL = "emotional"

@dataclass
class ModelConfig:
    """Konfiguration für ein Sprachmodell."""
    name: str
    path: str
    model_type: ModelType = ModelType.GENERAL
    chat_format: str = "llama-2"
    n_ctx: int = 2048
    n_threads: int = 4
    n_gpu_layers: int = 0
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    stop: List[str] = field(default_factory=lambda: ["\n###"])
    echo: bool = False

class EnhancedModelManager:
    """Verwaltet die KI-Modelle und deren Lebenszyklus."""
    
    def __init__(self, config: ConfigManager):
        """Initialisiert den ModelManager mit optimierter Speichernutzung für M2.
        
        Args:
            config: Eine ConfigManager-Instanz
        """
        self.config = config
        self.models: Dict[str, Llama] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self.loading_queue = queue.Queue(maxsize=2)  # Begrenze die Warteschlange
        self.load_lock = threading.Lock()
        self.performance_monitor = PerformanceMonitor()
        self.smart_selector = SmartModelSelector()
        self.token_calculator = TokenCalculator()
        
        # Speicheroptimierungen
        self._cleanup_temp_files()
        
        # Modellverzeichnis erstellen, falls nicht vorhanden
        self.models_dir = Path(self.config.get('PATHS', 'models', fallback='models'))
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Automatische Modellerkennung mit Speicheroptimierung
        self._discover_models()
        
        # Hintergrund-Thread für das Laden von Modellen mit reduzierter Priorität starten
        self._start_background_loader()
    
    def _cleanup_temp_files(self):
        """Bereinigt temporäre Dateien und gibt Speicher frei."""
        import gc
        import tempfile
        
        try:
            # Manuelle Garbage Collection erzwingen
            gc.collect()
            
            # Temporäre Dateien bereinigen
            temp_dir = tempfile.gettempdir()
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith(('.tmp', '.swp', '.swx')):
                        try:
                            file_path = os.path.join(root, file)
                            # Lösche nur ältere temporäre Dateien (älter als 1 Stunde)
                            if time.time() - os.path.getmtime(file_path) > 3600:
                                os.remove(file_path)
                        except Exception as e:
                            logger.debug(f"Konnte temporäre Datei nicht löschen: {e}")
                            
            # Numpy-Cache leeren, falls numpy importiert wurde
            if 'numpy' in sys.modules:
                np.warnings.filterwarnings('ignore')
                np.set_printoptions(threshold=10)
                
        except Exception as e:
            logger.warning(f"Fehler bei der Bereinigung temporärer Dateien: {e}")
    
    def _start_background_loader(self) -> None:
        """Startet den Hintergrund-Thread zum Laden von Modellen."""
        self.loader_thread = threading.Thread(
            target=self._model_loading_worker,
            daemon=True,
            name="ModelLoaderThread"
        )
        # Setze niedrigere Priorität für den Lade-Thread
        self.loader_thread.start()
        
        # Setze die Thread-Priorität (nur unter Windows)
        if sys.platform == 'win32':
            try:
                import ctypes
                from ctypes import wintypes
                
                THREAD_PRIORITY_BELOW_NORMAL = 0xffffffff
                handle = wintypes.HANDLE(self.loader_thread.ident)
                ctypes.windll.kernel32.SetThreadPriority(handle, THREAD_PRIORITY_BELOW_NORMAL)
            except Exception as e:
                logger.debug(f"Konnte Thread-Priorität nicht setzen: {e}")
    

    
    def get_best_model(self, prompt: str, conversation_history: List[Dict[str, Any]] = None) -> Any:
        """Wählt das beste verfügbare Modell basierend auf dem Prompt und der Historie aus.
        
        Args:
            prompt: Die aktuelle Benutzereingabe
            conversation_history: Liste der bisherigen Nachrichten im Chat
            
        Returns:
            Das am besten geeignete Modell
        """
        try:
            # Wenn nur ein Modell verfügbar ist, dieses zurückgeben
            if len(self.models) == 1:
                return next(iter(self.models.values()))
                
            # Standardmäßig das erste verfügbare Modell verwenden
            default_model = next(iter(self.models.values()))
            
            # Hier könnte eine erweiterte Logik zur Auswahl des besten Modells stehen
            # Zum Beispiel basierend auf der Komplexität des Prompts, der Konversationshistorie, etc.
            
            # Beispiel: Verwende ein spezielles Modell für Code-bezogene Anfragen
            code_keywords = ['code', 'programmieren', 'funktion', 'klasse', 'variable', 'debug', 'fehler']
            if any(keyword in prompt.lower() for keyword in code_keywords):
                for model in self.models.values():
                    if hasattr(model, 'model_type') and model.model_type == ModelType.CODE:
                        return model
            
            return default_model
            
        except Exception as e:
            logger.error(f"Fehler bei der Modellauswahl: {e}")
            # Im Fehlerfall das erste verfügbare Modell zurückgeben
            return next(iter(self.models.values())) if self.models else None
    
    def generate_response(
        self,
        prompt: str,
        system_prompt: str = "",
        context: str = "",
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        model_name: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generiert eine Antwort auf den gegebenen Prompt.
        
        Args:
            prompt: Die Benutzereingabe
            system_prompt: System-Prompt für die Konfiguration des Modells
            context: Zusätzlicher Kontext für die Antwortgenerierung
            conversation_history: Liste der bisherigen Nachrichten im Chat
            model_name: Optionaler spezifischer Modellname, der verwendet werden soll
            **kwargs: Zusätzliche Parameter für die Generierung
            
        Returns:
            Die generierte Antwort als String
        """
        try:
            # Bestes Modell auswählen, falls keins angegeben
            if model_name is None:
                model = self.get_best_model(prompt, conversation_history or [])
            else:
                model = self.models.get(model_name)
                if model is None:
                    raise ValueError(f"Modell {model_name} nicht gefunden")
            
            # Kontext erstellen
            messages = []
            
            # System-Prompt hinzufügen
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Konversationsverlauf hinzufügen
            if conversation_history:
                for msg in conversation_history[-5:]:  # Begrenze die Historie
                    messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    })
            
            # Aktuellen Kontext hinzufügen
            if context:
                messages.append({"role": "system", "content": f"Kontext: {context}"})
            
            # Aktuelle Nachricht hinzufügen
            messages.append({"role": "user", "content": prompt})
            
            # Antwort generieren
            response = model.create_chat_completion(
                messages=messages,
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 512),
                top_p=kwargs.get("top_p", 0.9),
                top_k=kwargs.get("top_k", 40),
                repeat_penalty=kwargs.get("repeat_penalty", 1.1),
                stop=kwargs.get("stop", ["\n###"])
            )
            
            # Antwort extrahieren
            if "choices" in response and len(response["choices"]) > 0:
                return response["choices"][0]["message"]["content"].strip()
            else:
                raise ValueError("Keine gültige Antwort vom Modell erhalten")
                
        except Exception as e:
            logger.error(f"Fehler bei der Antwortgenerierung: {e}", exc_info=True)
            return f"Entschuldigung, ein Fehler ist aufgetreten: {str(e)}"
    
    def _discover_models(self) -> None:
        """Durchsucht das Modellverzeichnis nach verfügbaren Modellen."""
        try:
            # Standardmodellkonfigurationen
            default_models = [
                ModelConfig(
                    name="llama-2-7b-chat",
                    path=str(self.models_dir / "llama-2-7b-chat.gguf"),
                    model_type=ModelType.GENERAL,
                    n_ctx=2048,
                    n_threads=4,
                    n_gpu_layers=0
                ),
                ModelConfig(
                    name="codellama-7b",
                    path=str(self.models_dir / "codellama-7b.gguf"),
                    model_type=ModelType.CODE,
                    chat_format="codellama",
                    n_ctx=4096,
                    n_threads=4,
                    n_gpu_layers=0
                )
            ]
            
            # Standardmodelle zur Konfiguration hinzufügen, falls nicht vorhanden
            for model_cfg in default_models:
                if model_cfg.name not in self.model_configs:
                    self.model_configs[model_cfg.name] = model_cfg
            
            # Benutzerdefinierte Modelle aus der Konfiguration laden
            model_section = 'MODELS'
            if self.config.has_section(model_section):
                for model_name in self.config.options(model_section):
                    if model_name in self.model_configs:
                        continue  # Überspringe bereits geladene Modelle
                    
                    try:
                        model_path = self.config.get(model_section, model_name)
                        if not os.path.isabs(model_path):
                            model_path = str(self.models_dir / model_path)
                        
                        model_cfg = ModelConfig(
                            name=model_name,
                            path=model_path,
                            model_type=ModelType(self.config.get(f"{model_section}.{model_name}", "type", fallback="general")),
                            chat_format=self.config.get(f"{model_section}.{model_name}", "chat_format", fallback="llama-2"),
                            n_ctx=int(self.config.get(f"{model_section}.{model_name}", "n_ctx", fallback=2048)),
                            n_threads=int(self.config.get(f"{model_section}.{model_name}", "n_threads", fallback=4)),
                            n_gpu_layers=int(self.config.get(f"{model_section}.{model_name}", "n_gpu_layers", fallback=0)),
                            temperature=float(self.config.get(f"{model_section}.{model_name}", "temperature", fallback=0.7)),
                            max_tokens=int(self.config.get(f"{model_section}.{model_name}", "max_tokens", fallback=512)),
                            top_p=float(self.config.get(f"{model_section}.{model_name}", "top_p", fallback=0.9)),
                            top_k=int(self.config.get(f"{model_section}.{model_name}", "top_k", fallback=40)),
                            repeat_penalty=float(self.config.get(f"{model_section}.{model_name}", "repeat_penalty", fallback=1.1))
                        )
                        
                        self.model_configs[model_name] = model_cfg
                        logger.info(f"Modellkonfiguration geladen: {model_name}")
                        
                    except Exception as e:
                        logger.error(f"Fehler beim Laden der Konfiguration für Modell {model_name}: {e}")
            
            logger.info(f"Insgesamt {len(self.model_configs)} Modellkonfigurationen geladen")
            
        except Exception as e:
            logger.error(f"Fehler bei der Modellerkennung: {e}")
    
    def _start_background_loader(self) -> None:
        """Startet einen Hintergrund-Thread zum Laden von Modellen."""
        self.loader_thread = threading.Thread(target=self._loader_worker, daemon=True)
        self.loader_thread.start()
        logger.info("Hintergrund-Thread für Modellladen gestartet")
    
    def _loader_worker(self) -> None:
        """Arbeiter-Thread zum Laden von Modellen."""
        while True:
            try:
                model_name = self.loading_queue.get()
                if model_name is None:  # Beenden-Signal
                    break
                
                self._load_model_sync(model_name)
                
            except Exception as e:
                logger.error(f"Fehler im Modell-Lade-Thread: {e}")
            finally:
                self.loading_queue.task_done()
    
    def _load_model(self, model_name: str) -> None:
        """Lädt ein Modell mit optimierter Speichernutzung für M2.
        
        Args:
            model_name: Der Name des zu ladenden Modells
        """
        if model_name in self.models:
            return  # Modell bereits geladen
        
        if model_name not in self.model_configs:
            logger.error(f"Keine Konfiguration für Modell {model_name} gefunden")
            return
        
        model_cfg = self.model_configs[model_name]
        
        try:
            logger.info(f"Lade Modell mit optimierter Speichernutzung: {model_name}...")
            start_time = time.time()
            
            # Bereinige Speicher vor dem Laden
            self._cleanup_temp_files()
            
            # Überprüfe, ob die Modell-Datei existiert
            if not os.path.exists(model_cfg.path):
                logger.error(f"Modell-Datei nicht gefunden: {model_cfg.path}")
                return
            
            # Optimierte Parameter für M2
            model_params = {
                'model_path': model_cfg.path,
                'n_ctx': min(model_cfg.n_ctx, 4096),  # Begrenze den Kontext
                'n_threads': max(1, os.cpu_count() // 2),  # Verwendet die Hälfte der Kerne
                'n_gpu_layers': min(model_cfg.n_gpu_layers, 35),  # Begrenze GPU-Layer
                'n_batch': 512,  # Kleinere Batch-Größe
                'n_threads_batch': max(1, os.cpu_count() // 2),  # Threads für Batch-Verarbeitung
                'offload_kqv': True,  # Entlastet den GPU-Speicher
                'last_n_tokens_size': 64,  # Reduzierte Größe des Token-Cache
                'seed': 42,
                'f16_kv': True,  # 16-bit Key-Value Cache
                'use_mmap': True,  # Memory Mapping für große Modelle
                'use_mlock': False,  # Verhindert Swapping, benötigt aber mehr RAM
                'use_mutex': True,  # Thread-Sicherheit
                'vocab_only': False,
                'verbose': False,  # Weniger Ausgaben für bessere Leistung
                'rope_freq_base': 10000,
                'rope_freq_scale': 1.0,
                'low_vram': False,
                'mul_mat_q': True,
                'logits_all': False,
                'embedding': False
            }
            
            # Füge chat_format nur hinzu, wenn benötigt
            if hasattr(model_cfg, 'chat_format'):
                model_params['chat_format'] = model_cfg.chat_format
            
            logger.debug(f"Lade Modell mit optimierten Parametern: {model_params}")
            
            # Modell initialisieren
            model = Llama(**model_params)
            
            # Bereinige temporäre Dateien nach dem Laden
            self._cleanup_temp_files()
            
            # Speicherverbrauch protokollieren
            process = psutil.Process(os.getpid())
            logger.info(f"Modell geladen. Speichernutzung: {process.memory_info().rss / 1024 / 1024:.2f} MB")
            
            # Modell im Cache speichern
            with self.load_lock:
                self.models[model_name] = model
            
            load_time = time.time() - start_time
            logger.info(f"Modell {model_name} in {load_time:.2f}s geladen")
            
        except Exception as e:
            logger.error(f"Fehler beim Laden des Modells {model_name}: {e}")
    
    def queue_model_for_loading(self, model_name: str) -> None:
        """Fügt ein Modell zur Lade-Warteschlange hinzu.
        
        Args:
            model_name: Der Name des zu ladenden Modells
        """
        if model_name not in self.model_configs:
            logger.warning(f"Kann Modell nicht laden: Unbekanntes Modell {model_name}")
            return
        
        if model_name not in self.models:
            self.loading_queue.put(model_name)
            logger.debug(f"Modell {model_name} zur Lade-Warteschlange hinzugefügt")
    
    def get_model(self, model_name: str) -> Optional[Llama]:
        """Gibt ein geladenes Modell zurück.
        
        Args:
            model_name: Der Name des Modells
            
        Returns:
            Die Llama-Instanz oder None, wenn nicht geladen
        """
        return self.models.get(model_name)
    
    def unload_model(self, model_name: str) -> None:
        """Entlädt ein Modell aus dem Speicher.
        
        Args:
            model_name: Der Name des zu entladenden Modells
        """
        with self.load_lock:
            if model_name in self.models:
                del self.models[model_name]
                logger.info(f"Modell {model_name} wurde entladen")
    
    def get_best_model(self, prompt: str, conversation_history: List[Dict] = None) -> str:
        """Wählt das beste Modell für den gegebenen Prompt aus.
        
        Args:
            prompt: Der Eingabe-Prompt
            conversation_history: Optional: Die bisherige Konversationshistorie
            
        Returns:
            Der Name des besten Modells
        """
        return self.smart_selector.select_model(prompt, conversation_history)
    
    def generate_response(
        self, 
        prompt: str, 
        model_name: str = None,
        conversation_history: List[Dict] = None,
        max_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        top_k: int = None,
        repeat_penalty: float = None,
        stop: List[str] = None
    ) -> Dict[str, Any]:
        """Generiert eine Antwort auf den gegebenen Prompt.
        
        Args:
            prompt: Der Eingabe-Prompt
            model_name: Optional: Name des zu verwendenden Modells
            conversation_history: Optional: Die bisherige Konversationshistorie
            max_tokens: Maximale Anzahl der zu generierenden Tokens
            temperature: Kreativitätsparameter (0.0 - 1.0)
            top_p: Nukleus-Sampling-Parameter
            top_k: Top-K-Sampling-Parameter
            repeat_penalty: Bestrafung für Wiederholungen
            stop: Liste von Stopp-Tokens
            
        Returns:
            Ein Dictionary mit der generierten Antwort und Metadaten
        """
        # Wähle das beste Modell, falls keins angegeben
        if model_name is None:
            model_name = self.get_best_model(prompt, conversation_history)
        
        # Lade das Modell, falls noch nicht geschehen
        if model_name not in self.models:
            self.queue_model_for_loading(model_name)
            return {
                "error": f"Modell {model_name} wird geladen. Bitte versuchen Sie es in einigen Sekunden erneut.",
                "model": model_name,
                "status": "loading"
            }
        
        model = self.models[model_name]
        model_cfg = self.model_configs.get(model_name, ModelConfig(name=model_name, path=""))
        
        # Standardwerte aus der Modellkonfiguration verwenden, falls nicht überschrieben
        if max_tokens is None:
            max_tokens = model_cfg.max_tokens
        if temperature is None:
            temperature = model_cfg.temperature
        if top_p is None:
            top_p = model_cfg.top_p
        if top_k is None:
            top_k = model_cfg.top_k
        if repeat_penalty is None:
            repeat_penalty = model_cfg.repeat_penalty
        if stop is None:
            stop = model_cfg.stop
        
        try:
            # Starte das Performance-Monitoring
            self.performance_monitor.start_generation()
            start_time = time.time()
            
            # Generiere die Antwort
            response = model.create_chat_completion(
                messages=conversation_history or [{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                stop=stop,
                stream=False
            )
            
            # Berechne die Generierungszeit
            generation_time = time.time() - start_time
            
            # Extrahiere die generierte Antwort
            if "choices" in response and len(response["choices"]) > 0:
                generated_text = response["choices"][0]["message"]["content"]
                tokens_used = response.get("usage", {}).get("total_tokens", 0)
                tokens_per_second = tokens_used / generation_time if generation_time > 0 else 0
                
                # Aktualisiere die Performance-Metriken
                self.performance_monitor.record_generation(
                    tokens=tokens_used,
                    duration=generation_time,
                    model=model_name
                )
                
                return {
                    "response": generated_text,
                    "model": model_name,
                    "tokens_used": tokens_used,
                    "generation_time": generation_time,
                    "tokens_per_second": tokens_per_second,
                    "finish_reason": response["choices"][0].get("finish_reason", "unknown"),
                    "status": "success"
                }
            else:
                return {
                    "error": "Keine Antwort vom Modell erhalten",
                    "model": model_name,
                    "status": "error"
                }
                
        except Exception as e:
            logger.error(f"Fehler bei der Antwortgenerierung mit Modell {model_name}: {e}")
            return {
                "error": str(e),
                "model": model_name,
                "status": "error"
            }
    
    def generate_response_async(
            self, 
            prompt: str, 
            model_name: str = None,
            conversation_history: List[Dict] = None,
            max_tokens: int = None,
            temperature: float = None,
            top_p: float = None,
            top_k: int = None,
            repeat_penalty: float = None,
            stop: List[str] = None
        ) -> str:
        """Synchrone Version von generate_response.
        
        Diese Methode führt die Antwortgenerierung direkt aus.
        """
        try:
            # Bestes Modell auswählen, falls keins angegeben
            if model_name is None:
                model = self.get_best_model(prompt, conversation_history or [])
            else:
                model = self.models.get(model_name)
                if model is None:
                    raise ValueError(f"Modell {model_name} nicht gefunden")
            
            # Kontext erstellen
            messages = []
            
            # System-Prompt hinzufügen (falls vorhanden)
            if hasattr(self, '_create_system_prompt'):
                system_prompt = self._create_system_prompt()
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
            
            # Konversationsverlauf hinzufügen
            if conversation_history:
                for msg in conversation_history[-5:]:  # Begrenze die Historie
                    messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    })
            
            # Aktuelle Nachricht hinzufügen
            messages.append({"role": "user", "content": prompt})
            
            # Antwort generieren
            response = model.create_chat_completion(
                messages=messages,
                temperature=temperature or 0.7,
                max_tokens=max_tokens or 512,
                top_p=top_p or 0.9,
                top_k=top_k or 40,
                repeat_penalty=repeat_penalty or 1.1,
                stop=stop or ["\n###"]
            )
            
            # Antwort extrahieren
            if "choices" in response and len(response["choices"]) > 0:
                return response["choices"][0]["message"]["content"].strip()
            else:
                raise ValueError("Keine gültige Antwort vom Modell erhalten")
                
        except Exception as e:
            logger.error(f"Fehler bei der Antwortgenerierung: {e}", exc_info=True)
            return f"Entschuldigung, ein Fehler ist aufgetreten: {str(e)}"
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Gibt eine Liste der verfügbaren Modelle zurück.
        
        Returns:
            Eine Liste von Dictionaries mit Modellinformationen
        """
        models = []
        
        for name, cfg in self.model_configs.items():
            models.append({
                "name": name,
                "type": cfg.model_type.value,
                "path": cfg.path,
                "loaded": name in self.models,
                "config": {
                    "n_ctx": cfg.n_ctx,
                    "n_threads": cfg.n_threads,
                    "n_gpu_layers": cfg.n_gpu_layers,
                    "temperature": cfg.temperature,
                    "max_tokens": cfg.max_tokens
                }
            })
        
        return models
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Gibt die Konfiguration für ein bestimmtes Modell zurück.
        
        Args:
            model_name: Der Name des Modells
            
        Returns:
            Die ModelConfig-Instanz oder None, wenn nicht gefunden
        """
        return self.model_configs.get(model_name)
    
    def update_model_config(self, model_name: str, **kwargs) -> bool:
        """Aktualisiert die Konfiguration für ein Modell.
        
        Args:
            model_name: Der Name des Modells
            **kwargs: Zu aktualisierende Konfigurationsparameter
            
        Returns:
            True, wenn die Aktualisierung erfolgreich war, sonst False
        """
        if model_name not in self.model_configs:
            logger.warning(f"Kann Konfiguration nicht aktualisieren: Unbekanntes Modell {model_name}")
            return False
        
        try:
            model_cfg = self.model_configs[model_name]
            
            # Aktualisiere die Konfiguration
            for key, value in kwargs.items():
                if hasattr(model_cfg, key):
                    setattr(model_cfg, key, value)
            
            # Speichere die aktualisierte Konfiguration
            self._save_model_config(model_cfg)
            
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Aktualisieren der Modellkonfiguration: {e}")
            return False
    
    def _save_model_config(self, model_cfg: ModelConfig) -> None:
        """Speichert die Konfiguration für ein Modell.
        
        Args:
            model_cfg: Die zu speichernde ModelConfig-Instanz
        """
        section = f"MODEL.{model_cfg.name}"
        
        # Aktuelle Konfiguration speichern
        self.config.set(section, "path", model_cfg.path)
        self.config.set(section, "type", model_cfg.model_type.value)
        self.config.set(section, "chat_format", model_cfg.chat_format)
        self.config.set(section, "n_ctx", str(model_cfg.n_ctx))
        self.config.set(section, "n_threads", str(model_cfg.n_threads))
        self.config.set(section, "n_gpu_layers", str(model_cfg.n_gpu_layers))
        self.config.set(section, "temperature", str(model_cfg.temperature))
        self.config.set(section, "max_tokens", str(model_cfg.max_tokens))
        self.config.set(section, "top_p", str(model_cfg.top_p))
        self.config.set(section, "top_k", str(model_cfg.top_k))
        self.config.set(section, "repeat_penalty", str(model_cfg.repeat_penalty))
        
        # Konfiguration speichern
        self.config.save()
        
        logger.info(f"Konfiguration für Modell {model_cfg.name} gespeichert")
    
    def cleanup(self) -> None:
        """Bereinigt Ressourcen und stoppt Hintergrund-Threads."""
        # Signalisiere dem Lade-Thread, dass er beenden soll
        self.loading_queue.put(None)
        
        # Warte auf das Beenden des Lade-Threads
        if self.loader_thread.is_alive():
            self.loader_thread.join(timeout=5.0)
        
        # Entlade alle Modelle
        with self.load_lock:
            for model_name in list(self.models.keys()):
                self.unload_model(model_name)
        
        logger.info("Modell-Manager wurde bereinigt")
