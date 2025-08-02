import configparser
import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional

# Erstelle einen Logger
logger = logging.getLogger(__name__)

class ConfigManager:
    """Verwaltet die Anwendungskonfiguration."""
    
    def __init__(self, config_file=None):
        """Initialisiert den ConfigManager.
        
        Args:
            config_file: Pfad zur Konfigurationsdatei (optional)
        """
        self.config = configparser.ConfigParser()
        
        # Standardkonfigurationsdatei im gleichen Verzeichnis wie das Skript
        if config_file is None:
            self.config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini')
        else:
            self.config_file = os.path.abspath(config_file)
            
        # Erstelle das Verzeichnis, falls es nicht existiert
        config_dir = os.path.dirname(self.config_file)
        if config_dir:  # Nur erstellen, wenn ein Verzeichnis angegeben ist
            os.makedirs(config_dir, exist_ok=True)
            
        self.load_config()
    
    def load_config(self) -> None:
        """Lädt die Konfiguration aus der Datei oder erstellt eine Standardkonfiguration."""
        if os.path.exists(self.config_file):
            try:
                self.config.read(self.config_file, encoding='utf-8')
            except Exception as e:
                logger.error(f"Fehler beim Laden der Konfiguration: {e}")
                self.create_default_config()
        else:
            self.create_default_config()
    
    def create_default_config(self) -> None:
        """Erstellt eine Standardkonfiguration."""
        self.config['MODEL'] = {
            'default_model': 'llama-2-7b-chat.gguf',
            'n_ctx': '2048',
            'n_threads': '4',
            'n_gpu_layers': '0',
            'temperature': '0.7',
            'max_tokens': '512',
            'top_p': '0.9',
            'top_k': '40',
            'repeat_penalty': '1.1',
            'stop': '[INST]',
            'echo': 'False'
        }
        
        self.config['UI'] = {
            'theme': 'system',
            'font_size': '12',
            'font_family': 'Arial',
            'window_width': '1000',
            'window_height': '700',
            'sidebar_width': '250',
            'chat_width': '700'
        }
        
        self.config['FEATURES'] = {
            'speech_recognition': 'False',
            'encryption': 'False',
            'auto_update': 'True',
            'notifications': 'True',
            'analytics': 'False'
        }
        
        self.config['PATHS'] = {
            'models': 'models',
            'downloads': 'downloads',
            'exports': 'exports',
            'plugins': 'plugins'
        }
        
        self.save_config()
    
    def save_config(self):
        """Speichert die Konfiguration in die Datei."""
        try:
            # Stelle sicher, dass das Verzeichnis existiert
            config_dir = os.path.dirname(self.config_file)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir, exist_ok=True)
                
            with open(self.config_file, 'w') as configfile:
                self.config.write(configfile)
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Konfiguration: {e}")
            # Wir werfen die Ausnahme nicht weiter, um die Anwendung nicht zum Absturz zu bringen
            # Stattdessen loggen wir den Fehler und fahren fort
    
    def get(self, section: str, key: str, fallback: Any = None) -> Any:
        """Holt einen Konfigurationswert.
        
        Args:
            section: Der Abschnitt in der Konfiguration
            key: Der Schlüssel des Werts
            fallback: Der Standardwert, falls der Schlüssel nicht existiert
            
        Returns:
            Der Wert des Schlüssels oder der Fallback-Wert
        """
        try:
            return self.config.get(section, key, fallback=fallback)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return fallback
    
    def set(self, section: str, key: str, value: Any) -> None:
        """Setzt einen Konfigurationswert.
        
        Args:
            section: Der Abschnitt in der Konfiguration
            key: Der Schlüssel des Werts
            value: Der zu setzende Wert
        """
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = str(value)
        self.save_config()
    
    def get_section(self, section: str) -> Dict[str, str]:
        """Holt einen ganzen Abschnitt der Konfiguration.
        
        Args:
            section: Der Name des Abschnitts
            
        Returns:
            Ein Dictionary mit den Schlüssel-Wert-Paaren des Abschnitts
        """
        if section in self.config:
            return dict(self.config[section])
        return {}
    
    def set_section(self, section: str, values: Dict[str, Any]) -> None:
        """Setzt einen ganzen Abschnitt der Konfiguration.
        
        Args:
            section: Der Name des Abschnitts
            values: Ein Dictionary mit den neuen Werten
        """
        self.config[section] = {}
        for key, value in values.items():
            self.config[section][key] = str(value)
        self.save_config()
