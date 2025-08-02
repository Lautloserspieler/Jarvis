import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ModelType(str, Enum):
    """Enum für verschiedene Modelltypen."""
    CODE = "code"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    GENERAL = "general"
    EMOTIONAL = "emotional"

@dataclass
class ModelPattern:
    """Definiert ein Muster für die Modellauswahl."""
    pattern: re.Pattern
    model_type: ModelType
    priority: int = 5  # 1-10, wobei 10 die höchste Priorität hat

class SmartModelSelector:
    """Wählt basierend auf dem Eingabetext das am besten geeignete Modell aus."""
    
    def __init__(self):
        """Initialisiert den SmartModelSelector mit Standardmustern."""
        self.patterns = self._initialize_patterns()
    
    def _initialize_patterns(self) -> List[ModelPattern]:
        """Initialisiert die Standardmuster für die Modellauswahl.
        
        Returns:
            Eine Liste von ModelPattern-Objekten
        """
        return [
            # Code-bezogene Muster
            ModelPattern(
                pattern=re.compile(r'\b(?:code|programm|funktion|klasse|methode|variable|syntax|fehler|debug|test|entwickl|api|import|from|def |class |\(.*\):)\b', re.IGNORECASE),
                model_type=ModelType.CODE,
                priority=9
            ),
            
            # Analytische Muster
            ModelPattern(
                pattern=re.compile(r'\b(?:analys|vergleichen|vergleich|untersuchen|untersuchung|statistik|daten|auswert|studie|forschung|wissen|fakten|erklären|erklärung|warum|wieso|weshalb|wie funktioniert)\b', re.IGNORECASE),
                model_type=ModelType.ANALYTICAL,
                priority=8
            ),
            
            # Kreative Muster
            ModelPattern(
                pattern=re.compile(r'\b(?:geschichte|erzähl|gedicht|lied|kreativ|fantasie|erfinden|vorstellen|was wäre wenn|stell dir vor|schreibe eine|schreib mir)\b', re.IGNORECASE),
                model_type=ModelType.CREATIVE,
                priority=7
            ),
            
            # Emotionale Muster
            ModelPattern(
                pattern=re.compile(r'\b(?:gefühl|fühle|glücklich|traurig|wütend|ängstlich|freude|liebe|hass|einsam|nervös|gestresst|deprimiert|begeistert|motiviert|enttäuscht|enttäuschung|freundschaft|beziehung)\b', re.IGNORECASE),
                model_type=ModelType.EMOTIONAL,
                priority=6
            ),
            
            # Allgemeine Muster (niedrige Priorität)
            ModelPattern(
                pattern=re.compile(r'\b(?:hallo|hey|hi|guten tag|guten morgen|guten abend|wie geht\'s|was geht|was läuft|hilf|hilfe|frage|fragen|verstehe|verstanden|danke|bitte|entschuldigung|tut mir leid)\b', re.IGNORECASE),
                model_type=ModelType.GENERAL,
                priority=3
            )
        ]
    
    def select_model(self, prompt: str, conversation_history: List[Dict] = None) -> str:
        """Wählt das beste Modell für den gegebenen Prompt aus.
        
        Args:
            prompt: Der Eingabe-Prompt
            conversation_history: Optional: Die bisherige Konversationshistorie
            
        Returns:
            Der Name des ausgewählten Modells
        """
        if not prompt and not conversation_history:
            return "llama-2-7b-chat"  # Standardmodell
        
        # Analysiere den Prompt und die Konversationshistorie
        text_to_analyze = prompt.lower()
        
        if conversation_history:
            # Füge die letzten 5 Nachrichten der Konversation zur Analyse hinzu
            for msg in conversation_history[-5:]:
                role = msg.get('role', '').lower()
                content = msg.get('content', '').lower()
                if content:
                    text_to_analyze += f" {content}"
        
        # Bestimme den Modelltyp basierend auf den Mustern
        model_scores = {
            ModelType.CODE: 0,
            ModelType.ANALYTICAL: 0,
            ModelType.CREATIVE: 0,
            ModelType.EMOTIONAL: 0,
            ModelType.GENERAL: 1  # Standardwert für allgemeine Anfragen
        }
        
        # Bewerte jeden Mustertyp
        for pattern in self.patterns:
            if pattern.pattern.search(text_to_analyze):
                model_scores[pattern.model_type] += pattern.priority
        
        # Wähle den Modelltyp mit der höchsten Punktzahl
        selected_type = max(model_scores.items(), key=lambda x: x[1])[0]
        
        # Wähle das konkrete Modell basierend auf dem Typ
        model_mapping = {
            ModelType.CODE: "codellama-7b",
            ModelType.ANALYTICAL: "llama-2-13b-chat",
            ModelType.CREATIVE: "llama-2-7b-chat",
            ModelType.EMOTIONAL: "llama-2-7b-chat",
            ModelType.GENERAL: "llama-2-7b-chat"
        }
        
        selected_model = model_mapping.get(selected_type, "llama-2-7b-chat")
        
        logger.debug(f"Ausgewähltes Modell: {selected_model} (Typ: {selected_type.value}, Score: {model_scores[selected_type]})")
        
        return selected_model
    
    def add_custom_pattern(self, pattern: str, model_type: ModelType, priority: int = 5) -> None:
        """Fügt ein benutzerdefiniertes Muster für die Modellauswahl hinzu.
        
        Args:
            pattern: Das Regex-Muster
            model_type: Der zuzuordnende Modelltyp
            priority: Die Priorität des Musters (1-10)
        """
        try:
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
            self.patterns.append(
                ModelPattern(
                    pattern=compiled_pattern,
                    model_type=model_type,
                    priority=max(1, min(10, priority))  # Auf 1-10 begrenzen
                )
            )
            logger.info(f"Benutzerdefiniertes Muster für Modelltyp {model_type} hinzugefügt")
        except re.error as e:
            logger.error(f"Ungültiges Regex-Muster: {e}")
    
    def remove_pattern(self, pattern: str) -> bool:
        """Entfernt ein Muster anhand des Regex-Strings.
        
        Args:
            pattern: Das zu entfernende Regex-Muster
            
        Returns:
            True, wenn das Muster gefunden und entfernt wurde, sonst False
        """
        initial_count = len(self.patterns)
        self.patterns = [p for p in self.patterns if p.pattern.pattern != pattern]
        
        if len(self.patterns) < initial_count:
            logger.info(f"Muster '{pattern}' wurde entfernt")
            return True
        
        logger.warning(f"Muster '{pattern}' nicht gefunden")
        return False
