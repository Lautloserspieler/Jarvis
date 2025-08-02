import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import deque
import statistics
import logging

logger = logging.getLogger(__name__)

@dataclass
class GenerationStats:
    """Statistiken für eine einzelne Generierung."""
    tokens: int = 0
    duration: float = 0.0
    tokens_per_second: float = 0.0
    model: str = ""
    timestamp: float = field(default_factory=time.time)

class PerformanceMonitor:
    """Überwacht die Leistung der Modellgenerierung."""
    
    def __init__(self, window_size: int = 100):
        """Initialisiert den PerformanceMonitor.
        
        Args:
            window_size: Größe des Fensters für gleitende Durchschnitte
        """
        self.window_size = window_size
        self.generations: deque[GenerationStats] = deque(maxlen=window_size)
        self.current_start: Optional[float] = None
        self.current_tokens: int = 0
        self.model_stats: Dict[str, Dict[str, float]] = {}
        self._is_monitoring = False
    
    def start_generation(self) -> None:
        """Startet die Zeitmessung für eine neue Generierung."""
        self.current_start = time.time()
        self.current_tokens = 0
    
    def record_generation(self, tokens: int, duration: float, model: str) -> None:
        """Zeichnet die Statistiken für eine abgeschlossene Generierung auf.
        
        Args:
            tokens: Anzahl der generierten Tokens
            duration: Dauer der Generierung in Sekunden
            model: Name des verwendeten Modells
        """
        if tokens <= 0 or duration <= 0:
            return
        
        tps = tokens / duration
        stats = GenerationStats(
            tokens=tokens,
            duration=duration,
            tokens_per_second=tps,
            model=model
        )
        
        self.generations.append(stats)
        
        # Aktualisiere die Modellstatistiken
        if model not in self.model_stats:
            self.model_stats[model] = {
                'total_tokens': 0,
                'total_duration': 0.0,
                'count': 0,
                'avg_tps': 0.0
            }
        
        model_stat = self.model_stats[model]
        model_stat['total_tokens'] += tokens
        model_stat['total_duration'] += duration
        model_stat['count'] += 1
        model_stat['avg_tps'] = model_stat['total_tokens'] / model_stat['total_duration'] if model_stat['total_duration'] > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Gibt die aktuellen Leistungsstatistiken zurück.
        
        Returns:
            Ein Dictionary mit Leistungsstatistiken
        """
        if not self.generations:
            return {
                'avg_tokens_per_second': 0.0,
                'total_generations': 0,
                'total_tokens': 0,
                'model_stats': {}
            }
            
        # Berechne Statistiken über alle Generationen
        all_tps = [g.tokens_per_second for g in self.generations]
        avg_tps = sum(all_tps) / len(all_tps)
        
        # Berechne Modell-spezifische Statistiken
        model_stats = {}
        for model, stats in self.model_stats.items():
            if stats['count'] > 0:
                model_stats[model] = {
                    'avg_tokens_per_second': stats['total_tokens'] / stats['total_duration'],
                    'count': stats['count'],
                    'total_tokens': stats['total_tokens']
                }
        
        return {
            'avg_tokens_per_second': avg_tps,
            'total_generations': len(self.generations),
            'total_tokens': sum(g.tokens for g in self.generations),
            'model_stats': model_stats
        }
        
    def start_monitoring(self) -> None:
        """Startet die Überwachung der Leistung."""
        self._is_monitoring = True
        
    def stop_monitoring(self) -> None:
        """Beendet die Überwachung der Leistung."""
        self._is_monitoring = False
        # Optional: Hier können zusätzliche Aufräumarbeiten durchgeführt werden
        self.generations.clear()
        self.model_stats.clear()
    
    def get_model_stats(self, model: str) -> Dict[str, float]:
        """Gibt die Statistiken für ein bestimmtes Modell zurück.
        
        Args:
            model: Der Name des Modells
            
        Returns:
            Ein Dictionary mit den Modellstatistiken
        """
        return self.model_stats.get(model, {})
    
    def reset(self) -> None:
        """Setzt alle Statistiken zurück."""
        self.generations.clear()
        self.model_stats = {}
        self.current_start = None
        self.current_tokens = 0
