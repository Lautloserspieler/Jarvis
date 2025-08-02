from datetime import datetime
from typing import List, Dict, Optional, Set
import json
import os
from pathlib import Path

class ContextScorer:
    """Bewertet die Relevanz von Nachrichten basierend auf Benutzerzielen und Interessen."""
    
    def __init__(self):
        self.user_goals: Set[str] = set()
        self.interests: Set[str] = set()
        self.keyword_weights = {
            'ziel': 2.0,
            'wichtig': 1.5,
            'projekt': 1.5,
            'hobby': 1.2,
            'interessiere': 1.2,
            'möchte': 1.1
        }
    
    def update_goals(self, goals: List[str]):
        """Aktualisiert die Benutzerziele."""
        self.user_goals.update(goal.lower() for goal in goals)
    
    def update_interests(self, interests: List[str]):
        """Aktualisiert die Benutzerinteressen."""
        self.interests.update(interest.lower() for interest in interests)
    
    def score_relevance(self, message: str) -> float:
        """Bewertet die Relevanz einer Nachricht."""
        if not message.strip():
            return 0.0
            
        score = 0.0
        message_lower = message.lower()
        
        # Bewerte nach Schlüsselwörtern
        for keyword, weight in self.keyword_weights.items():
            if keyword in message_lower:
                score += weight
        
        # Bewerte nach Übereinstimmungen mit Zielen/Interessen
        for goal in self.user_goals:
            if goal and goal.lower() in message_lower:
                score += 2.0  # Höheres Gewicht für direkte Übereinstimmungen
        
        for interest in self.interests:
            if interest and interest.lower() in message_lower:
                score += 1.5
        
        # Normalisiere den Score (0-10 Skala)
        return min(10.0, score) / 10.0
        
    def extract_keywords(self, text: str, top_n: int = 5) -> List[tuple]:
        """
        Extrahiert die wichtigsten Schlüsselwörter aus einem Text.
        
        Args:
            text: Der zu analysierende Text
            top_n: Anzahl der zurückzugebenden Keywords
            
        Returns:
            Liste von Tupeln im Format (keyword, score)
        """
        from collections import defaultdict
        import re
        
        if not text.strip():
            return []
            
        # Wörter zählen (einfache Implementierung)
        word_counts = defaultdict(int)
        words = re.findall(r'\b\w{3,}\b', text.lower())  # Mindestens 3 Buchstaben
        
        for word in words:
            # Überspringe Stoppwörter
            if word in {'der', 'die', 'das', 'und', 'oder', 'in', 'auf', 'ist', 'sind', 'ein', 'eine'}:
                continue
                
            # Erhöhe die Gewichtung für Wörter, die in den Benutzerzielen oder Interessen vorkommen
            weight = 1.0
            if any(goal in word for goal in self.user_goals):
                weight += 2.0
            elif any(interest in word for interest in self.interests):
                weight += 1.5
                
            word_counts[word] += weight
        
        # Sortiere nach Häufigkeit und Gewichtung
        sorted_keywords = sorted(
            word_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Beschränke auf die Top-N Schlüsselwörter
        return sorted_keywords[:top_n]


class LongTermMemory:
    """Verwaltet das Langzeitgedächtnis mit priorisierten Fakten."""
    
    def __init__(self, storage_file: str = "memory.json"):
        self.storage_file = storage_file
        self.pinned_facts: Dict[str, dict] = {}
        self.next_id = 1
        self._load_memory()
    
    def _load_memory(self):
        """Lädt gespeicherte Fakten aus der Datei."""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.pinned_facts = data.get('facts', {})
                    self.next_id = data.get('next_id', 1)
            except Exception as e:
                print(f"Fehler beim Laden des Gedächtnisses: {e}")
    
    def _save_memory(self):
        """Speichert die Fakten in einer Datei."""
        try:
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'facts': self.pinned_facts,
                    'next_id': self.next_id
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Fehler beim Speichern des Gedächtnisses: {e}")
    
    def add_fact(self, fact: str, priority: int = 1, source: str = "user", metadata: Optional[dict] = None) -> str:
        """Fügt einen neuen Fakt zum Gedächtnis hinzu."""
        fact_id = f"fact_{self.next_id}"
        self.pinned_facts[fact_id] = {
            "fact": fact,
            "priority": min(max(1, priority), 5),  # 1-5 Skala
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "metadata": metadata or {}
        }
        self.next_id += 1
        self._save_memory()
        return fact_id
    
    def remove_fact(self, fact_id: str) -> bool:
        """Entfernt einen Fakt anhand seiner ID."""
        if fact_id in self.pinned_facts:
            del self.pinned_facts[fact_id]
            self._save_memory()
            return True
        return False
    
    def update_fact_priority(self, fact_id: str, priority: int) -> bool:
        """Aktualisiert die Priorität eines Fakts."""
        if fact_id in self.pinned_facts:
            self.pinned_facts[fact_id]["priority"] = min(max(1, priority), 5)
            self.pinned_facts[fact_id]["timestamp"] = datetime.now().isoformat()
            self._save_memory()
            return True
        return False
    
    def get_relevant_facts(self, query: str = "", limit: int = 5, min_priority: int = 1) -> List[dict]:
        """Gibt die relevantesten Fakten zurück, optional gefiltert nach einer Abfrage."""
        # Filtere nach Mindestpriorität
        facts = [
            {"id": fid, **data} 
            for fid, data in self.pinned_facts.items()
            if data["priority"] >= min_priority
        ]
        
        # Sortiere nach Priorität und Aktualität
        facts.sort(
            key=lambda x: (x["priority"], x["timestamp"]),
            reverse=True
        )
        
        # Filtere nach Abfrage, falls vorhanden
        if query:
            query = query.lower()
            facts = [
                fact for fact in facts
                if query in fact["fact"].lower() or 
                   any(query in str(v).lower() for v in fact.get("metadata", {}).values())
            ]
        
        return facts[:limit]
    
    def search_facts(self, query: str, limit: int = 10) -> List[dict]:
        """Durchsucht Fakten nach einer Abfrage."""
        return self.get_relevant_facts(query, limit)


if __name__ == "__main__":
    # Beispielverwendung
    memory = LongTermMemory()
    
    # Fakten hinzufügen
    memory.add_fact("Erik arbeitet an einem KI-Projekt", priority=3)
    memory.add_fact("Lieblingsfarbe: Blau", priority=2, source="conversation")
    
    # Fakten abrufen
    print("Alle Fakten:")
    for fact in memory.get_relevant_facts():
        print(f"- {fact['fact']} (Priorität: {fact['priority']})")
    
    # Nach etwas Bestimmtem suchen
    print("\nSuche nach 'KI':")
    for fact in memory.search_facts("KI"):
        print(f"- {fact['fact']}")
