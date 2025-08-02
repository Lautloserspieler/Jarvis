import json
import os
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import hashlib

logger = logging.getLogger(__name__)

class LongTermMemory:
    """Verwaltet das Langzeitgedächtnis des KI-Assistenten."""
    
    def __init__(self, storage_file: str = "memory.json"):
        """Initialisiert das Langzeitgedächtnis.
        
        Args:
            storage_file: Pfad zur Speicherdatei für das Gedächtnis
        """
        self.storage_file = storage_file
        self.memories: Dict[str, Dict[str, Any]] = {}
        self._load_memories()
    
    def _load_memories(self) -> None:
        """Lädt gespeicherte Erinnerungen aus der Datei."""
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    self.memories = json.load(f)
                logger.info(f"{len(self.memories)} Erinnerungen aus {self.storage_file} geladen")
            else:
                self.memories = {}
                logger.info("Keine vorhandenen Erinnerungen gefunden, neues Gedächtnis erstellt")
        except Exception as e:
            logger.error(f"Fehler beim Laden der Erinnerungen: {e}")
            self.memories = {}
    
    def _save_memories(self) -> None:
        """Speichert die Erinnerungen in die Datei."""
        try:
            os.makedirs(os.path.dirname(self.storage_file) or '.', exist_ok=True)
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(self.memories, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Erinnerungen: {e}")
            raise RuntimeError("Konnte die Erinnerungen nicht speichern")
    
    def _generate_memory_id(self, content: str, context: str = "") -> str:
        """Generiert eine eindeutige ID für eine Erinnerung.
        
        Args:
            content: Der Inhalt der Erinnerung
            context: Zusätzlicher Kontext für die ID-Generierung
            
        Returns:
            Eine eindeutige ID als Hex-String
        """
        unique_str = f"{content}:{context}:{datetime.now().isoformat()}"
        return hashlib.sha256(unique_str.encode('utf-8')).hexdigest()
    
    def add_memory(
        self, 
        content: str, 
        memory_type: str = "fact", 
        importance: int = 5, 
        expiration_days: Optional[int] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Fügt eine neue Erinnerung hinzu.
        
        Args:
            content: Der Inhalt der Erinnerung
            memory_type: Typ der Erinnerung (z.B. 'fact', 'preference', 'event')
            importance: Wichtigkeit von 1 (niedrig) bis 10 (hoch)
            expiration_days: Anzahl der Tage, bis die Erinnerung verfällt (None für nie)
            tags: Liste von Tags zur Kategorisierung
            metadata: Zusätzliche Metadaten
            
        Returns:
            Die ID der hinzugefügten Erinnerung
        """
        memory_id = self._generate_memory_id(content)
        now = datetime.now().isoformat()
        
        if expiration_days is not None:
            expires = (datetime.now() + timedelta(days=expiration_days)).isoformat()
        else:
            expires = None
        
        self.memories[memory_id] = {
            "id": memory_id,
            "content": content,
            "type": memory_type,
            "importance": max(1, min(10, importance)),  # Auf 1-10 begrenzen
            "created_at": now,
            "last_accessed": now,
            "access_count": 0,
            "expires": expires,
            "tags": tags or [],
            "metadata": metadata or {}
        }
        
        self._save_memories()
        logger.debug(f"Neue Erinnerung hinzugefügt: {memory_id}")
        return memory_id
    
    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Holt eine spezifische Erinnerung anhand ihrer ID.
        
        Args:
            memory_id: Die ID der Erinnerung
            
        Returns:
            Die Erinnerung als Dictionary oder None, wenn nicht gefunden
        """
        if memory_id not in self.memories:
            return None
        
        memory = self.memories[memory_id]
        
        # Prüfe auf Ablaufdatum
        if memory.get("expires"):
            expires = datetime.fromisoformat(memory["expires"])
            if datetime.now() > expires:
                logger.debug(f"Erinnerung {memory_id} ist abgelaufen und wird entfernt")
                del self.memories[memory_id]
                self._save_memories()
                return None
        
        # Aktualisiere Zugriffsdaten
        memory["last_accessed"] = datetime.now().isoformat()
        memory["access_count"] = memory.get("access_count", 0) + 1
        self._save_memories()
        
        return memory
    
    def search_memories(
        self, 
        query: str = "", 
        memory_type: Optional[str] = None, 
        min_importance: int = 0,
        tags: Optional[List[str]] = None,
        limit: int = 10,
        include_expired: bool = False
    ) -> List[Dict[str, Any]]:
        """Durchsucht die Erinnerungen nach verschiedenen Kriterien.
        
        Args:
            query: Suchbegriff, der im Inhalt vorkommen soll
            memory_type: Filter nach Erinnerungstyp
            min_importance: Minimale Wichtigkeit (1-10)
            tags: Liste von Tags, die alle vorhanden sein müssen
            limit: Maximale Anzahl der Ergebnisse
            include_expired: Abgelaufene Erinnerungen einbeziehen
            
        Returns:
            Eine Liste von passenden Erinnerungen
        """
        results = []
        now = datetime.now()
        
        for memory in self.memories.values():
            # Prüfe auf Ablaufdatum
            if not include_expired and memory.get("expires"):
                expires = datetime.fromisoformat(memory["expires"])
                if now > expires:
                    continue
            
            # Filter nach Typ
            if memory_type and memory.get("type") != memory_type:
                continue
            
            # Filter nach Wichtigkeit
            if memory.get("importance", 0) < min_importance:
                continue
            
            # Filter nach Tags
            if tags and not all(tag in memory.get("tags", []) for tag in tags):
                continue
            
            # Suche im Inhalt
            if query.lower() not in memory["content"].lower():
                continue
            
            results.append(memory)
        
        # Sortiere nach Wichtigkeit und letztem Zugriff
        results.sort(
            key=lambda x: (
                -x.get("importance", 0),  # Höhere Wichtigkeit zuerst
                x.get("last_accessed", "")  # Neueste zuerst
            ),
            reverse=True
        )
        
        return results[:limit]
    
    def update_memory(
        self, 
        memory_id: str, 
        content: Optional[str] = None, 
        importance: Optional[int] = None,
        expires_in_days: Optional[int] = None,
        add_tags: Optional[List[str]] = None,
        remove_tags: Optional[List[str]] = None,
        metadata_updates: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Aktualisiert eine bestehende Erinnerung.
        
        Args:
            memory_id: Die ID der zu aktualisierenden Erinnerung
            content: Neuer Inhalt (optional)
            importance: Neue Wichtigkeit (optional)
            expires_in_days: Neue Gültigkeitsdauer in Tagen (None für unbegrenzt)
            add_tags: Tags, die hinzugefügt werden sollen
            remove_tags: Tags, die entfernt werden sollen
            metadata_updates: Metadaten-Aktualisierungen
            
        Returns:
            True, wenn die Aktualisierung erfolgreich war, sonst False
        """
        if memory_id not in self.memories:
            logger.warning(f"Konnte Erinnerung {memory_id} nicht aktualisieren: Nicht gefunden")
            return False
        
        memory = self.memories[memory_id]
        
        if content is not None:
            memory["content"] = content
        
        if importance is not None:
            memory["importance"] = max(1, min(10, importance))
        
        if expires_in_days is not None:
            if expires_in_days <= 0:
                memory["expires"] = None
            else:
                memory["expires"] = (datetime.now() + timedelta(days=expires_in_days)).isoformat()
        
        if add_tags:
            for tag in add_tags:
                if tag not in memory.get("tags", []):
                    if "tags" not in memory:
                        memory["tags"] = []
                    memory["tags"].append(tag)
        
        if remove_tags and "tags" in memory:
            memory["tags"] = [t for t in memory["tags"] if t not in remove_tags]
            if not memory["tags"]:  # Leere Tag-Listen entfernen
                del memory["tags"]
        
        if metadata_updates and isinstance(metadata_updates, dict):
            if "metadata" not in memory:
                memory["metadata"] = {}
            memory["metadata"].update(metadata_updates)
        
        memory["last_updated"] = datetime.now().isoformat()
        
        self._save_memories()
        logger.debug(f"Erinnerung {memory_id} wurde aktualisiert")
        return True
    
    def delete_memory(self, memory_id: str) -> bool:
        """Löscht eine Erinnerung.
        
        Args:
            memory_id: Die ID der zu löschenden Erinnerung
            
        Returns:
            True, wenn die Erinnerung gelöscht wurde, sonst False
        """
        if memory_id in self.memories:
            del self.memories[memory_id]
            self._save_memories()
            logger.debug(f"Erinnerung {memory_id} wurde gelöscht")
            return True
        return False
    
    def cleanup_expired(self) -> int:
        """Entfernt alle abgelaufenen Erinnerungen.
        
        Returns:
            Anzahl der entfernten Erinnerungen
        """
        initial_count = len(self.memories)
        now = datetime.now()
        
        self.memories = {
            k: v for k, v in self.memories.items()
            if not v.get("expires") or datetime.fromisoformat(v["expires"]) > now
        }
        
        removed_count = initial_count - len(self.memories)
        
        if removed_count > 0:
            self._save_memories()
            logger.info(f"{removed_count} abgelaufene Erinnerungen wurden gelöscht")
        
        return removed_count
    
    def get_statistics(self) -> Dict[str, Any]:
        """Gibt Statistiken über die gespeicherten Erinnerungen zurück.
        
        Returns:
            Ein Dictionary mit verschiedenen Statistiken
        """
        now = datetime.now()
        stats = {
            "total_memories": len(self.memories),
            "memory_types": {},
            "expired_memories": 0,
            "total_tags": 0,
            "most_common_tags": [],
            "importance_distribution": {i: 0 for i in range(1, 11)},
            "oldest_memory": None,
            "newest_memory": None,
            "total_accesses": 0
        }
        
        if not self.memories:
            return stats
        
        tag_counts = {}
        oldest = now
        newest = datetime.min.replace(tzinfo=now.tzinfo)
        
        for memory in self.memories.values():
            # Zähle nach Typ
            mem_type = memory.get("type", "unknown")
            stats["memory_types"][mem_type] = stats["memory_types"].get(mem_type, 0) + 1
            
            # Prüfe auf abgelaufen
            if memory.get("expires"):
                expires = datetime.fromisoformat(memory["expires"])
                if now > expires:
                    stats["expired_memories"] += 1
            
            # Zähle Tags
            for tag in memory.get("tags", []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            # Zähle Wichtigkeitsverteilung
            importance = memory.get("importance", 5)
            stats["importance_distribution"][importance] += 1
            
            # Finde älteste/neueste Erinnerung
            created = datetime.fromisoformat(memory["created_at"])
            if created < oldest:
                oldest = created
                stats["oldest_memory"] = memory["created_at"]
            if created > newest:
                newest = created
                stats["newest_memory"] = memory["created_at"]
            
            # Zähle Zugriffe
            stats["total_accesses"] += memory.get("access_count", 0)
        
        # Berechne Tag-Statistiken
        stats["total_tags"] = len(tag_counts)
        if tag_counts:
            sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
            stats["most_common_tags"] = sorted_tags[:10]  # Top 10 Tags
        
        return stats
