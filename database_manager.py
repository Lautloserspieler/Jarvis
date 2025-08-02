import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import logging
import threading

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Verwaltet die Datenbankoperationen für die Anwendung."""
    
    def __init__(self, db_path: str = "chat_history.db"):
        """Initialisiert den Datenbankmanager.
        
        Args:
            db_path: Pfad zur SQLite-Datenbankdatei
        """
        self.db_path = db_path
        self.conn = None
        self.lock = threading.Lock()  # Thread lock for thread safety
        self.init_database()
    
    def get_connection(self):
        """Stellt eine Verbindung zur Datenbank her, falls noch nicht geschehen."""
        with self.lock:
            if self.conn is None:
                # Aktiviere Thread-Sicherheit
                self.conn = sqlite3.connect(
                    self.db_path,
                    check_same_thread=False,  # Erlaubt die Verwendung in verschiedenen Threads
                    isolation_level=None     # Auto-commit Modus
                )
                self.conn.row_factory = sqlite3.Row
                # Aktiviere Fremdschlüssel-Unterstützung
                self.conn.execute("PRAGMA foreign_keys = ON")
            return self.conn
    
    def init_database(self):
        """Initialisiert die Datenbank mit den erforderlichen Tabellen."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Tabelle für Benutzer
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    display_name TEXT,
                    preferences TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_active TIMESTAMP
                )
            ''')
            
            # Tabelle für Konversationen
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    title TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
            ''')
            
            # Tabelle für Nachrichten
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    model_used TEXT,
                    tokens_used INTEGER,
                    generation_time REAL,
                    metadata TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
                )
            ''')
            
            # Indizes für häufig abgefragte Spalten
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id)')
            
            # Überprüfe und füge fehlende Spalten hinzu
            cursor.execute("PRAGMA table_info(messages)")
            columns = [column[1] for column in cursor.fetchall()]
            
            # Füge fehlende Spalten hinzu, falls nicht vorhanden
            if 'model_used' not in columns:
                cursor.execute('ALTER TABLE messages ADD COLUMN model_used TEXT')
            if 'tokens_used' not in columns:
                cursor.execute('ALTER TABLE messages ADD COLUMN tokens_used INTEGER')
            if 'generation_time' not in columns:
                cursor.execute('ALTER TABLE messages ADD COLUMN generation_time REAL')
            if 'metadata' not in columns:
                cursor.execute('ALTER TABLE messages ADD COLUMN metadata TEXT')
            
            conn.commit()
            
        except sqlite3.Error as e:
            logger.error(f"Fehler bei der Datenbankinitialisierung: {e}")
            # Versuche, die Verbindung wiederherzustellen
            if conn:
                conn.rollback()
            raise
    
    def create_conversation(self, user_id: int, title: str = "Neue Unterhaltung") -> int:
        """Erstellt eine neue Konversation.
        
        Args:
            user_id: Die ID des Benutzers
            title: Titel der Konversation
            
        Returns:
            Die ID der erstellten Konversation oder None bei Fehler
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO conversations (user_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                RETURNING id
            ''', (
                user_id,
                title,
                datetime.now().isoformat(),
                datetime.now().isoformat()
            ))
            
            conv_id = cursor.fetchone()[0]
            conn.commit()
            return conv_id
            
        except sqlite3.Error as e:
            logger.error(f"Fehler beim Erstellen der Konversation: {e}")
            if conn:
                conn.rollback()
            return None
    
    def save_conversation(self, user_id: int, messages: List[Dict]) -> int:
        """Speichert eine Konversation mit Nachrichten.
        
        Args:
            user_id: Die ID des Benutzers
            messages: Liste der Nachrichten als Dictionaries
            
        Returns:
            Die ID der gespeicherten Konversation oder None bei Fehler
        """
        if not messages:
            return None
            
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Titel aus der ersten Benutzer-Nachricht ableiten
            title = 'Neue Unterhaltung'
            if messages and messages[0].get('role') == 'user':
                title = messages[0].get('content', title)[:50]
            
            # Konversation erstellen
            cursor.execute('''
                INSERT INTO conversations (user_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                RETURNING id
            ''', (
                user_id,
                title,
                datetime.now().isoformat(),
                datetime.now().isoformat()
            ))
            
            conv_id = cursor.fetchone()[0]
            
            # Nachrichten einfügen
            for msg in messages:
                cursor.execute('''
                    INSERT INTO messages (
                        conversation_id, role, content, timestamp,
                        model_used, tokens_used, generation_time
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    conv_id,
                    msg.get('role'),
                    msg.get('content', ''),
                    msg.get('timestamp', datetime.now().isoformat()),
                    msg.get('model_used'),
                    msg.get('tokens_used'),
                    msg.get('generation_time')
                ))
            
            conn.commit()
            return conv_id
            
        except sqlite3.Error as e:
            logger.error(f"Fehler beim Speichern der Konversation: {e}")
            if conn:
                conn.rollback()
            return None
    
    def update_conversation_title(self, conversation_id: int, title: str) -> bool:
        """Aktualisiert den Titel einer Konversation.
        
        Args:
            conversation_id: Die ID der Konversation
            title: Der neue Titel
            
        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE conversations
                SET title = ?, updated_at = ?
                WHERE id = ?
            ''', (
                title,
                datetime.now().isoformat(),
                conversation_id
            ))
            
            conn.commit()
            return cursor.rowcount > 0
            
        except sqlite3.Error as e:
            logger.error(f"Fehler beim Aktualisieren des Konversationstitels: {e}")
            if conn:
                conn.rollback()
            return False
    
    def get_conversations(self, user_id: int, limit: int = 50) -> List[Dict]:
        """Holt die Konversationen eines Benutzers.
        
        Args:
            user_id: Die ID des Benutzers
            limit: Maximale Anzahl der zurückgegebenen Konversationen
            
        Returns:
            Eine Liste der Konversationen als Dictionaries
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT c.id, c.title, c.created_at, c.updated_at,
                       (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) as message_count
                FROM conversations c
                WHERE c.user_id = ?
                ORDER BY c.updated_at DESC
                LIMIT ?
            ''', (user_id, limit))
            
            conversations = []
            for row in cursor.fetchall():
                conversations.append(dict(row))
                
            return conversations
            
        except sqlite3.Error as e:
            logger.error(f"Fehler beim Abrufen der Konversationen: {e}")
            return []
    
    def delete_messages(self, conversation_id: int) -> bool:
        """Löscht alle Nachrichten einer Konversation.
        
        Args:
            conversation_id: Die ID der Konversation
            
        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Lösche alle Nachrichten der Konversation
            cursor.execute('''
                DELETE FROM messages
                WHERE conversation_id = ?
            ''', (conversation_id,))
            
            # Aktualisiere den updated_at-Zeitstempel der Konversation
            cursor.execute('''
                UPDATE conversations
                SET updated_at = ?
                WHERE id = ?
            ''', (datetime.now().isoformat(), conversation_id))
            
            conn.commit()
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Fehler beim Löschen der Nachrichten: {e}")
            if conn:
                conn.rollback()
            return False
    
    def delete_all_conversations(self, user_id: int) -> bool:
        """Löscht alle Konversationen eines Benutzers.
        
        Args:
            user_id: Die ID des Benutzers
            
        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Lösche alle Nachrichten der Konversationen des Benutzers
            cursor.execute('''
                DELETE FROM messages
                WHERE conversation_id IN (
                    SELECT id FROM conversations WHERE user_id = ?
                )
            ''', (user_id,))
            
            # Lösche alle Konversationen des Benutzers
            cursor.execute('''
                DELETE FROM conversations
                WHERE user_id = ?
            ''', (user_id,))
            
            conn.commit()
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Fehler beim Löschen der Konversationen: {e}")
            if conn:
                conn.rollback()
            return False
    
    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        model_used: str = None,
        tokens_used: int = None,
        generation_time: float = None,
        metadata: Dict = None,
        timestamp: str = None
    ) -> int:
        """Fügt eine Nachricht zu einer Konversation hinzu.
        
        Args:
            conversation_id: Die ID der Konversation
            role: Die Rolle des Absenders ('user' oder 'assistant')
            content: Der Inhalt der Nachricht
            model_used: Optional, das verwendete Modell
            tokens_used: Optional, Anzahl der verwendeten Tokens
            generation_time: Optional, Generierungszeit in Sekunden
            metadata: Optional, zusätzliche Metadaten als Dictionary
            timestamp: Optional, Zeitstempel der Nachricht (ISO-Format)
            
        Returns:
            Die ID der eingefügten Nachricht oder None bei Fehler
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
            
        if metadata is not None:
            try:
                metadata_json = json.dumps(metadata)
            except (TypeError, ValueError) as e:
                logger.warning(f"Konnte Metadaten nicht serialisieren: {e}")
                metadata_json = None
        else:
            metadata_json = None
            
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Füge die Nachricht ein (ohne RETURNING, da nicht von allen SQLite-Versionen unterstützt)
            cursor.execute('''
                INSERT INTO messages (
                    conversation_id, role, content, timestamp,
                    model_used, tokens_used, generation_time, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                conversation_id,
                role,
                content,
                timestamp,
                model_used,
                tokens_used,
                generation_time,
                metadata_json
            ))
            
            # Hole die ID der eingefügten Nachricht (kompatibler Weg)
            message_id = cursor.lastrowid
            
            # Aktualisiere den updated_at-Zeitstempel der Konversation
            cursor.execute('''
                UPDATE conversations
                SET updated_at = ?
                WHERE id = ?
            ''', (timestamp, conversation_id))
            
            conn.commit()
            return message_id
            
        except sqlite3.Error as e:
            logger.error(f"Fehler beim Hinzufügen der Nachricht: {e}")
            if conn:
                conn.rollback()
            return None
            
    def get_recent_messages(self, conversation_id: int, limit: int = 5) -> List[Dict]:
        """Holt die letzten Nachrichten einer Konversation.
        
        Args:
            conversation_id: Die ID der Konversation
            limit: Maximale Anzahl der zurückgegebenen Nachrichten
            
        Returns:
            Eine Liste der letzten Nachrichten als Dictionaries
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    id, 
                    conversation_id, 
                    role, 
                    content, 
                    timestamp, 
                    model_used, 
                    tokens_used,
                    generation_time
                FROM messages 
                WHERE conversation_id = ? 
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (int(conversation_id), int(limit)))
            
            messages = []
            for row in cursor.fetchall():
                message = {
                    'id': row[0],
                    'conversation_id': row[1],
                    'role': row[2],
                    'content': row[3],
                    'timestamp': row[4],
                    'model_used': row[5],
                    'tokens_used': row[6],
                    'generation_time': row[7]
                }
                messages.append(message)
                
            return messages
            
        except sqlite3.Error as e:
            logger.error(f"Fehler beim Abrufen der letzten Nachrichten: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Unerwarteter Fehler beim Abrufen der letzten Nachrichten: {e}", exc_info=True)
            return []
    
    def get_conversation(self, conversation_id: int) -> Optional[Dict[str, Any]]:
        """Holt eine einzelne Konversation anhand ihrer ID.
        
        Args:
            conversation_id: Die ID der Konversation
            
        Returns:
            Ein Dictionary mit den Konversationsdaten oder None, wenn nicht gefunden
        """
        with self.lock:
            try:
                conn = self.get_connection()
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT id, title, created_at, updated_at, user_id, model_used, 
                           system_prompt, temperature, max_tokens, top_p, 
                           frequency_penalty, presence_penalty, stop_sequences
                    FROM conversations 
                    WHERE id = ?
                ''', (conversation_id,))
                
                row = cursor.fetchone()
                if row:
                    return dict(row)
                return None
                
            except Exception as e:
                logger.error(f"Fehler beim Abrufen der Konversation {conversation_id}: {e}")
                return None
    
    def get_conversations(self, user_id: int = None, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Holt die Konversationen eines Benutzers mit Vorschau der ersten Nachricht.
        
        Args:
            user_id: Die ID des Benutzers
            limit: Maximale Anzahl der zurückgegebenen Konversationen
            offset: Anzahl der zu überspringenden Konversationen
            
        Returns:
            Eine Liste von Konversationen mit Metadaten
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Hole Konversationen mit Vorschau der ersten Nachricht
            cursor.execute('''
                SELECT 
                    c.id,
                    c.title,
                    c.created_at,
                    c.updated_at,
                    (SELECT content FROM messages WHERE conversation_id = c.id ORDER BY timestamp ASC LIMIT 1) as preview
                FROM conversations c
                WHERE c.user_id = ?
                ORDER BY c.updated_at DESC
                LIMIT ? OFFSET ?
            ''', (int(user_id), int(limit), int(offset)))
            
            conversations = []
            for row in cursor.fetchall():
                conversation = {
                    'id': row[0],
                    'title': row[1] or 'Neue Unterhaltung',
                    'created_at': row[2],
                    'updated_at': row[3],
                    'preview': (row[4] or '')[:100]  # Kürze die Vorschau
                }
                conversations.append(conversation)
                
            return conversations
            
        except sqlite3.Error as e:
            logger.error(f"Fehler beim Abrufen der Konversationen: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Unerwarteter Fehler beim Abrufen der Konversationen: {e}", exc_info=True)
            return []
    
    def get_messages(self, conversation_id: int, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Holt die Nachrichten einer bestimmten Konversation.
        
        Args:
            conversation_id: Die ID der Konversation
            limit: Maximale Anzahl der zurückzugebenden Nachrichten
            offset: Anzahl der zu überspringenden Nachrichten
            
        Returns:
            Eine Liste von Nachrichten als Dictionaries
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    id, 
                    conversation_id, 
                    role, 
                    content, 
                    timestamp, 
                    model_used, 
                    tokens_used,
                    generation_time
                FROM messages 
                WHERE conversation_id = ? 
                ORDER BY timestamp ASC
                LIMIT ? OFFSET ?
            ''', (int(conversation_id), int(limit), int(offset)))
            
            messages = []
            for row in cursor.fetchall():
                message = {
                    'id': row[0],
                    'conversation_id': row[1],
                    'role': row[2],
                    'content': row[3],
                    'timestamp': row[4],
                    'model_used': row[5],
                    'tokens_used': row[6],
                    'generation_time': row[7]
                }
                messages.append(message)
                
            return messages
            
        except sqlite3.Error as e:
            logger.error(f"Fehler beim Abrufen der Nachrichten: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Unerwarteter Fehler beim Abrufen der Nachrichten: {e}", exc_info=True)
            return []
    
    def delete_messages(self, conversation_id: int) -> bool:
        """Löscht alle Nachrichten einer bestimmten Konversation.
        
        Args:
            conversation_id: Die ID der Konversation
            
        Returns:
            True, wenn die Löschung erfolgreich war, sonst False
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM messages WHERE conversation_id = ?', (int(conversation_id),))
            
            # Aktualisiere den updated_at-Zeitstempel der Konversation
            cursor.execute('''
                UPDATE conversations 
                SET updated_at = ? 
                WHERE id = ?
            ''', (datetime.now().isoformat(), int(conversation_id)))
            
            conn.commit()
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Fehler beim Löschen der Nachrichten: {e}", exc_info=True)
            if conn:
                conn.rollback()
            return False
        except Exception as e:
            logger.error(f"Unerwarteter Fehler beim Löschen der Nachrichten: {e}", exc_info=True)
            if conn:
                conn.rollback()
            return False
    
    def delete_all_conversations(self, user_id: int) -> bool:
        """Löscht alle Konversationen und zugehörige Nachrichten eines Benutzers.
        
        Args:
            user_id: Die ID des Benutzers
            
        Returns:
            True, wenn die Löschung erfolgreich war, sonst False
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Lösche zuerst alle Nachrichten der Konversationen des Benutzers
            cursor.execute('''
                DELETE FROM messages 
                WHERE conversation_id IN (
                    SELECT id FROM conversations WHERE user_id = ?
                )
            ''', (int(user_id),))
            
            # Lösche dann alle Konversationen des Benutzers
            cursor.execute('DELETE FROM conversations WHERE user_id = ?', (int(user_id),))
            
            conn.commit()
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Fehler beim Löschen der Konversationen: {e}", exc_info=True)
            if conn:
                conn.rollback()
            return False
        except Exception as e:
            logger.error(f"Unerwarteter Fehler beim Löschen der Konversationen: {e}", exc_info=True)
            if conn:
                conn.rollback()
            return False
    
    def close(self):
        """Schließt die Datenbankverbindung."""
        if self.conn is not None:
            try:
                self.conn.close()
                self.conn = None
            except Exception as e:
                logger.error(f"Fehler beim Schließen der Datenbankverbindung: {e}", exc_info=True)

# Testen der Datenbankverbindung
if __name__ == "__main__":
    db = DatabaseManager("test_db.db")
    print("Datenbank erfolgreich initialisiert!")
