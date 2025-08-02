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
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
        return self.conn
    
    def init_database(self) -> None:
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
                    user_id INTEGER,
                    title TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # Tabelle für Nachrichten
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    model_used TEXT,
                    tokens_used INTEGER,
                    generation_time REAL,
                    metadata TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (id)
                )
            ''')
            
            # Indizes für bessere Performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)')
            
            conn.commit()
            logger.info("Datenbank erfolgreich initialisiert")
            
        except sqlite3.Error as e:
            logger.error(f"Fehler bei der Datenbankinitialisierung: {e}")
            raise
    
    def create_user(self, username: str, display_name: Optional[str] = None, preferences: Optional[Dict] = None) -> int:
        """Erstellt einen neuen Benutzer.
        
        Args:
            username: Der eindeutige Benutzername
            display_name: Der Anzeigename des Benutzers
            preferences: Benutzereinstellungen als Dictionary
            
        Returns:
            Die ID des erstellten Benutzers
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            preferences_json = json.dumps(preferences) if preferences else '{}'
            
            cursor.execute('''
                INSERT INTO users (username, display_name, preferences, last_active)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ''', (username, display_name or username, preferences_json))
            
            user_id = cursor.lastrowid
            conn.commit()
            return user_id
            
        except sqlite3.IntegrityError:
            logger.warning(f"Benutzer '{username}' existiert bereits")
            raise ValueError(f"Benutzername '{username}' ist bereits vergeben")
        except sqlite3.Error as e:
            logger.error(f"Fehler beim Erstellen des Benutzers: {e}")
            raise
    
    def get_user(self, user_id: int = None, username: str = None) -> Optional[Dict]:
        """Holt einen Benutzer anhand der ID oder des Benutzernamens.
        
        Args:
            user_id: Die ID des Benutzers
            username: Der Benutzername
            
        Returns:
            Ein Dictionary mit den Benutzerdaten oder None, wenn nicht gefunden
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            if user_id is not None:
                cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
            elif username is not None:
                cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
            else:
                return None
                
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
            
        except sqlite3.Error as e:
            logger.error(f"Fehler beim Abrufen des Benutzers: {e}")
            return None
    
    def update_user_last_active(self, user_id: int) -> None:
        """Aktualisiert den Zeitstempel der letzten Aktivität eines Benutzers.
        
        Args:
            user_id: Die ID des Benutzers
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE users 
                SET last_active = CURRENT_TIMESTAMP 
                WHERE id = ?
            ''', (user_id,))
            
            conn.commit()
            
        except sqlite3.Error as e:
            logger.error(f"Fehler beim Aktualisieren des letzten Aktivitätszeitpunkts: {e}")
    
    def create_conversation(self, user_id: int, title: str = None) -> int:
        """Erstellt eine neue Konversation.
        
        Args:
            user_id: Die ID des Benutzers
            title: Der Titel der Konversation
            
        Returns:
            Die ID der erstellten Konversation
        """
        with self.lock:
            try:
                conn = self.get_connection()
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO conversations (user_id, title, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    RETURNING id
                ''', (
                    int(user_id),
                    title,
                    datetime.now().isoformat(),
                    datetime.now().isoformat()
                ))
                
                conv_id = cursor.fetchone()[0]
                conn.commit()
                return int(conv_id)
                
            except sqlite3.Error as e:
                logger.error(f"Fehler beim Erstellen der Konversation: {e}", exc_info=True)
                if conn:
                    conn.rollback()
                raise
    
    def get_conversation(self, conversation_id: int) -> Optional[Dict]:
        """Holt eine Konversation anhand der ID.
        
        Args:
            conversation_id: Die ID der Konversation
            
        Returns:
            Ein Dictionary mit den Konversationsdaten oder None, wenn nicht gefunden
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM conversations WHERE id = ?', (conversation_id,))
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return None
            
        except sqlite3.Error as e:
            logger.error(f"Fehler beim Abrufen der Konversation {conversation_id}: {e}")
            return None
    
    def get_user_conversations(self, user_id: int, limit: int = 50, offset: int = 0) -> List[Dict]:
        """Holt die Konversationen eines Benutzers.
        
        Args:
            user_id: Die ID des Benutzers
            limit: Maximale Anzahl der zurückgegebenen Konversationen
            offset: Anzahl der zu überspringenden Konversationen
            
        Returns:
            Eine Liste von Konversationen als Dictionaries
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT c.*, 
                       (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) as message_count,
                       (SELECT m.content FROM messages m 
                        WHERE m.conversation_id = c.id 
                        ORDER BY m.timestamp DESC 
                        LIMIT 1) as last_message_preview
                FROM conversations c
                WHERE c.user_id = ?
                ORDER BY c.updated_at DESC
                LIMIT ? OFFSET ?
            ''', (user_id, limit, offset))
            
            return [dict(row) for row in cursor.fetchall()]
            
        except sqlite3.Error as e:
            logger.error(f"Fehler beim Abrufen der Konversationen für Benutzer {user_id}: {e}")
            return []
    
    def update_conversation_title(self, conversation_id: int, title: str) -> bool:
        """Aktualisiert den Titel einer Konversation.
        
        Args:
            conversation_id: Die ID der Konversation
            title: Der neue Titel der Konversation
            
        Returns:
            True, wenn die Aktualisierung erfolgreich war, sonst False
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE conversations 
                SET title = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (title, conversation_id))
            
            conn.commit()
            return cursor.rowcount > 0
            
        except sqlite3.Error as e:
            logger.error(f"Fehler beim Aktualisieren des Konversationstitels {conversation_id}: {e}")
            return False
            
    def update_conversation(self, conversation_id: int, title: str = None, metadata: Dict = None) -> bool:
        """Aktualisiert eine Konversation.
        
        Args:
            conversation_id: Die ID der Konversation
            title: Der neue Titel der Konversation
            metadata: Aktualisierte Metadaten
            
        Returns:
            True, wenn die Aktualisierung erfolgreich war, sonst False
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            updates = []
            params = []
            
            if title is not None:
                updates.append("title = ?")
                params.append(title)
            
            if metadata is not None:
                updates.append("metadata = ?")
                params.append(json.dumps(metadata))
            
            if not updates:
                return False
                
            # Aktualisiere updated_at
            updates.append("updated_at = CURRENT_TIMESTAMP")
            
            query = f"""
                UPDATE conversations 
                SET {', '.join(updates)}
                WHERE id = ?
            """
            
            params.append(conversation_id)
            cursor.execute(query, tuple(params))
            conn.commit()
            
            return cursor.rowcount > 0
            
        except sqlite3.Error as e:
            logger.error(f"Fehler beim Aktualisieren der Konversation {conversation_id}: {e}")
            return False
    
    def delete_conversation(self, conversation_id: int) -> bool:
        """Löscht eine Konversation und alle zugehörigen Nachrichten.
        
        Args:
            conversation_id: Die ID der zu löschenden Konversation
            
        Returns:
            True, wenn die Löschung erfolgreich war, sonst False
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Lösche zuerst alle Nachrichten der Konversation
            cursor.execute('DELETE FROM messages WHERE conversation_id = ?', (conversation_id,))
            
            # Lösche die Konversation
            cursor.execute('DELETE FROM conversations WHERE id = ?', (conversation_id,))
            
            conn.commit()
            return cursor.rowcount > 0
            
        except sqlite3.Error as e:
            logger.error(f"Fehler beim Löschen der Konversation {conversation_id}: {e}")
            return False
    
    def add_message(self, conversation_id: int, role: str, content: str, 
                   model_used: str = None, tokens_used: int = None, 
                   generation_time: float = None, metadata: Dict = None,
                   timestamp: str = None) -> int:
        """Fügt eine Nachricht zu einer Konversation hinzu.
        
        Args:
            conversation_id: Die ID der Konversation
            role: Die Rolle des Absenders (z.B. 'user', 'assistant')
            content: Der Inhalt der Nachricht
            model_used: Das verwendete Modell (optional)
            tokens_used: Anzahl der verwendeten Tokens (optional)
            generation_time: Generierungszeit in Sekunden (optional)
            metadata: Zusätzliche Metadaten (optional)
            timestamp: Zeitstempel der Nachricht (optional, Standard: aktuelle Zeit)
            
        Returns:
            Die ID der eingefügten Nachricht
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            metadata_json = json.dumps(metadata) if metadata else '{}'
            
            # Wenn kein Zeitstempel angegeben wurde, verwende die aktuelle Zeit
            if timestamp is None:
                timestamp = datetime.now().isoformat()
            
            cursor.execute('''
                INSERT INTO messages (
                    conversation_id, 
                    role, 
                    content, 
                    timestamp, 
                    model_used, 
                    tokens_used,
                    generation_time,
                    metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                int(conversation_id),
                str(role),
                str(content),
                timestamp,
                str(model_used) if model_used else None,
                int(tokens_used) if tokens_used is not None else None,
                float(generation_time) if generation_time is not None else None,
                metadata_json
            ))
            
            # Aktualisiere den updated_at-Zeitstempel der Konversation
            cursor.execute('''
                UPDATE conversations 
                SET updated_at = ? 
                WHERE id = ?
            ''', (timestamp, int(conversation_id)))
            
            conn.commit()
            return cursor.lastrowid
            
        except sqlite3.Error as e:
            logger.error(f"Fehler beim Hinzufügen der Nachricht: {e}", exc_info=True)
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM messages 
                WHERE conversation_id = ? 
                ORDER BY timestamp ASC
                LIMIT ? OFFSET ?
            ''', (conversation_id, limit, offset))
            
            return [dict(row) for row in cursor.fetchall()]
            
        except sqlite3.Error as e:
            logger.error(f"Fehler beim Abrufen der Nachrichten für Konversation {conversation_id}: {e}")
            return []
    
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
    
    def search_messages(self, user_id: int, query: str, limit: int = 50) -> List[Dict]:
        """Durchsucht Nachrichten nach einem bestimmten Text.
        
        Args:
            user_id: Die ID des Benutzers
            query: Der Suchbegriff
            limit: Maximale Anzahl der Suchergebnisse
            
        Returns:
            Eine Liste von gefundenen Nachrichten mit Kontextinformationen
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            search_term = f"%{query}%"
            
            cursor.execute('''
                SELECT m.*, c.title as conversation_title
                FROM messages m
                JOIN conversations c ON m.conversation_id = c.id
                WHERE c.user_id = ? AND m.content LIKE ?
                ORDER BY m.timestamp DESC
                LIMIT ?
            ''', (user_id, search_term, limit))
            
            return [dict(row) for row in cursor.fetchall()]
            
        except sqlite3.Error as e:
            logger.error(f"Fehler bei der Nachrichtensuche: {e}")
            return []
    
    def save_conversation(self, user_id: int, messages: List[Dict]):
        """Speichert eine Konversation in der Datenbank.
        
        Args:
            user_id: Die ID des Benutzers
            messages: Liste von Nachrichten im Format [
                {
                    'role': 'user'|'assistant', 
                    'content': str, 
                    'timestamp': str, 
                    'model_used': str, 
                    'tokens_used': int,
                    'generation_time': float
                }
            ]
            
        Returns:
            Die ID der gespeicherten Konversation oder None bei Fehler
        """
        if not messages:
            logger.warning("Keine Nachrichten zum Speichern vorhanden")
            return None
            
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Erstelle eine neue Konversation
            title = 'Neue Unterhaltung'
            if messages and messages[0].get('role') == 'user':
                title = messages[0].get('content', title)[:50]  # Kürze den Titel
            
            cursor.execute('''
                INSERT INTO conversations (user_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                RETURNING id
            ''', (
                int(user_id),
                title,
                datetime.now().isoformat(),
                datetime.now().isoformat()
            ))
            
            conv_id = cursor.fetchone()[0]
            
            # Füge die Nachrichten hinzu
            for msg in messages:
                cursor.execute('''
                    INSERT INTO messages (
                        conversation_id, 
                        role, 
                        content, 
                        timestamp, 
                        model_used, 
                        tokens_used,
                        generation_time
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
            logger.error(f"Fehler beim Speichern der Unterhaltung: {e}", exc_info=True)
            if conn:
                conn.rollback()
            return None
        except Exception as e:
            logger.error(f"Unerwarteter Fehler beim Speichern der Unterhaltung: {e}", exc_info=True)
            if conn:
                conn.rollback()
            return None
            
    def delete_messages(self, conversation_id: int) -> bool:
        """Löscht alle Nachrichten einer Konversation.
        
        Args:
            conversation_id: Die ID der Konversation, deren Nachrichten gelöscht werden sollen
            
        Returns:
        
        return [dict(row) for row in cursor.fetchall()]
            limit: Maximale Anzahl der zurückgegebenen Konversationen
            
        Returns:
            Eine Liste von Konversationen mit Metadaten und Vorschau der ersten Nachricht
        """
        with self.lock:  # Thread-Sicherheit mit Lock
            try:
                conn = self.get_connection()
                cursor = conn.cursor()
                
                # SQL-Abfrage, um Konversationen mit einer Vorschau der ersten Nachricht abzurufen
                cursor.execute('''
                    SELECT 
                        c.id, 
                        c.title, 
                        c.updated_at,
                        (SELECT content FROM messages 
                         WHERE conversation_id = c.id 
                         ORDER BY timestamp ASC LIMIT 1) as preview,
                        (SELECT COUNT(*) FROM messages WHERE conversation_id = c.id) as message_count
                    FROM conversations c
                    WHERE c.user_id = ?
                    ORDER BY c.updated_at DESC
                    LIMIT ?
                ''', (int(user_id), int(limit)))
                
                conversations = []
                for row in cursor.fetchall():
                    try:
                        preview = str(row['preview']) if row['preview'] is not None else ''
                        if len(preview) > 100:
                            preview = preview[:100] + '...'
                        
                        conversations.append({
                            'id': int(row['id']),
                            'title': str(row['title']) if row['title'] else 'Neue Unterhaltung',
                            'timestamp': row['updated_at'],  # Verwende updated_at als timestamp für die Anzeige
                            'message_count': int(row['message_count']) if row['message_count'] is not None else 0,
                            'preview': preview
                        })
                    except Exception as row_error:
                        logger.error(f"Fehler beim Verarbeiten der Konversationsdaten (ID: {row.get('id', 'unbekannt')}): {row_error}", 
                                   exc_info=True)
                        continue  # Überspringe fehlerhafte Einträge
                
                return conversations
                
            except sqlite3.Error as e:
                logger.error(f"Datenbankfehler beim Abrufen der Konversationen für Benutzer {user_id}: {e}", 
                           exc_info=True)
                return []
            except Exception as e:
                logger.error(f"Unerwarteter Fehler beim Abrufen der Konversationen: {e}", 
                           exc_info=True)
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
            logger.error(f"Fehler beim Abrufen der Nachrichten für Konversation {conversation_id}: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Unerwarteter Fehler beim Abrufen der Nachrichten: {e}", exc_info=True)
            return []
    
    def get_conversations(self, user_id: int, limit: int = 50) -> List[Dict]:
        """Holt die Konversationen eines Benutzers mit Vorschau der ersten Nachricht.
        
        Args:
            user_id: Die ID des Benutzers
            limit: Maximale Anzahl der zurückzugebenden Konversationen
            
        Returns:
            Eine Liste von Konversationen mit Metadaten
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    c.id, 
                    c.title, 
                    c.created_at,
                    c.updated_at,
                    (SELECT content FROM messages 
                     WHERE conversation_id = c.id 
                     ORDER BY timestamp ASC 
                     LIMIT 1) as preview
                FROM conversations c
                WHERE c.user_id = ?
                ORDER BY c.updated_at DESC
                LIMIT ?
            ''', (int(user_id), int(limit)))
            
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
            logger.error(f"Fehler beim Abrufen der Konversationen: {e}")
            return []
        except Exception as e:
            logger.error(f"Unerwarteter Fehler beim Abrufen der Konversationen: {e}")
            return []
    
    def close(self) -> None:
        """Schließt die Datenbankverbindung."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None
            
    def delete_all_conversations(self, user_id: int) -> bool:
        """Löscht alle Konversationen eines Benutzers.
        
        Args:
            user_id: Die ID des Benutzers, dessen Konversationen gelöscht werden sollen
            
        Returns:
            True, wenn das Löschen erfolgreich war, sonst False
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Lösche zuerst alle Nachrichten des Benutzers
            cursor.execute('''
                DELETE FROM messages 
                WHERE conversation_id IN (
                    SELECT id FROM conversations WHERE user_id = ?
                )
            ''', (int(user_id),))
            
            # Lösche alle Konversationen des Benutzers
            cursor.execute('''
                DELETE FROM conversations 
                WHERE user_id = ?
            ''', (int(user_id),))
            
            conn.commit()
            logger.info(f"Alle Konversationen des Benutzers {user_id} wurden gelöscht")
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
            
    def __del__(self):
        """Stellt sicher, dass die Datenbankverbindung beim Zerstören des Objekts geschlossen wird."""
        self.close()
