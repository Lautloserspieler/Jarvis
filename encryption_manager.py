import os
import base64
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)

class EncryptionManager:
    """Verwaltet die Verschlüsselung für sensible Daten."""
    
    def __init__(self, key_file: str = "encryption.key", password: str = None):
        """Initialisiert den EncryptionManager.
        
        Args:
            key_file: Pfad zur Schlüsseldatei
            password: Optional: Passwort zur Schlüsselgenerierung
        """
        self.key_file = key_file
        self.password = password
        self.key = self._load_or_create_key()
        self.cipher = Fernet(self.key)
    
    def _load_or_create_key(self) -> bytes:
        """Lädt einen vorhandenen Schlüssel oder erstellt einen neuen.
        
        Returns:
            Der Verschlüsselungsschlüssel als Bytes
        """
        if os.path.exists(self.key_file):
            try:
                with open(self.key_file, 'rb') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Fehler beim Lesen des Schlüssels: {e}")
                raise RuntimeError("Konnte den Verschlüsselungsschlüssel nicht laden")
        else:
            # Erstelle einen neuen Schlüssel
            if self.password:
                # Generiere einen Schlüssel aus dem Passwort
                salt = os.urandom(16)
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(self.password.encode()))
            else:
                # Generiere einen zufälligen Schlüssel
                key = Fernet.generate_key()
            
            # Speichere den Schlüssel
            try:
                os.makedirs(os.path.dirname(self.key_file) or '.', exist_ok=True)
                with open(self.key_file, 'wb') as f:
                    f.write(key)
                os.chmod(self.key_file, 0o600)  # Sichere Berechtigungen
                return key
            except Exception as e:
                logger.error(f"Fehler beim Speichern des Schlüssels: {e}")
                raise RuntimeError("Konnte den Verschlüsselungsschlüssel nicht speichern")
    
    def encrypt(self, data: str) -> str:
        """Verschlüsselt einen String.
        
        Args:
            data: Der zu verschlüsselnde String
            
        Returns:
            Der verschlüsselte String als Base64
        """
        if not data:
            return ""
        
        try:
            encrypted = self.cipher.encrypt(data.encode('utf-8'))
            return base64.b64encode(encrypted).decode('utf-8')
        except Exception as e:
            logger.error(f"Verschlüsselungsfehler: {e}")
            raise RuntimeError("Fehler bei der Verschlüsselung der Daten")
    
    def decrypt(self, encrypted_data: str) -> str:
        """Entschlüsselt einen String.
        
        Args:
            encrypted_data: Der verschlüsselte String als Base64
            
        Returns:
            Der entschlüsselte String
        """
        if not encrypted_data:
            return ""
        
        try:
            encrypted = base64.b64decode(encrypted_data)
            return self.cipher.decrypt(encrypted).decode('utf-8')
        except Exception as e:
            logger.error(f"Entschlüsselungsfehler: {e}")
            raise RuntimeError("Fehler bei der Entschlüsselung der Daten")
    
    def encrypt_file(self, input_file: str, output_file: str = None) -> str:
        """Verschlüsselt eine Datei.
        
        Args:
            input_file: Pfad zur Eingabedatei
            output_file: Pfad zur Ausgabedatei (optional)
            
        Returns:
            Der Pfad zur verschlüsselten Datei
        """
        if not output_file:
            output_file = input_file + '.enc'
        
        try:
            with open(input_file, 'rb') as f:
                data = f.read()
            
            encrypted = self.cipher.encrypt(data)
            
            with open(output_file, 'wb') as f:
                f.write(encrypted)
            
            return output_file
        except Exception as e:
            logger.error(f"Fehler beim Verschlüsseln der Datei {input_file}: {e}")
            raise RuntimeError(f"Konnte die Datei nicht verschlüsseln: {e}")
    
    def decrypt_file(self, input_file: str, output_file: str = None) -> str:
        """Entschlüsselt eine Datei.
        
        Args:
            input_file: Pfad zur verschlüsselten Datei
            output_file: Pfad zur Ausgabedatei (optional)
            
        Returns:
            Der Pfad zur entschlüsselten Datei
        """
        if not output_file:
            if input_file.endswith('.enc'):
                output_file = input_file[:-4]
            else:
                output_file = input_file + '.dec'
        
        try:
            with open(input_file, 'rb') as f:
                encrypted = f.read()
            
            decrypted = self.cipher.decrypt(encrypted)
            
            with open(output_file, 'wb') as f:
                f.write(decrypted)
            
            return output_file
        except Exception as e:
            logger.error(f"Fehler beim Entschlüsseln der Datei {input_file}: {e}")
            raise RuntimeError(f"Konnte die Datei nicht entschlüsseln: {e}")
    
    def rotate_key(self, new_key_file: str = None, new_password: str = None) -> None:
        """Rotiert den Verschlüsselungsschlüssel.
        
        Args:
            new_key_file: Pfad zur neuen Schlüsseldatei (optional)
            new_password: Neues Passwort für die Schlüsselgenerierung (optional)
        """
        old_key = self.key
        
        # Speichere die alten Schlüsselparameter
        old_key_file = self.key_file
        old_password = self.password
        
        try:
            # Setze die neuen Parameter
            if new_key_file:
                self.key_file = new_key_file
            
            if new_password:
                self.password = new_password
            
            # Generiere einen neuen Schlüssel
            self.key = self._load_or_create_key()
            self.cipher = Fernet(self.key)
            
            logger.info("Verschlüsselungsschlüssel wurde erfolgreich rotiert")
            
        except Exception as e:
            # Bei Fehlern die alten Parameter wiederherstellen
            self.key = old_key
            self.key_file = old_key_file
            self.password = old_password
            self.cipher = Fernet(self.key)
            
            logger.error(f"Fehler beim Rotieren des Schlüssels: {e}")
            raise RuntimeError("Konnte den Verschlüsselungsschlüssel nicht rotieren")
