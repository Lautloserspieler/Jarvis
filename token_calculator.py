import re
import logging
from typing import List, Dict, Any, Union, Optional

logger = logging.getLogger(__name__)

class TokenCalculator:
    """Berechnet die ungefähre Anzahl der Tokens für einen gegebenen Text.
    
    Diese Klasse bietet Methoden zur Schätzung der Token-Anzahl für verschiedene
    Tokenizer, einschließlich der von OpenAI und anderer gängiger Modelle.
    """
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        """Initialisiert den TokenCalculator.
        
        Args:
            encoding_name: Der Name der zu verwendenden Kodierung (z.B. "cl100k_base")
        """
        self.encoding_name = encoding_name
        self._initialize_encoding()
    
    def _initialize_encoding(self) -> None:
        """Initialisiert die Kodierung für die Token-Berechnung."""
        try:
            import tiktoken
            self.encoding = tiktoken.get_encoding(self.encoding_name)
            self._encode_method = self._encode_with_tiktoken
            logger.info(f"Verwende tiktoken mit Kodierung: {self.encoding_name}")
        except ImportError:
            self.encoding = None
            self._encode_method = self._estimate_with_heuristics
            logger.warning(
                "tiktoken nicht gefunden, verwende heuristische Schätzung. "
                "Installieren Sie tiktoken für genauere Ergebnisse: pip install tiktoken"
            )
    
    def _encode_with_tiktoken(self, text: str) -> List[int]:
        """Kodiert Text mit tiktoken.
        
        Args:
            text: Der zu kodierende Text
            
        Returns:
            Eine Liste von Token-IDs
        """
        return self.encoding.encode(text)
    
    def _estimate_with_heuristics(self, text: str) -> List[str]:
        """Schätzt die Token mit einer Heuristik (wenn tiktoken nicht verfügbar ist).
        
        Args:
            text: Der zu analysierende Text
            
        Returns:
            Eine Liste von geschätzten Token-Strings
        """
        # Einfache Heuristik: Teile nach Leerzeichen und Satzzeichen
        if not text:
            return []
        
        # Ersetze Satzzeichen durch Leerzeichen + Satzzeichen
        text = re.sub(r'([.,!?()])', r' \1 ', text)
        # Entferne doppelte Leerzeichen
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text.split() if text else []
    
    def count_tokens(self, text: str) -> int:
        """Zählt die ungefähre Anzahl der Tokens im Text.
        
        Args:
            text: Der zu analysierende Text
            
        Returns:
            Die geschätzte Anzahl der Tokens
        """
        if not text:
            return 0
            
        tokens = self._encode_method(text)
        return len(tokens)
    
    def count_tokens_for_messages(self, messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo") -> int:
        """Zählt die Tokens für eine Liste von Nachrichten im Chat-Format.
        
        Args:
            messages: Eine Liste von Nachrichten im Format [{"role": "user", "content": "..."}, ...]
            model: Der Name des Modells (für modellspezifische Tokenisierung)
            
        Returns:
            Die geschätzte Gesamtzahl der Tokens
        """
        try:
            import tiktoken
        except ImportError:
            # Fallback: Einfache Schätzung ohne tiktoken
            total = 0
            for msg in messages:
                content = msg.get("content", "")
                total += self.count_tokens(content) + 4  # +4 für Rollen-Tags
            return total
        
        # Verwende tiktoken für genauere Berechnung
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        
        if model in {"gpt-3.5-turbo", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-16k-0613",
                    "gpt-4", "gpt-4-0314", "gpt-4-32k", "gpt-4-32k-0314", "gpt-4-0613", "gpt-4-32k-0613"}:
            tokens_per_message = 3
            tokens_per_name = 1
        elif model == "gpt-3.5-turbo-0301":
            tokens_per_message = 4  # Jede Nachricht folgt auf <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # Wenn der Name fehlt, ist der Inhalt leer
        else:
            # Für unbekannte Modelle eine Schätzung vornehmen
            tokens_per_message = 3
            tokens_per_name = 1
        
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        
        num_tokens += 3  # Jede Antwort beginnt mit <|start|>assistant<|message|>
        return num_tokens
    
    def estimate_max_tokens(self, text: str, max_context_length: int, safety_margin: float = 0.1) -> int:
        """Schätzt die maximale Anzahl der Tokens, die für eine Antwort verbleiben.
        
        Args:
            text: Der Eingabetext
            max_context_length: Die maximale Kontextlänge des Modells
            safety_margin: Sicherheitsabstand (z.B. 0.1 für 10%)
            
        Returns:
            Die geschätzte maximale Anzahl der Antwort-Tokens
        """
        input_tokens = self.count_tokens(text)
        safety_tokens = int(max_context_length * safety_margin)
        max_tokens = max_context_length - input_tokens - safety_tokens
        
        # Stelle sicher, dass wir nicht negativ werden
        return max(0, max_tokens)
    
    def truncate_to_max_tokens(self, text: str, max_tokens: int, from_end: bool = False) -> str:
        """Kürzt einen Text auf eine maximale Anzahl von Tokens.
        
        Args:
            text: Der zu kürzende Text
            max_tokens: Die maximale Anzahl der Tokens
            from_end: Wenn True, wird vom Ende her gekürzt
            
        Returns:
            Der gekürzte Text
        """
        if not text or max_tokens <= 0:
            return ""
        
        tokens = self._encode_method(text)
        
        if len(tokens) <= max_tokens:
            return text
        
        if from_end:
            tokens = tokens[-max_tokens:]
        else:
            tokens = tokens[:max_tokens]
        
        # Konvertiere die Token-IDs zurück in Text
        if hasattr(self, 'encoding') and hasattr(self.encoding, 'decode'):
            return self.encoding.decode(tokens)
        else:
            # Fallback: Verwende die Heuristik
            return ' '.join(tokens[:max_tokens])  # Vereinfachte Rückgabe für Heuristik
    
    def get_token_usage_stats(self, prompt: str, completion: str, model: str = None) -> Dict[str, Any]:
        """Gibt detaillierte Token-Statistiken zurück.
        
        Args:
            prompt: Der Eingabe-Prompt
            completion: Die generierte Antwort
            model: Optional: Der Name des Modells für genauere Berechnungen
            
        Returns:
            Ein Dictionary mit Token-Statistiken
        """
        prompt_tokens = self.count_tokens(prompt)
        completion_tokens = self.count_tokens(completion)
        total_tokens = prompt_tokens + completion_tokens
        
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "prompt_length": len(prompt),
            "completion_length": len(completion),
            "chars_per_token_prompt": len(prompt) / prompt_tokens if prompt_tokens > 0 else 0,
            "chars_per_token_completion": len(completion) / completion_tokens if completion_tokens > 0 else 0,
            "model": model or "unknown"
        }
    
    def is_within_token_limit(self, text: str, max_tokens: int) -> bool:
        """Überprüft, ob ein Text innerhalb des Token-Limits liegt.
        
        Args:
            text: Der zu überprüfende Text
            max_tokens: Das maximale Token-Limit
            
        Returns:
            True, wenn der Text innerhalb des Limits liegt, sonst False
        """
        return self.count_tokens(text) <= max_tokens
    
    def split_into_chunks(self, text: str, max_tokens: int, overlap: int = 0) -> List[str]:
        """Teilt einen Text in Chunks mit maximaler Token-Größe auf.
        
        Args:
            text: Der aufzuteilende Text
            max_tokens: Maximale Anzahl der Tokens pro Chunk
            overlap: Anzahl der überlappenden Tokens zwischen benachbarten Chunks
            
        Returns:
            Eine Liste von Text-Chunks
        """
        if max_tokens <= 0:
            return [text]
        
        tokens = self._encode_method(text)
        chunks = []
        
        for i in range(0, len(tokens), max_tokens - overlap):
            chunk_tokens = tokens[i:i + max_tokens]
            
            # Konvertiere die Token-IDs zurück in Text
            if hasattr(self, 'encoding') and hasattr(self.encoding, 'decode'):
                chunk = self.encoding.decode(chunk_tokens)
            else:
                # Fallback: Verwende die Heuristik
                chunk = ' '.join(chunk_tokens)
            
            chunks.append(chunk)
        
        return chunks
