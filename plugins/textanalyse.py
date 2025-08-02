import re
from typing import Dict, List, Tuple
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# Lade erforderliche NLTK-Ressourcen
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class Plugin:
    def __init__(self):
        self.name = "Textanalyse"
        self.description = "Analysiert Texte auf verschiedene Aspekte wie Wortanzahl, Lesbarkeit, Stimmung, etc."
        self.stop_words = set(stopwords.words('german') + stopwords.words('english'))
    
    def _display_response(self, response: str):
        """Zeigt die Antwort im Chat an"""
        if not response:
            response = "Entschuldigung, ich konnte keine Analyse durchführen."
        return response
    
    def _clean_text(self, text: str) -> str:
        """Bereinigt den Text für die Analyse"""
        # Entferne Sonderzeichen und Zahlen, behalte nur Buchstaben und Satzzeichen
        text = re.sub(r'[^\w\s.,;:!?]', '', text)
        return text.strip()
    
    def get_word_count(self, text: str) -> int:
        """Zählt die Anzahl der Wörter im Text"""
        words = word_tokenize(text, language='german')
        return len(words)
    
    def get_character_count(self, text: str, include_spaces: bool = True) -> int:
        """Zählt die Anzahl der Zeichen im Text"""
        if not include_spaces:
            text = text.replace(" ", "")
        return len(text)
    
    def get_sentence_count(self, text: str) -> int:
        """Zählt die Anzahl der Sätze im Text"""
        sentences = sent_tokenize(text, language='german')
        return len(sentences)
    
    def get_average_word_length(self, text: str) -> float:
        """Berechnet die durchschnittliche Wortlänge"""
        words = word_tokenize(text, language='german')
        if not words:
            return 0.0
        return sum(len(word) for word in words) / len(words)
    
    def get_reading_time(self, text: str, words_per_minute: int = 200) -> float:
        """Schätzt die Lesezeit in Minuten"""
        word_count = self.get_word_count(text)
        return round(word_count / words_per_minute, 1)
    
    def get_keywords(self, text: str, top_n: int = 5) -> List[Tuple[str, int]]:
        """Findet die häufigsten Schlüsselwörter im Text"""
        # Entferne Satzzeichen und konvertiere zu Kleinbuchstaben
        words = [word.lower() for word in word_tokenize(text, language='german') 
                if word.isalnum() and word.lower() not in self.stop_words]
        
        # Zähle die Häufigkeit der Wörter
        word_freq = Counter(words)
        return word_freq.most_common(top_n)
    
    def get_readability_score(self, text: str) -> float:
        """Berechnet den Lesbarkeitsindex (Flesch-Reading-Ease für Deutsch)"""
        sentences = sent_tokenize(text, language='german')
        words = word_tokenize(text, language='german')
        
        if not sentences or not words:
            return 0.0
            
        # Anzahl der Silben schätzen (einfache Methode)
        syllables = sum([len(re.findall(r'[aeiouäöüyAEIOUÄÖÜY]', word)) 
                       for word in words if word.isalnum()])
        
        # Flesch-Reading-Ease Formel für Deutsch
        words_per_sentence = len(words) / len(sentences)
        syllables_per_word = syllables / len(words)
        
        score = 180 - words_per_sentence - (58.5 * syllables_per_word)
        return max(0, min(100, score))  # Auf Bereich 0-100 begrenzen
    
    def analyze_text(self, text: str) -> Dict:
        """Führt eine umfassende Textanalyse durch"""
        if not text.strip():
            return {"error": "Kein Text zur Analyse vorhanden."}
        
        cleaned_text = self._clean_text(text)
        
        return {
            "word_count": self.get_word_count(cleaned_text),
            "character_count": self.get_character_count(cleaned_text),
            "character_count_no_spaces": self.get_character_count(cleaned_text, include_spaces=False),
            "sentence_count": self.get_sentence_count(cleaned_text),
            "average_word_length": round(self.get_average_word_length(cleaned_text), 2),
            "reading_time_minutes": self.get_reading_time(cleaned_text),
            "readability_score": round(self.get_readability_score(cleaned_text), 1),
            "top_keywords": self.get_keywords(cleaned_text, top_n=5)
        }
    
    def format_analysis(self, analysis: Dict) -> str:
        """Formatiert die Analyseergebnisse als lesbaren Text"""
        if "error" in analysis:
            return analysis["error"]
        
        result = [
            "📊 **Textanalyse-Ergebnisse:**",
            f"• Wörter: {analysis['word_count']}",
            f"• Zeichen: {analysis['character_count']} (davon {analysis['character_count_no_spaces']} ohne Leerzeichen)",
            f"• Sätze: {analysis['sentence_count']}",
            f"• Durchschnittliche Wortlänge: {analysis['average_word_length']} Zeichen",
            f"• Geschätzte Lesezeit: {analysis['reading_time_minutes']} Minuten (200 WpM)",
            f"• Lesbarkeitsindex: {analysis['readability_score']}/100 (höher = leichter zu lesen)",
            "\n🔍 **Häufigste Begriffe:**"
        ]
        
        # Füge die häufigsten Begriffe hinzu
        if analysis['top_keywords']:
            for word, count in analysis['top_keywords']:
                result.append(f"  - {word}: {count}x")
        else:
            result.append("  Keine signifikanten Begriffe gefunden.")
        
        # Füge eine Bewertung der Lesbarkeit hinzu
        readability = analysis['readability_score']
        if readability >= 80:
            rating = "Sehr leicht zu lesen (Grundschulniveau)"
        elif readability >= 60:
            rating = "Leicht zu lesen"
        elif readability >= 40:
            rating = "Mittelschwer zu lesen"
        elif readability >= 20:
            rating = "Schwer zu lesen (akademisches Niveau)"
        else:
            rating = "Sehr schwer zu lesen (Fachliteratur)"
            
        result.append(f"\n📚 **Lesbarkeitsbewertung:** {rating}")
        
        # Füge Tipps zur Verbesserung der Lesbarkeit hinzu, wenn nötig
        if readability < 50:
            result.extend([
                "",
                "💡 **Tipps zur besseren Lesbarkeit:**",
                "• Kürzere Sätze verwenden",
                "• Komplexe Wörter vereinfachen",
                "• Absätze einfügen, um den Text zu gliedern",
                "• Aufzählungen für Listen verwenden"
            ])
        
        return "\n".join(result)
    
    def execute(self, command: str, text: str = "") -> str:
        """Führt den Befehl mit dem angegebenen Text aus.
        
        Args:
            command: Der auszuführende Befehl
            text: Der zu analysierende Text
            
        Returns:
            Die formatierte Analyse oder eine Fehlermeldung
        """
        try:
            if not text.strip():
                return "Bitte geben Sie einen Text zur Analyse ein."
                
            # Führe die Analyse durch
            analysis = self.analyze_text(text)
            
            # Formatiere das Ergebnis
            return self.format_analysis(analysis)
            
        except Exception as e:
            logger.error(f"Fehler bei der Textanalyse: {e}", exc_info=True)
            return f"Bei der Textanalyse ist ein Fehler aufgetreten: {str(e)}"
