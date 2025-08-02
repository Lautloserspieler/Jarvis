import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from collections import defaultdict, Counter
import math

logger = logging.getLogger(__name__)

@dataclass
class ScoredContext:
    """Repräsentiert einen bewerteten Kontext."""
    text: str
    score: float
    metadata: Dict[str, Any] = None
    source: str = None

class ContextScorer:
    """Bewertet und priorisiert Kontexte für die KI-Antwortgenerierung."""
    
    def __init__(self, n_gram_range: Tuple[int, int] = (1, 3)):
        """Initialisiert den ContextScorer.
        
        Args:
            n_gram_range: Bereich der N-Gramme für die Ähnlichkeitsberechnung (min, max)
        """
        self.n_gram_range = n_gram_range
        self.stop_words = self._load_stop_words()
    
    def _load_stop_words(self) -> set:
        """Lädt eine Liste von Stoppwörtern.
        
        Returns:
            Ein Set von Stoppwörtern
        """
        # Einfache deutsche Stoppwörter
        stop_words = {
            'der', 'die', 'das', 'und', 'oder', 'aber', 'denn', 'weil', 'wenn', 'als',
            'ein', 'eine', 'einer', 'eines', 'einem', 'einen', 'mein', 'dein', 'sein',
            'ihr', 'unser', 'euer', 'meine', 'deine', 'seine', 'ihre', 'unsere', 'eure',
            'in', 'an', 'auf', 'aus', 'bei', 'mit', 'nach', 'seit', 'von', 'zu', 'bis',
            'durch', 'für', 'gegen', 'ohne', 'um', 'als', 'wie', 'auch', 'sich', 'nicht'
        }
        return stop_words
    
    def preprocess_text(self, text: str) -> str:
        """Bereitet den Text für die Verarbeitung vor.
        
        Args:
            text: Der zu verarbeitende Text
            
        Returns:
            Der vorverarbeitete Text
        """
        if not text:
            return ""
        
        # Kleinschreibung
        text = text.lower()
        
        # Sonderzeichen entfernen, außer Satzzeichen
        text = re.sub(r'[^\w\s.,!?]', ' ', text)
        
        # Mehrfache Leerzeichen entfernen
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Zerlegt den Text in Tokens.
        
        Args:
            text: Der zu tokenisierende Text
            
        Returns:
            Eine Liste von Tokens
        """
        if not text:
            return []
        
        # Einfache Tokenisierung an Leerzeichen
        tokens = text.split()
        
        # Entferne Stoppwörter
        tokens = [token for token in tokens if token not in self.stop_words]
        
        return tokens
    
    def get_ngrams(self, tokens: List[str], n: int) -> List[tuple]:
        """Generiert N-Gramme aus einer Token-Liste.
        
        Args:
            tokens: Liste von Tokens
            n: Größe der N-Gramme
            
        Returns:
            Eine Liste von N-Grammen
        """
        if n <= 0 or not tokens:
            return []
        
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    def get_all_ngrams(self, tokens: List[str]) -> Dict[int, List[tuple]]:
        """Generiert N-Gramme für alle Größen im n_gram_range.
        
        Args:
            tokens: Liste von Tokens
            
        Returns:
            Ein Dictionary mit N-Grammen für jede Größe
        """
        ngrams = {}
        min_n, max_n = self.n_gram_range
        
        for n in range(min_n, max_n + 1):
            ngrams[n] = self.get_ngrams(tokens, n)
            
        return ngrams
    
    def calculate_tfidf(self, documents: List[str]) -> Tuple[Dict[str, float], Dict[tuple, float]]:
        """Berechnet die TF-IDF-Gewichtung für eine Sammlung von Dokumenten.
        
        Args:
            documents: Liste von Dokumenten (Strings)
            
        Returns:
            Ein Tupel aus (tfidf_weights, doc_freq)
        """
        # Dokumenthäufigkeit (DF) berechnen
        doc_freq = defaultdict(int)
        tf = []
        
        for doc in documents:
            # Text vorverarbeiten und tokenisieren
            processed = self.preprocess_text(doc)
            tokens = self.tokenize(processed)
            
            # Termfrequenz (TF) für dieses Dokument berechnen
            doc_tf = Counter(tokens)
            tf.append(doc_tf)
            
            # Einmal pro Dokument zählen
            for term in set(tokens):
                doc_freq[term] += 1
        
        # Inverse Dokumenthäufigkeit (IDF) berechnen
        idf = {}
        num_docs = len(documents)
        
        for term, df in doc_freq.items():
            idf[term] = math.log((num_docs + 1) / (df + 1)) + 1  # Glättung für Terme, die in allen Dokumenten vorkommen
        
        # TF-IDF für jedes Dokument berechnen
        tfidf_weights = []
        
        for doc_tf in tf:
            doc_tfidf = {}
            for term, freq in doc_tf.items():
                doc_tfidf[term] = freq * idf[term]
            tfidf_weights.append(doc_tfidf)
        
        return tfidf_weights, idf
    
    def calculate_similarity(self, query: str, contexts: List[Dict[str, Any]]) -> List[ScoredContext]:
        """Berechnet die Ähnlichkeit zwischen einer Anfrage und einer Liste von Kontexten.
        
        Args:
            query: Die Suchanfrage
            contexts: Liste von Kontexten mit mindestens einem 'text'-Feld
            
        Returns:
            Eine Liste von ScoredContext-Objekten, sortiert nach Relevanz
        """
        if not query or not contexts:
            return []
        
        # Extrahiere Texte aus den Kontexten
        texts = [ctx.get('text', '') for ctx in contexts]
        
        # TF-IDF für alle Dokumente berechnen (inkl. Query)
        all_texts = texts + [query]
        tfidf_weights, _ = self.calculate_tfidf(all_texts)
        
        # TF-IDF für die Abfrage (letztes Element)
        query_tfidf = tfidf_weights[-1]
        
        # Kosinus-Ähnlichkeit für jeden Kontext berechnen
        scored_contexts = []
        
        for i, ctx in enumerate(contexts):
            if i >= len(tfidf_weights) - 1:  # -1, da die Query das letzte Element ist
                continue
                
            doc_tfidf = tfidf_weights[i]
            
            # Kosinus-Ähnlichkeit berechnen
            similarity = self.cosine_similarity(query_tfidf, doc_tfidf)
            
            # Kontext mit Bewertung speichern
            scored = ScoredContext(
                text=ctx.get('text', ''),
                score=similarity,
                metadata=ctx.get('metadata', {}),
                source=ctx.get('source')
            )
            scored_contexts.append(scored)
        
        # Nach Relevanz sortieren (absteigend)
        scored_contexts.sort(key=lambda x: x.score, reverse=True)
        
        return scored_contexts
    
    def cosine_similarity(self, vec1: Dict[Any, float], vec2: Dict[Any, float]) -> float:
        """Berechnet die Kosinus-Ähnlichkeit zwischen zwei Vektoren.
        
        Args:
            vec1: Erster Vektor als Dictionary
            vec2: Zweiter Vektor als Dictionary
            
        Returns:
            Die Kosinus-Ähnlichkeit zwischen den Vektoren
        """
        # Alle einzigartigen Terme
        terms = set(vec1.keys()).union(set(vec2.keys()))
        
        if not terms:
            return 0.0
        
        # Vektoren erstellen
        v1 = np.array([vec1.get(term, 0.0) for term in terms])
        v2 = np.array([vec2.get(term, 0.0) for term in terms])
        
        # Kosinus-Ähnlichkeit berechnen
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
            
        return dot_product / (norm_v1 * norm_v2)
    
    def rank_contexts(
        self, 
        query: str, 
        contexts: List[Dict[str, Any]], 
        top_k: int = 5,
        min_score: float = 0.1
    ) -> List[Dict[str, Any]]:
        """Sortiert und filtert Kontexte basierend auf ihrer Relevanz für die Anfrage.
        
        Args:
            query: Die Suchanfrage
            contexts: Liste von Kontexten mit mindestens einem 'text'-Feld
            top_k: Maximale Anzahl zurückzugebender Kontexte
            min_score: Mindestähnlichkeits-Score für die Berücksichtigung
            
        Returns:
            Eine Liste der relevantesten Kontexte mit zusätzlichen Metadaten
        """
        scored = self.calculate_similarity(query, contexts)
        
        # Filter nach Mindest-Score und begrenze die Anzahl
        filtered = [
            {
                'text': ctx.text,
                'score': ctx.score,
                'metadata': ctx.metadata or {},
                'source': ctx.source
            }
            for ctx in scored 
            if ctx.score >= min_score
        ][:top_k]
        
        return filtered
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """Extrahiert die wichtigsten Schlüsselwörter aus einem Text.
        
        Args:
            text: Der Eingabetext
            top_n: Anzahl der zurückzugebenden Schlüsselwörter
            
        Returns:
            Eine Liste von (wort, score) Tupeln, sortiert nach Wichtigkeit
        """
        if not text:
            return []
        
        # Text vorverarbeiten und tokenisieren
        processed = self.preprocess_text(text)
        tokens = self.tokenize(processed)
        
        # TF-IDF für das einzelne Dokument berechnen
        tf = Counter(tokens)
        
        # Einfache Gewichtung: TF * IDF (hier nur TF, da nur ein Dokument)
        # In einer echten Anwendung würde man eine größere Dokumentensammlung verwenden
        scores = {term: freq * (1.0 + math.log(1.0 + math.log(freq))) for term, freq in tf.items()}
        
        # Nach Wichtigkeit sortieren
        sorted_terms = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_terms[:top_n]
