import os
import json
import glob
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DatensatzManager:
    """
    Verwaltet das Laden und Verarbeiten von Datensätzen aus dem datasets-Ordner.
    Unterstützt verschiedene Dateiformate wie .txt, .json, .csv.
    """
    
    def __init__(self, daten_verzeichnis: str = "datasets"):
        """
        Initialisiert den Datensatz-Manager.
        
        Args:
            daten_verzeichnis: Pfad zum Verzeichnis mit den Datensätzen
        """
        self.daten_verzeichnis = Path(daten_verzeichnis)
        self.daten_verzeichnis.mkdir(exist_ok=True)
        self.geladene_daten: Dict[str, Any] = {}
        
    def lade_alle_datensaetze(self) -> Dict[str, Any]:
        """
        Lädt alle verfügbaren Datensätze aus dem Verzeichnis.
        
        Returns:
            Ein Dictionary mit den geladenen Datensätzen, gruppiert nach Dateityp
        """
        self.geladene_daten = {
            'text': self._lade_textdateien(),
            'json': self._lade_json_dateien(),
            'csv': self._lade_csv_dateien()
        }
        return self.geladene_daten
    
    def _lade_textdateien(self) -> Dict[str, str]:
        """Lädt alle .txt Dateien aus dem Verzeichnis."""
        texte = {}
        for datei in self.daten_verzeichnis.glob('*.txt'):
            try:
                with open(datei, 'r', encoding='utf-8') as f:
                    texte[datei.stem] = f.read()
                logger.info(f"Textdatei geladen: {datei.name}")
            except Exception as e:
                logger.error(f"Fehler beim Lesen von {datei}: {e}")
        return texte
    
    def _lade_json_dateien(self) -> Dict[str, Any]:
        """Lädt alle .json Dateien aus dem Verzeichnis."""
        daten = {}
        for datei in self.daten_verzeichnis.glob('*.json'):
            try:
                with open(datei, 'r', encoding='utf-8') as f:
                    daten[datei.stem] = json.load(f)
                logger.info(f"JSON-Datei geladen: {datei.name}")
            except json.JSONDecodeError:
                logger.error(f"Ungültiges JSON-Format in {datei}")
            except Exception as e:
                logger.error(f"Fehler beim Lesen von {datei}: {e}")
        return daten
    
    def _lade_csv_dateien(self) -> Dict[str, List[Dict]]:
        """Lädt alle .csv Dateien aus dem Verzeichnis."""
        daten = {}
        for datei in self.daten_verzeichnis.glob('*.csv'):
            try:
                import pandas as pd
                daten[datei.stem] = pd.read_csv(datei).to_dict('records')
                logger.info(f"CSV-Datei geladen: {datei.name}")
            except ImportError:
                logger.warning("Pandas nicht installiert. CSV-Dateien können nicht geladen werden.")
                break
            except Exception as e:
                logger.error(f"Fehler beim Lesen von {datei}: {e}")
        return daten
    
    def suche_in_datensaetzen(self, suchbegriff: str) -> Dict[str, List[Dict]]:
        """
        Durchsucht alle geladenen Datensätze nach einem Suchbegriff.
        
        Args:
            suchbegriff: Der zu suchende Begriff
            
        Returns:
            Ein Dictionary mit den Fundstellen in den verschiedenen Datensätzen
        """
        if not self.geladene_daten:
            self.lade_alle_datensaetze()
            
        ergebnisse = {}
        suchbegriff = suchbegriff.lower()
        
        # Durchsuche Textdateien
        for name, inhalt in self.geladene_daten['text'].items():
            if suchbegriff in inhalt.lower():
                ergebnisse[f"text_{name}"] = [{"fundstelle": inhalt}]
        
        # Durchsuche JSON-Daten
        for name, daten in self.geladene_daten['json'].items():
            if isinstance(daten, dict):
                funde = self._durchsuche_dict(daten, suchbegriff)
                if funde:
                    ergebnisse[f"json_{name}"] = funde
        
        # Durchsuche CSV-Daten
        for name, eintraege in self.geladene_daten['csv'].items():
            if isinstance(eintraege, list):
                funde = []
                for eintrag in eintraege:
                    if any(suchbegriff in str(wert).lower() for wert in eintrag.values()):
                        funde.append(eintrag)
                if funde:
                    ergebnisse[f"csv_{name}"] = funde
        
        return ergebnisse
    
    def _durchsuche_dict(self, daten: Dict, suchbegriff: str, pfad: str = "") -> List[Dict]:
        """Hilfsfunktion zum Rekursiven Durchsuchen von Dictionaries."""
        funde = []
        for schluessel, wert in daten.items():
            aktueller_pfad = f"{pfad}.{schluessel}" if pfad else schluessel
            
            if isinstance(wert, dict):
                funde.extend(self._durchsuche_dict(wert, suchbegriff, aktueller_pfad))
            elif isinstance(wert, (list, tuple)):
                for i, item in enumerate(wert):
                    if isinstance(item, (dict, list, tuple)):
                        funde.extend(self._durchsuche_dict(item, suchbegriff, f"{aktueller_pfad}[{i}]"))
                    elif suchbegriff in str(item).lower():
                        funde.append({"pfad": f"{aktueller_pfad}[{i}]", "wert": item})
            elif suchbegriff in str(wert).lower():
                funde.append({"pfad": aktueller_pfad, "wert": wert})
                
        return funde
    
    def get_datensatz_info(self) -> Dict[str, int]:
        """Gibt Informationen über die verfügbaren Datensätze zurück."""
        if not self.geladene_daten:
            self.lade_alle_datensaetze()
            
        return {
            "text_dateien": len(self.geladene_daten['text']),
            "json_dateien": len(self.geladene_daten['json']),
            "csv_dateien": len(self.geladene_daten['csv']),
            "gesamt_dateien": (
                len(self.geladene_daten['text']) + 
                len(self.geladene_daten['json']) + 
                len(self.geladene_daten['csv'])
            )
        }
