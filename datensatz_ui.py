import customtkinter as ctk
from typing import Optional, Dict, Any, List
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

class DatensatzUI(ctk.CTkToplevel):
    """
    Benutzeroberfläche zur Verwaltung und Anzeige von Datensätzen.
    """
    
    def __init__(self, parent, datensatz_manager, **kwargs):
        """
        Initialisiert die Datensatz-Benutzeroberfläche.
        
        Args:
            parent: Elternfenster
            datensatz_manager: Instanz des Datensatz-Managers
        """
        super().__init__(parent, **kwargs)
        self.title("Datensatz-Manager")
        self.geometry("800x600")
        self.datensatz_manager = datensatz_manager
        
        # Variablen
        self.aktuelle_ergebnisse = {}
        
        # Benutzeroberfläche erstellen
        self._erstelle_ui()
        
        # Initialen Status aktualisieren
        self._aktualisiere_status()
    
    def _erstelle_ui(self):
        """Erstellt die Benutzeroberfläche."""
        # Hauptlayout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        # Suchleiste
        such_frame = ctk.CTkFrame(self)
        such_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        such_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(such_frame, text="Suche:").grid(row=0, column=0, padx=5, pady=5)
        
        self.such_eingabe = ctk.CTkEntry(such_frame)
        self.such_eingabe.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.such_eingabe.bind("<Return>", lambda e: self._suche_starten())
        
        such_button = ctk.CTkButton(such_frame, text="Suchen", command=self._suche_starten)
        such_button.grid(row=0, column=2, padx=5, pady=5)
        
        # Registerkarten für verschiedene Datensatztypen
        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        
        # Tab für Suchergebnisse
        self.tab_suche = self.tabview.add("Suchergebnisse")
        self.tab_suche.grid_columnconfigure(0, weight=1)
        self.tab_suche.grid_rowconfigure(0, weight=1)
        
        self.ergebnis_text = ctk.CTkTextbox(self.tab_suche, wrap="word")
        self.ergebnis_text.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        # Statusleiste
        self.status_label = ctk.CTkLabel(self, text="Bereit", anchor="w")
        self.status_label.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="ew")
    
    def _suche_starten(self):
        """Startet die Suche nach dem eingegebenen Begriff."""
        suchbegriff = self.such_eingabe.get().strip()
        if not suchbegriff:
            self.status_label.configure(text="Bitte geben Sie einen Suchbegriff ein.")
            return
            
        self.status_label.configure(text=f"Suche nach '{suchbegriff}'...")
        self.ergebnis_text.delete("1.0", "end")
        
        try:
            ergebnisse = self.datensatz_manager.suche_in_datensaetzen(suchbegriff)
            self.aktuelle_ergebnisse = ergebnisse
            self._zeige_ergebnisse(ergebnisse)
            self.status_label.configure(text=f"Suche abgeschlossen. {len(ergebnisse)} Ergebnisse gefunden.")
        except Exception as e:
            logger.error(f"Fehler bei der Suche: {e}")
            self.ergebnis_text.insert("end", f"Fehler bei der Suche: {str(e)}")
            self.status_label.configure(text="Fehler bei der Suche.")
    
    def _zeige_ergebnisse(self, ergebnisse: Dict[str, Any]):
        """Zeigt die Suchergebnisse an."""
        if not ergebnisse:
            self.ergebnis_text.insert("end", "Keine Ergebnisse gefunden.")
            return
            
        for dateiname, fundstellen in ergebnisse.items():
            self.ergebnis_text.insert("end", f"\n=== {dateiname} ===\n\n")
            
            if isinstance(fundstellen, list):
                for i, fund in enumerate(fundstellen, 1):
                    if isinstance(fund, dict):
                        if 'pfad' in fund:
                            self.ergebnis_text.insert("end", f"{i}. [{fund['pfad']}]\n")
                        if 'wert' in fund:
                            wert = str(fund['wert'])
                            if len(wert) > 200:
                                wert = wert[:200] + "..."
                            self.ergebnis_text.insert("end", f"   {wert}\n\n")
                    else:
                        self.ergebnis_text.insert("end", f"{i}. {str(fund)}\n\n")
            else:
                self.ergebnis_text.insert("end", f"{str(fundstellen)}\n\n")
    
    def _aktualisiere_status(self):
        """Aktualisiert den Status der geladenen Datensätze."""
        try:
            info = self.datensatz_manager.get_datensatz_info()
            status_text = (
                f"Geladene Datensätze: {info['gesamt_dateien']} "
                f"(TXT: {info['text_dateien']}, "
                f"JSON: {info['json_dateien']}, "
                f"CSV: {info['csv_dateien']})"
            )
            self.status_label.configure(text=status_text)
        except Exception as e:
            logger.error(f"Fehler beim Aktualisieren des Status: {e}")
            self.status_label.configure(text="Fehler beim Laden der Datensatz-Informationen")
