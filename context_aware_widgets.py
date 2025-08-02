import customtkinter as ctk
import tkinter as tk
from typing import Optional, Callable, Dict, Any
from context_manager import LongTermMemory, ContextScorer

class ContextAwareText(ctk.CTkTextbox):
    """Eine erweiterte Textbox mit Kontext-Markierungsfunktionen."""
    
    def __init__(self, *args, **kwargs):
        self.context_colors = kwargs.pop('context_colors', None) or {
            'high': '#2a5c8a',    # Dunkelblau für hohe Priorität
            'medium': '#3a6ea5',  # Mittelblau für mittlere Priorität
            'low': '#4a7ebf'      # Hellblau für niedrige Priorität
        }
        
        # Ensure we have text_color and fg_color for proper theming
        kwargs.setdefault('text_color', ('gray10', 'gray90'))
        kwargs.setdefault('fg_color', ('gray95', 'gray15'))
        
        super().__init__(*args, **kwargs)
        self.context_tags: Dict[str, str] = {}
        self._setup_context_tags()
    
    def _setup_context_tags(self):
        """Initialisiert die Tags für die Kontext-Hervorhebung."""
        # Konfiguriere Tags für verschiedene Kontext-Level
        for level, color in self.context_colors.items():
            self.tag_configure(f'context_{level}', background=color)
            
        # Tag für Auswahl
        self.tag_configure('selected', background='#3a7ebf', foreground='white')
        
        # Tag für hervorgehobenen Text
        self.tag_configure('highlight', background='#ffeb3b')
        
        # Ensure we have default font settings that work with CustomTkinter
        self.configure(
            font=ctk.CTkFont(family='Segoe UI', size=11),
            wrap='word',
            undo=True
        )
    
    @staticmethod
    def _adjust_color(hex_color: str, opacity: float) -> str:
        """Ändert die Deckkraft einer Hex-Farbe."""
        # For CustomTkinter, we'll use the color as is and let it handle opacity
        # through the alpha channel if needed
        return hex_color
    
    def add_context_highlight(self, start: str, end: str, context_level: str = 'medium'):
        """Fügt eine Kontext-Hervorhebung hinzu."""
        tag_name = f'context_{context_level}'
        self.tag_add(tag_name, start, end)
    
    def clear_context_highlights(self):
        """Entfernt alle Kontext-Hervorhebungen."""
        for tag in self.tag_names():
            if tag.startswith('context_'):
                self.tag_remove(tag, '1.0', 'end')
    
    def highlight_selection(self):
        """Markiert den ausgewählten Text."""
        try:
            sel_start = self.index("sel.first")
            sel_end = self.index("sel.last")
            self.tag_add('selected', sel_start, sel_end)
            return sel_start, sel_end
        except tk.TclError:
            return None
            
    # Alias for compatibility with both Tkinter and CustomTkinter
    tag_configure = ctk.CTkTextbox.tag_config


class ContextAwareChatDisplay(ContextAwareText):
    """Eine Chat-Anzeige mit Kontextbewusstsein."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_context_menu()
        self.memory: Optional[LongTermMemory] = None
        self.scorer: Optional[ContextScorer] = None
    
    def set_memory(self, memory: LongTermMemory):
        """Setzt die Gedächtnisinstanz."""
        self.memory = memory
    
    def set_scorer(self, scorer: ContextScorer):
        """Setzt den Scorer für die Relevanzbewertung."""
        self.scorer = scorer
    
    def _setup_context_menu(self):
        """Richtet das Kontextmenü ein."""
        self.context_menu = tk.Menu(self, tearoff=0)
        self.context_menu.add_command(
            label="Als wichtigen Fakt speichern",
            command=self._save_selection_as_fact
        )
        self.context_menu.add_separator()
        self.context_menu.add_command(
            label="Kontext hervorheben",
            command=self._highlight_context
        )
        self.context_menu.add_command(
            label="Hervorhebungen löschen",
            command=self.clear_context_highlights
        )
        
        # Bind Rechtsklick-Ereignis
        self.bind("<Button-3>", self._show_context_menu)
    
    def _show_context_menu(self, event):
        """Zeigt das Kontextmenü an der Mausposition an."""
        try:
            self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.context_menu.grab_release()
    
    def _save_selection_as_fact(self):
        """Speichert die Auswahl als wichtigen Fakt."""
        try:
            selected_text = self.get("sel.first", "sel.last").strip()
            if selected_text and self.memory:
                # Bewerte die Relevanz des ausgewählten Textes
                relevance = 3  # Standardpriorität
                if self.scorer:
                    relevance = int(self.scorer.score_relevance(selected_text) * 5)
                    relevance = max(1, min(5, relevance))  # Auf 1-5 Skala beschränken
                
                # Füge den Fakt hinzu
                fact_id = self.memory.add_fact(
                    selected_text,
                    priority=relevance,
                    source="user_selection",
                    metadata={"created_from": "chat_selection"}
                )
                
                # Markiere den Text basierend auf der Priorität
                if relevance >= 4:
                    context_level = 'high'
                elif relevance >= 2:
                    context_level = 'medium'
                else:
                    context_level = 'low'
                
                self.add_context_highlight("sel.first", "sel.last", context_level)
                return fact_id
        except tk.TclError:
            pass  # Kein Text ausgewählt
        return None
    
    def _highlight_context(self):
        """Hebt den ausgewählten Text als Kontext hervor."""
        try:
            sel_start = self.index("sel.first")
            sel_end = self.index("sel.last")
            self.add_context_highlight(sel_start, sel_end, 'medium')
        except tk.TclError:
            pass  # Kein Text ausgewählt


class FactSidebar(ctk.CTkFrame):
    """Eine Sidebar zur Anzeige und Verwaltung von Fakten."""
    
    def __init__(self, master, memory: LongTermMemory, **kwargs):
        super().__init__(master, **kwargs)
        self.memory = memory
        self._setup_ui()
    
    def _setup_ui(self):
        """Richtet die Benutzeroberfläche ein."""
        # Titel
        title = ctk.CTkLabel(
            self, 
            text="Wichtige Fakten",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        title.pack(pady=10, padx=10, anchor='w')
        
        # Suchleiste
        self.search_var = ctk.StringVar()
        self.search_var.trace('w', self._on_search)
        
        search_frame = ctk.CTkFrame(self, fg_color="transparent")
        search_frame.pack(fill='x', padx=5, pady=5)
        
        search_entry = ctk.CTkEntry(
            search_frame, 
            placeholder_text="Fakten durchsuchen...",
            textvariable=self.search_var
        )
        search_entry.pack(side='left', fill='x', expand=True)
        
        # Fakten-Liste
        self.facts_frame = ctk.CTkScrollableFrame(self)
        self.facts_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Lade initiale Fakten
        self._update_facts_list()
    
    def _on_search(self, *args):
        """Wird aufgerufen, wenn sich der Suchtext ändert."""
        self._update_facts_list(self.search_var.get())
    
    def _update_facts_list(self, query: str = ""):
        """Aktualisiert die Liste der angezeigten Fakten."""
        # Lösche vorhandene Einträge
        for widget in self.facts_frame.winfo_children():
            widget.destroy()
        
        # Hole relevante Fakten
        facts = self.memory.get_relevant_facts(query, limit=20)
        
        if not facts:
            empty_label = ctk.CTkLabel(
                self.facts_frame, 
                text="Keine Fakten gefunden" if query else "Keine Fakten vorhanden",
                text_color="gray"
            )
            empty_label.pack(pady=10)
            return
        
        # Zeige Fakten an
        for fact in facts:
            self._add_fact_widget(fact)
    
    def _add_fact_widget(self, fact: dict):
        """Fügt ein Widget für einen einzelnen Fakt hinzu."""
        fact_frame = ctk.CTkFrame(self.facts_frame, corner_radius=5)
        fact_frame.pack(fill='x', pady=2, padx=2)
        
        # Fakt-Text
        text = fact.get('fact', '')
        if len(text) > 100:  # Kürze lange Texte
            text = text[:97] + '...'
        
        # Prioritätsanzeige
        priority = fact.get('priority', 1)
        priority_color = {
            1: '#4CAF50',  # Grün
            2: '#8BC34A',
            3: '#FFC107',  # Gelb
            4: '#FF9800',  # Orange
            5: '#F44336'   # Rot
        }.get(priority, '#9E9E9E')
        
        # Frame für die Kopfzeile (Priorität und Löschen-Button)
        header_frame = ctk.CTkFrame(fact_frame, fg_color="transparent")
        header_frame.pack(fill='x', padx=5, pady=(5, 0))
        
        # Prioritätsanzeige
        ctk.CTkLabel(
            header_frame, 
            text="★" * min(5, priority) + "☆" * max(0, 5 - min(5, priority)),
            text_color=priority_color,
            font=ctk.CTkFont(weight="bold")
        ).pack(side='left')
        
        # Löschen-Button
        def delete_fact():
            if self.memory.remove_fact(fact['id']):
                self._update_facts_list(self.search_var.get())
        
        ctk.CTkButton(
            header_frame,
            text="×",
            width=20,
            height=20,
            fg_color="transparent",
            hover_color="#ff4444",
            text_color="#999999",
            font=ctk.CTkFont(weight="bold"),
            command=delete_fact
        ).pack(side='right')
        
        # Fakt-Text
        text_label = ctk.CTkLabel(
            fact_frame, 
            text=text,
            wraplength=250,
            justify='left',
            anchor='w'
        )
        text_label.pack(fill='x', padx=5, pady=(0, 5))
        
        # Metadaten (Datum, Quelle)
        meta_texts = []
        if 'timestamp' in fact:
            try:
                dt = fact['timestamp']
                if isinstance(dt, str):
                    from datetime import datetime
                    dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
                meta_texts.append(dt.strftime('%d.%m.%Y %H:%M'))
            except (ValueError, AttributeError):
                pass
        
        if 'source' in fact:
            meta_texts.append(f"Quelle: {fact['source']}")
        
        if meta_texts:
            meta_label = ctk.CTkLabel(
                fact_frame,
                text=" • ".join(meta_texts),
                text_color="#888888",
                font=ctk.CTkFont(size=10)
            )
            meta_label.pack(side='bottom', anchor='e', padx=5, pady=(0, 5))
