import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw
import math
import time

class JarvisTheme:
    def __init__(self):
        # Farbpalette
        self.colors = {
            'primary': '#0a192f',      # Dunkelblau (Hauptfarbe)
            'secondary': '#172a45',    # Helleres Blau (Sekundärfarbe)
            'accent': '#64ffda',       # Türkis (Akzentfarbe)
            'text': '#e6f1ff',         # Heller Text
            'subtext': '#8892b0',      # Hellerer Text für Sekundärinformationen
            'dark': '#020c1b',         # Sehr dunkelblau für Kontraste
            'success': '#4CAF50',      # Grün für Erfolgsmeldungen
            'warning': '#FFC107',      # Gelb für Warnungen
            'error': '#F44336',        # Rot für Fehler
            'highlight': '#1e90ff',    # Helles Blau für Hervorhebungen
            'button': '#1a365d',       # Dunkelblau für Buttons
            'button_hover': '#2c5282',  # Helleres Blau für Hover-Effekte
            'status': '#172a45'         # Farbe für die Statusleiste
        }
        
        # Schriftarten
        self.fonts = {
            'title': ('Rajdhani', 24, 'bold'),
            'subtitle': ('Rajdhani', 16, 'bold'),
            'normal': ('Segoe UI', 10),
            'mono': ('Consolas', 10),
            'button': ('Segoe UI', 10, 'bold'),
            'small': ('Segoe UI', 8),
        }
        
        # Stile für ttk-Widgets
        self.setup_styles()
    
    def setup_styles(self):
        """Initialisiert die ttk-Styles"""
        style = ttk.Style()
        
        # Allgemeine Einstellungen
        style.theme_use('clam')
        style.configure('.', background=self.colors['primary'])
        
        # Frame-Stile
        style.configure('TFrame', background=self.colors['primary'])
        style.configure('Secondary.TFrame', background=self.colors['secondary'])
        
        # Label-Stile
        style.configure('TLabel', 
                       background=self.colors['primary'],
                       foreground=self.colors['text'],
                       font=self.fonts['normal'])
        
        style.configure('Title.TLabel',
                       font=self.fonts['title'],
                       foreground=self.colors['accent'])
        
        style.configure('Subtitle.TLabel',
                       font=self.fonts['subtitle'],
                       foreground=self.colors['subtext'])
        
        # Button-Stile
        style.configure('TButton',
                       font=self.fonts['button'],
                       background=self.colors['secondary'],
                       foreground=self.colors['text'],
                       borderwidth=1,
                       relief='flat')
        
        style.map('TButton',
                 background=[('active', self.colors['highlight']),
                           ('pressed', self.colors['accent'])],
                 foreground=[('active', 'white'),
                           ('pressed', 'white')])
        
        # Entry-Stile
        style.configure('TEntry',
                      fieldbackground=self.colors['secondary'],
                      foreground=self.colors['text'],
                      borderwidth=0,
                      insertcolor=self.colors['accent'])
        
        # Notebook-Stile
        style.configure('TNotebook', background=self.colors['primary'])
        style.configure('TNotebook.Tab',
                      background=self.colors['secondary'],
                      foreground=self.colors['text'],
                      padding=[10, 5],
                      font=self.fonts['button'])
        
        style.map('TNotebook.Tab',
                background=[('selected', self.colors['accent']),
                          ('active', self.colors['highlight'])],
                foreground=[('selected', 'black'),
                          ('active', 'white')])
        
        # Scrollbar-Stile
        style.configure('Vertical.TScrollbar',
                      background=self.colors['secondary'],
                      arrowcolor=self.colors['text'],
                      bordercolor=self.colors['primary'],
                      arrowsize=12)
        
        style.map('Vertical.TScrollbar',
                background=[('active', self.colors['highlight'])])

    def create_gradient(self, width, height, color1, color2):
        """Erstellt einen Farbverlauf als Bild"""
        image = Image.new('RGBA', (width, height), color1)
        draw = ImageDraw.Draw(image)
        
        for y in range(height):
            # Interpoliere zwischen den beiden Farben
            r = int(color1[0] + (color2[0] - color1[0]) * y / height)
            g = int(color1[1] + (color2[1] - color1[1]) * y / height)
            b = int(color1[2] + (color2[2] - color1[2]) * y / height)
            draw.line([(0, y), (width, y)], fill=(r, g, b, 255))
        
        return ImageTk.PhotoImage(image)

    def hex_to_rgb(self, hex_color):
        """Konvertiert einen Hex-Farbcode in ein RGB-Tupel"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def create_rounded_rectangle(self, x1, y1, x2, y2, radius=25, **kwargs):
        """Erstellt ein abgerundetes Rechteck"""
        points = [x1+radius, y1,
                 x1+radius, y1,
                 x2-radius, y1,
                 x2-radius, y1,
                 x2, y1,
                 x2, y1+radius,
                 x2, y2-radius,
                 x2, y2,
                 x2-radius, y2,
                 x1+radius, y2,
                 x1, y2,
                 x1, y2-radius,
                 x1, y1+radius,
                 x1, y1]
        
        return points

    def add_glow_effect(self, widget, color):
        """Fügt einen Glüheffekt zu einem Widget hinzu"""
        def update_glow():
            try:
                if not widget.winfo_exists():
                    return
                
                # Erstelle ein leicht größeres Rechteck für den Glüheffekt
                x, y, w, h = (widget.winfo_x()-2, widget.winfo_y()-2,
                            widget.winfo_width()+4, widget.winfo_height()+4)
                
                # Erstelle einen Canvas für den Glüheffekt, falls noch nicht vorhanden
                if not hasattr(widget, '_glow_canvas'):
                    parent = widget.master
                    widget._glow_canvas = tk.Canvas(parent, highlightthickness=0)
                    widget._glow_canvas.place(x=x, y=y, width=w, height=h)
                
                # Zeichne den Glüheffekt
                widget._glow_canvas.delete('glow')
                widget._glow_canvas.create_rectangle(2, 2, w-2, h-2,
                                                  outline=color,
                                                  width=2,
                                                  tags='glow')
                
                # Bewege den Canvas hinter das Widget
                widget._glow_canvas.lower(widget)
                
                # Plane die nächste Aktualisierung
                widget.after(50, update_glow)
            except:
                pass
        
        # Starte die Animation
        update_glow()

    def add_typing_effect(self, widget, text, delay=30):
        """Fügt einen Schreibmaschinen-Effekt zu einem Label hinzu"""
        widget.config(text='')
        
        def type_text(index=0):
            if index < len(text):
                widget.config(text=widget.cget('text') + text[index])
                widget.after(delay, type_text, index + 1)
        
        type_text()

    def create_pulse_animation(self, widget, color1, color2, duration=2000):
        """Erstellt eine Puls-Animation für ein Widget"""
        def pulse(step=0):
            if not hasattr(widget, '_pulse_running') or not widget._pulse_running:
                return
                
            # Berechne die aktuelle Farbe basierend auf dem Schritt
            progress = (math.sin(step * 0.05) + 1) / 2  # Wert zwischen 0 und 1
            
            r = int(color1[0] + (color2[0] - color1[0]) * progress)
            g = int(color1[1] + (color2[1] - color1[1]) * progress)
            b = int(color1[2] + (color2[2] - color1[2]) * progress)
            
            hex_color = f'#{r:02x}{g:02x}{b:02x}'
            
            # Aktualisiere die Widget-Farbe
            widget.config(highlightbackground=hex_color,
                        highlightcolor=hex_color,
                        highlightthickness=1)
            
            # Nächster Schritt
            widget.after(30, pulse, step + 1)
        
        # Starte die Animation
        widget._pulse_running = True
        pulse()
    
    def stop_pulse_animation(self, widget):
        """Stoppt die Puls-Animation für ein Widget"""
        if hasattr(widget, '_pulse_running'):
            widget._pulse_running = False
            widget.config(highlightthickness=0)
