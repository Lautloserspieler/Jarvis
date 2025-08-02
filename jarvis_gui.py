import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog, Menu
from PIL import Image, ImageTk, ImageDraw
import os
import json
import threading
import queue
import time
from datetime import datetime
import logging
from jarvis_theme import JarvisTheme

class JarvisGUI:
    def __init__(self, root):
        self.root = root
        self.theme = JarvisTheme()
        self.setup_window()
        self.create_widgets()
        self.setup_menu()
        
    def setup_window(self):
        """Konfiguriert das Hauptfenster"""
        self.root.title("J.A.R.V.I.S. - Just A Rather Very Intelligent System")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        self.root.configure(bg=self.theme.colors['primary'])
        
        # Fenster-Icon setzen
        try:
            self.root.iconbitmap("assets/Jarvis.ico")
        except:
            pass
            
        # Stil für das Fenster
        self.root.option_add('*Font', self.theme.fonts['normal'])
        
    def create_widgets(self):
        """Erstellt die GUI-Elemente"""
        # Hauptcontainer
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Linke Seitenleiste
        self.sidebar = ttk.Frame(self.main_frame, style='Secondary.TFrame', width=250)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self.sidebar.pack_propagate(False)
        
        # Logo-Bereich
        self.logo_frame = ttk.Frame(self.sidebar, style='Secondary.TFrame')
        self.logo_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.logo_label = ttk.Label(
            self.logo_frame,
            text="J.A.R.V.I.S.",
            style='Title.TLabel',
            anchor='center'
        )
        self.logo_label.pack(fill=tk.X, pady=20)
        
        # Chat-Liste
        self.chat_list_frame = ttk.LabelFrame(
            self.sidebar,
            text=" Konversationen ",
            style='Secondary.TLabelframe'
        )
        self.chat_list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.chat_list = tk.Listbox(
            self.chat_list_frame,
            bg=self.theme.colors['secondary'],
            fg=self.theme.colors['text'],
            selectbackground=self.theme.colors['accent'],
            selectforeground='black',
            borderwidth=0,
            highlightthickness=0,
            font=self.theme.fonts['normal']
        )
        self.chat_list.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Neuer Chat-Button
        self.new_chat_btn = ttk.Button(
            self.sidebar,
            text="Neuer Chat",
            command=self.new_chat,
            style='TButton'
        )
        self.new_chat_btn.pack(fill=tk.X, pady=(0, 10))
        
        # Hauptbereich
        self.content_frame = ttk.Frame(self.main_frame)
        self.content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Chat-Anzeige
        self.chat_display = scrolledtext.ScrolledText(
            self.content_frame,
            wrap=tk.WORD,
            bg=self.theme.colors['primary'],
            fg=self.theme.colors['text'],
            insertbackground=self.theme.colors['accent'],
            selectbackground=self.theme.colors['accent'],
            selectforeground='black',
            padx=15,
            pady=15,
            font=self.theme.fonts['normal'],
            borderwidth=0,
            highlightthickness=0
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        
        # Eingabebereich
        self.input_frame = ttk.Frame(self.content_frame)
        self.input_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.user_input = tk.Text(
            self.input_frame,
            height=4,
            bg=self.theme.colors['secondary'],
            fg=self.theme.colors['text'],
            insertbackground=self.theme.colors['accent'],
            selectbackground=self.theme.colors['accent'],
            selectforeground='black',
            font=self.theme.fonts['normal'],
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=self.theme.colors['accent'],
            highlightcolor=self.theme.colors['accent']
        )
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        self.send_btn = ttk.Button(
            self.input_frame,
            text="Senden",
            command=self.send_message,
            style='TButton'
        )
        self.send_btn.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Statusleiste
        self.status_bar = ttk.Label(
            self.root,
            text="Bereit",
            relief=tk.SUNKEN,
            anchor=tk.W,
            style='TLabel'
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Tastenkürzel
        self.root.bind('<Return>', lambda e: self.send_message())
        self.root.bind('<Control-n>', lambda e: self.new_chat())
        
    def setup_menu(self):
        """Erstellt das Hauptmenü"""
        menubar = Menu(self.root, bg=self.theme.colors['secondary'], fg=self.theme.colors['text'])
        
        # Datei-Menü
        file_menu = Menu(menubar, tearoff=0, bg=self.theme.colors['secondary'], fg=self.theme.colors['text'])
        file_menu.add_command(label="Neuer Chat", command=self.new_chat, accelerator="Strg+N")
        file_menu.add_separator()
        file_menu.add_command(label="Speichern", command=self.save_chat)
        file_menu.add_command(label="Laden", command=self.load_chat)
        file_menu.add_separator()
        file_menu.add_command(label="Beenden", command=self.root.quit)
        menubar.add_cascade(label="Datei", menu=file_menu)
        
        # Bearbeiten-Menü
        edit_menu = Menu(menubar, tearoff=0, bg=self.theme.colors['secondary'], fg=self.theme.colors['text'])
        edit_menu.add_command(label="Ausschneiden", command=self.cut_text)
        edit_menu.add_command(label="Kopieren", command=self.copy_text)
        edit_menu.add_command(label="Einfügen", command=self.paste_text)
        menubar.add_cascade(label="Bearbeiten", menu=edit_menu)
        
        # Ansicht-Menü
        view_menu = Menu(menubar, tearoff=0, bg=self.theme.colors['secondary'], fg=self.theme.colors['text'])
        view_menu.add_checkbutton(label="Vollbild", command=self.toggle_fullscreen)
        menubar.add_cascade(label="Ansicht", menu=view_menu)
        
        # Hilfe-Menü
        help_menu = Menu(menubar, tearoff=0, bg=self.theme.colors['secondary'], fg=self.theme.colors['text'])
        help_menu.add_command(label="Über", command=self.show_about)
        help_menu.add_command(label="Hilfe", command=self.show_help)
        menubar.add_cascade(label="Hilfe", menu=help_menu)
        
        self.root.config(menu=menubar)
    
    def send_message(self):
        """Sendet eine Nachricht"""
        message = self.user_input.get("1.0", tk.END).strip()
        if not message:
            return
            
        # Nachricht anzeigen
        self.display_message("Benutzer", message)
        self.user_input.delete("1.0", tk.END)
        
        # Hier würde die KI-Antwort generiert werden
        self.display_message("J.A.R.V.I.S.", "Ich verarbeite Ihre Anfrage...")
    
    def display_message(self, sender, message):
        """Zeigt eine Nachricht im Chat an"""
        self.chat_display.config(state=tk.NORMAL)
        
        # Sender-Name
        self.chat_display.insert(tk.END, f"{sender}:\n", 'sender')
        
        # Nachricht
        self.chat_display.insert(tk.END, f"{message}\n\n")
        
        # Scrollen zum Ende
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def new_chat(self):
        """Startet einen neuen Chat"""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.config(state=tk.DISABLED)
        self.status_bar.config(text="Neuer Chat gestartet")
    
    def save_chat(self):
        """Speichert den aktuellen Chat"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Textdateien", "*.txt"), ("Alle Dateien", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(self.chat_display.get("1.0", tk.END))
                self.status_bar.config(text=f"Chat gespeichert: {file_path}")
            except Exception as e:
                messagebox.showerror("Fehler", f"Konnte Chat nicht speichern: {str(e)}")
    
    def load_chat(self):
        """Lädt einen gespeicherten Chat"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Textdateien", "*.txt"), ("Alle Dateien", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                self.chat_display.config(state=tk.NORMAL)
                self.chat_display.delete("1.0", tk.END)
                self.chat_display.insert(tk.END, content)
                self.chat_display.config(state=tk.DISABLED)
                
                self.status_bar.config(text=f"Chat geladen: {file_path}")
            except Exception as e:
                messagebox.showerror("Fehler", f"Konnte Chat nicht laden: {str(e)}")
    
    def cut_text(self):
        """Schneidet den ausgewählten Text aus"""
        self.user_input.event_generate("<<Cut>>")
    
    def copy_text(self):
        """Kopiert den ausgewählten Text"""
        self.user_input.event_generate("<<Copy>>")
    
    def paste_text(self):
        """Fügt Text an der Cursor-Position ein"""
        self.user_input.event_generate("<<Paste>>")
    
    def toggle_fullscreen(self):
        """Schaltet den Vollbildmodus um"""
        self.root.attributes('-fullscreen', not self.root.attributes('-fullscreen'))
    
    def show_about(self):
        """Zeigt den Info-Dialog an"""
        about_text = """J.A.R.V.I.S. - Just A Rather Very Intelligent System

Version 1.0

Eine moderne KI-Chat-Anwendung inspiriert von Tony Starks KI-Assistenten."""
        
        messagebox.showinfo("Über J.A.R.V.I.S.", about_text)
    
    def show_help(self):
        """Zeigt die Hilfe an"""
        help_text = """Tastenkürzel:
- Enter: Nachricht senden
- Strg+N: Neuer Chat
- Strg+S: Chat speichern
- Strg+O: Chat laden
- F11: Vollbildmodus umschalten"""
        
        messagebox.showinfo("Hilfe", help_text)

def main():
    root = tk.Tk()
    app = JarvisGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
