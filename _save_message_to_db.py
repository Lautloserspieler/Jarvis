    def _save_message_to_db(self, role: str, content: str, timestamp: str):
        """Speichert eine Nachricht in der Datenbank.

        Args:
            role (str): Die Rolle des Absenders ('user' oder 'assistant')
            content (str): Der Inhalt der Nachricht
            timestamp (str): Zeitstempel der Nachricht
        """
        # Aktiviere das Textfeld zum Bearbeiten
        self.chat_display.configure(state='normal')

        # Lösche die letzte Nachricht des Assistenten, falls vorhanden
        if role == 'assistant':
            self.chat_display.delete('end-2l', 'end')

        # Füge die Nachricht hinzu
        self._display_message(role, content)

        # Speichere die Nachricht in der Datenbank
        if self.current_conversation_id:
            self.database.save_message(
                self.current_conversation_id,
                role,
                content,
                timestamp
            )

        # Deaktiviere das Textfeld wieder
        self.chat_display.configure(state='disabled')
