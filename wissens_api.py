import json
import logging
import threading
from typing import Dict, List, Optional, Any, Union
import requests
import wikipediaapi
import isbnlib
from urllib.parse import quote, unquote
from SPARQLWrapper import SPARQLWrapper, JSON
import re

logger = logging.getLogger(__name__)

class WissensAPI:
    """
    Eine zentrale Klasse für den Zugriff auf verschiedene Wissensquellen wie Wikipedia und Open Library.
    """
    
    def __init__(self, sprache: str = 'de'):
        """
        Initialisiert die Wissens-API.
        
        Args:
            sprache: Die Sprache für die Suchergebnisse (z.B. 'de' für Deutsch, 'en' für Englisch)
        """
        self.sprache = sprache
        self.timeout = 10  # Timeout in Sekunden für API-Aufrufe
        
        # Wikipedia-API mit Timeout konfigurieren
        self.wiki_wiki = wikipediaapi.Wikipedia(
            language=sprache,
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            user_agent='JarvisKI/1.0 (your@email.com)',
            timeout=self.timeout
        )
        
        self.cache = {}
        
        # Requests-Session mit Timeout konfigurieren
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'JarvisKI/1.0 (your@email.com)',
            'Accept': 'application/json'
        })
        
        # Timeout für alle Requests in der Session setzen
        self.session.request = lambda method, url, **kwargs: (
            self.session.original_request(
                method, 
                url, 
                timeout=self.timeout, 
                **{k: v for k, v in kwargs.items() if k != 'timeout'}
            )
        )
        self.session.original_request = self.session.request
        
        # Cache für häufig abgerufene Informationen
        self.cache: Dict[str, Any] = {}
        
        # SPARQL-Endpunkte
        self.dbpedia_endpoint = "http://dbpedia.org/sparql"
        self.conceptnet_endpoint = "http://api.conceptnet.io"
    
    def suche_wikipedia(self, suchbegriff: str, zusaetzliche_info: bool = False) -> Dict[str, Any]:
        """
        Sucht nach einem Begriff in der Wikipedia.
        
        Args:
            suchbegriff: Der zu suchende Begriff
            zusaetzliche_info: Wenn True, werden zusätzliche Informationen abgerufen
            
        Returns:
            Ein Dictionary mit den Suchergebnissen
        """
        cache_key = f"wiki_{suchbegriff}_{zusaetzliche_info}"
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        ergebnis = {
            'erfolg': False,
            'titel': '',
            'zusammenfassung': '',
            'url': '',
            'kategorien': [],
            'zusaetzliche_infos': {},
            'fehler': None
        }
        
        def fetch_wiki():
            try:
                # Suche nach der Seite mit Timeout
                seite = self.wiki_wiki.page(suchbegriff)
                
                if seite.exists():
                    ergebnis.update({
                        'erfolg': True,
                        'titel': seite.title,
                        'zusammenfassung': seite.summary[:500] + '...' if len(seite.summary) > 500 else seite.summary,
                        'url': seite.fullurl,
                        'kategorien': list(seite.categories.keys())[:5]  # Erste 5 Kategorien
                    })
                    
                    # Zusätzliche Informationen abrufen, wenn gewünscht
                    if zusaetzliche_info:
                        ergebnis['zusaetzliche_infos'] = {
                            'seiten_id': seite.pageid,
                            'letzte_aenderung': seite.last_rev_timestamp,
                            'laenge_text': len(seite.text)
                        }
                else:
                    ergebnis['fehler'] = f"Keine Ergebnisse für '{suchbegriff}' gefunden"
            
            except requests.exceptions.Timeout:
                ergebnis['fehler'] = "Zeitüberschreitung bei der Wikipedia-Anfrage"
            except Exception as e:
                ergebnis['fehler'] = f"Fehler bei der Wikipedia-Suche: {str(e)}"
        
        # Führe die Abfrage in einem Thread mit Timeout aus
        thread = threading.Thread(target=fetch_wiki)
        thread.daemon = True
        thread.start()
        thread.join(timeout=self.timeout)  # Warte maximal self.timeout Sekunden
        
        if thread.is_alive():
            # Thread läuft noch -> Timeout
            ergebnis['fehler'] = "Zeitüberschreitung bei der Anfrage an die Wikipedia-API"
        
        if ergebnis.get('fehler'):
            logger.error(f"Fehler bei der Wikipedia-Suche nach '{suchbegriff}': {ergebnis['fehler']}")
        
        self.cache[cache_key] = ergebnis
        return ergebnis
    
    def suche_buch_nach_isbn(self, isbn: str) -> Dict[str, Any]:
        """
        Sucht nach einem Buch anhand seiner ISBN-Nummer.
        
        Args:
            isbn: Die ISBN-Nummer des Buches
            
        Returns:
            Ein Dictionary mit den Buchinformationen
        """
        cache_key = f"book_{isbn}"
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        ergebnis = {
            'erfolg': False,
            'titel': '',
            'fehler': None,
            'autoren': [],
            'veroeffentlichung': '',
            'beschreibung': '',
            'cover_url': ''
        }
        
        try:
            # ISBN bereinigen und validieren
            isbn = isbnlib.to_isbn13(isbnlib.clean(isbn))
            if not isbnlib.is_isbn13(isbn):
                raise ValueError("Ungültige ISBN-Nummer")
            
            # Suche nach der ISBN in der Open Library API mit Timeout
            url = f"https://openlibrary.org/api/books?bibkeys=ISBN:{isbn}&format=json&jscmd=data"
            
            def fetch_book():
                try:
                    response = self.session.get(url, timeout=self.timeout)
                    response.raise_for_status()
                    daten = response.json()
                    buch_key = f"ISBN:{isbn}"
                    
                    if buch_key in daten:
                        buch = daten[buch_key]
                        ergebnis.update({
                            'erfolg': True,
                            'titel': buch.get('title', ''),
                            'autoren': [autor['name'] for autor in buch.get('authors', [])],
                            'veroeffentlichung': buch.get('publish_date', ''),
                            'verlag': buch.get('publishers', [{}])[0].get('name', ''),
                            'seiten': buch.get('number_of_pages', 0),
                            'beschreibung': buch.get('subtitle', ''),
                            'cover_url': buch.get('cover', {}).get('large', '')
                        })
                    else:
                        ergebnis['fehler'] = f"Keine Ergebnisse für ISBN {isbn} gefunden"
                        
                except requests.exceptions.Timeout:
                    ergebnis['fehler'] = "Zeitüberschreitung bei der Buchsuche"
                except requests.exceptions.RequestException as e:
                    ergebnis['fehler'] = f"Netzwerkfehler bei der Buchsuche: {str(e)}"
                except Exception as e:
                    ergebnis['fehler'] = f"Fehler bei der Buchsuche: {str(e)}"
            
            # Führe die Abfrage in einem Thread mit Timeout aus
            thread = threading.Thread(target=fetch_book)
            thread.daemon = True
            thread.start()
            thread.join(timeout=self.timeout)  # Warte maximal self.timeout Sekunden
            
            if thread.is_alive():
                # Thread läuft noch -> Timeout
                ergebnis['fehler'] = "Zeitüberschreitung bei der Buchsuche"
            
            if ergebnis.get('fehler'):
                logger.error(f"Fehler bei der Buchsuche nach ISBN {isbn}: {ergebnis['fehler']}")
        
        except Exception as e:
            logger.error(f"Fehler bei der Buchsuche nach ISBN {isbn}: {e}")
            ergebnis['fehler'] = str(e)
        
        self.cache[cache_key] = ergebnis
        return ergebnis
    
    def suche_buecher_nach_titel(self, titel: str, limit: int = 3) -> Dict[str, Any]:
        """
        Sucht nach Büchern anhand des Titels.
        
        Args:
            titel: Der Titel des gesuchten Buches
            limit: Maximale Anzahl der Ergebnisse
            
        Returns:
            Ein Dictionary mit den Suchergebnissen
        """
        cache_key = f"book_search_{titel}_{limit}"
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        ergebnis = {
            'erfolg': False,
            'ergebnisse': [],
            'anzahl': 0
        }
        
        try:
            # URL-kodierten Titel erstellen
            kodierter_titel = quote(titel)
            
            # Suche nach Büchern mit dem Titel
            url = f"{self.open_library_url}/search.json?title={kodierter_titel}&limit={limit}"
            antwort = requests.get(url, headers={'User-Agent': self.benutzer_agent}, timeout=10)
            antwort.raise_for_status()
            
            suchergebnis = antwort.json()
            buecher = []
            
            for buch in suchergebnis.get('docs', [])[:limit]:
                # Autoren extrahieren
                autoren = buch.get('author_name', [])
                if not isinstance(autoren, list):
                    autoren = [autoren] if autoren else []
                
                # Cover-URL erstellen
                cover_id = buch.get('cover_i')
                cover_url = f"https://covers.openlibrary.org/b/id/{cover_id}-M.jpg" if cover_id else None
                
                buch_info = {
                    'titel': buch.get('title', 'Unbekannter Titel'),
                    'autoren': autoren,
                    'erscheinungsjahr': buch.get('first_publish_year', 'Unbekannt'),
                    'isbn': buch.get('isbn', [''])[0] if buch.get('isbn') else '',
                    'cover_url': cover_url,
                    'schluessel': buch.get('key', '')
                }
                buecher.append(buch_info)
            
            ergebnis.update({
                'erfolg': True,
                'ergebnisse': buecher,
                'anzahl': len(buecher)
            })
            
        except Exception as e:
            logger.error(f"Fehler bei der Buchsuche nach Titel '{titel}': {e}")
            ergebnis['fehler'] = str(e)
        
        self.cache[cache_key] = ergebnis
        return ergebnis
    
    def hole_zufaelligen_artikel(self) -> Dict[str, Any]:
        """
        Gibt einen zufälligen Wikipedia-Artikel zurück.
        
        Returns:
            Ein Dictionary mit den Artikeldaten
        """
        try:
            zufaelliger_titel = self.wiki_wiki.random(pages=1)
            if zufaelliger_titel:
                return self.suche_wikipedia(zufaelliger_titel)
        except Exception as e:
            logger.error(f"Fehler beim Abrufen eines zufälligen Artikels: {e}")
        
        return {'erfolg': False, 'fehler': 'Konnte keinen zufälligen Artikel laden'}
    
    # === DBpedia Methoden ===
    
    def _frage_dbpedia_ab(self, sparql_query: str) -> Dict:
        """Führt eine SPARQL-Abfrage gegen DBpedia aus."""
        try:
            sparql = SPARQLWrapper(self.dbpedia_endpoint)
            sparql.setQuery(sparql_query)
            sparql.setReturnFormat(JSON)
            ergebnis = sparql.query().convert()
            return ergebnis
        except Exception as e:
            logger.error(f"Fehler bei der DBpedia-Abfrage: {e}")
            return {}
    
    def hole_personendaten(self, name: str) -> Dict[str, Any]:
        """Holt strukturierte Informationen über eine Person von DBpedia."""
        cache_key = f"dbpedia_person_{name}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        ergebnis = {
            'erfolg': False,
            'name': name,
            'beschreibung': '',
            'beruf': [],
            'geburtsort': '',
            'geburtsdatum': '',
            'organisationen': [],
            'url': f"http://de.dbpedia.org/resource/{name.replace(' ', '_')}"
        }
        
        try:
            # SPARQL-Abfrage für Personeninformationen
            query = f"""
            PREFIX dbo: <http://dbpedia.org/ontology/>
            PREFIX dbp: <http://de.dbpedia.org/property/>
            PREFIX rdfs: <http://www.w3.org/2000/2001/rdf-schema#>
            
            SELECT DISTINCT ?abstract ?beruf ?berufLabel ?geburtsort ?geburtsortLabel ?geburtsdatum ?organisation ?orgLabel
            WHERE {{
              ?person a dbo:Person ;
                      rdfs:label ?label .
              FILTER(LANG(?label) = 'de' || LANG(?label) = 'en')
              FILTER(CONTAINS(LCASE(STR(?label)), LCASE(""" + name + """)))
              
              OPTIONAL {{ ?person dbo:abstract ?abstract .
                         FILTER(LANG(?abstract) = 'de') }}
                          
              OPTIONAL {{ ?person dbo:occupation ?beruf .
                         ?beruf rdfs:label ?berufLabel .
                         FILTER(LANG(?berufLabel) = 'de') }}
                          
              OPTIONAL {{ ?person dbo:birthPlace ?geburtsort .
                         ?geburtsort rdfs:label ?geburtsortLabel .
                         FILTER(LANG(?geburtsortLabel) = 'de') }}
                          
              OPTIONAL {{ ?person dbo:birthDate ?geburtsdatum }}
              
              OPTIONAL {{ ?person dbo:organisation ?organisation .
                         ?organisation rdfs:label ?orgLabel .
                         FILTER(LANG(?orgLabel) = 'de') }}
            }}
            LIMIT 10
            """
            
            daten = self._frage_dbpedia_ab(query)
            
            if 'results' in daten and 'bindings' in daten['results']:
                bindings = daten['results']['bindings']
                
                if bindings:
                    ergebnis['erfolg'] = True
                    
                    # Einzigartige Werte sammeln
                    berufe = set()
                    organisationen = set()
                    
                    for binding in bindings:
                        if 'abstract' in binding and not ergebnis['beschreibung']:
                            ergebnis['beschreibung'] = binding['abstract']['value']
                        
                        if 'berufLabel' in binding:
                            beruf = binding['berufLabel']['value']
                            berufe.add(beruf)
                            
                        if 'geburtsortLabel' in binding and not ergebnis['geburtsort']:
                            ergebnis['geburtsort'] = binding['geburtsortLabel']['value']
                            
                        if 'geburtsdatum' in binding and not ergebnis['geburtsdatum']:
                            ergebnis['geburtsdatum'] = binding['geburtsdatum']['value']
                            
                        if 'orgLabel' in binding:
                            org = binding['orgLabel']['value']
                            organisationen.add(org)
                    
                    ergebnis['beruf'] = list(berufe)
                    ergebnis['organisationen'] = list(organisationen)
        
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Personendaten: {e}")
        
        self.cache[cache_key] = ergebnis
        return ergebnis
    
    def hole_ortsinformationen(self, ortsname: str) -> Dict[str, Any]:
        """Holt strukturierte Informationen über einen Ort von DBpedia."""
        cache_key = f"dbpedia_ort_{ortsname}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        ergebnis = {
            'erfolg': False,
            'name': ortsname,
            'beschreibung': '',
            'land': '',
            'einwohner': '',
            'flaeche': '',
            'url': f"http://de.dbpedia.org/resource/{ortsname.replace(' ', '_')}"
        }
        
        try:
            # SPARQL-Abfrage für Ortsinformationen
            query = f"""
            PREFIX dbo: <http://dbpedia.org/ontology/>
            PREFIX dbp: <http://de.dbpedia.org/property/>
            PREFIX rdfs: <http://www.w3.org/2000/2001/rdf-schema#>
            
            SELECT DISTINCT ?abstract ?land ?landLabel ?einwohner ?flaeche
            WHERE {{
              ?ort a dbo:Place ;
                   rdfs:label ?label .
              FILTER(LANG(?label) = 'de' || LANG(?label) = 'en')
              FILTER(CONTAINS(LCASE(STR(?label)), LCASE(""" + ortsname + """)))
              
              OPTIONAL {{ ?ort dbo:abstract ?abstract .
                         FILTER(LANG(?abstract) = 'de') }}
              
              OPTIONAL {{ ?ort dbo:country ?land .
                         ?land rdfs:label ?landLabel .
                         FILTER(LANG(?landLabel) = 'de') }}
              
              OPTIONAL {{ ?ort dbo:populationTotal ?einwohner }}
              OPTIONAL {{ ?ort dbo:areaTotal ?flaeche }}
            }}
            LIMIT 1
            """
            
            daten = self._frage_dbpedia_ab(query)
            
            if 'results' in daten and 'bindings' in daten['results'] and daten['results']['bindings']:
                binding = daten['results']['bindings'][0]
                ergebnis['erfolg'] = True
                
                if 'abstract' in binding:
                    ergebnis['beschreibung'] = binding['abstract']['value']
                
                if 'landLabel' in binding:
                    ergebnis['land'] = binding['landLabel']['value']
                
                if 'einwohner' in binding:
                    ergebnis['einwohner'] = binding['einwohner']['value']
                
                if 'flaeche' in binding:
                    ergebnis['flaeche'] = f"{binding['flaeche']['value']} km²"
        
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Ortsinformationen: {e}")
        
        self.cache[cache_key] = ergebnis
        return ergebnis
    
    # === ConceptNet Methoden ===
    
    def hole_konzepte(self, begriff: str, limit: int = 5) -> Dict[str, Any]:
        """Holt verwandte Konzepte von ConceptNet."""
        cache_key = f"conceptnet_{begriff}_{limit}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        ergebnis = {
            'erfolg': False,
            'begriff': begriff,
            'beziehungen': [],
            'verwandte_konzepte': []
        }
        
        try:
            # API-Anfrage an ConceptNet
            url = f"{self.conceptnet_endpoint}/c/de/{quote(begriff)}"
            antwort = requests.get(url, 
                                 headers={'Accept': 'application/json', 'User-Agent': self.benutzer_agent},
                                 params={'limit': limit})
            
            if antwort.status_code == 200:
                daten = antwort.json()
                ergebnis['erfolg'] = True
                
                # Extrahiere Beziehungen
                for edge in daten.get('edges', []):
                    if 'rel' in edge and 'surfaceText' in edge and edge.get('surfaceText'):
                        beziehung = {
                            'text': edge['surfaceText'].replace('[[', '').replace(']]', ''),
                            'gewicht': edge.get('weight', 0),
                            'relation': edge['rel'].get('label', '')
                        }
                        ergebnis['beziehungen'].append(beziehung)
                
                # Extrahiere verwandte Konzepte
                for edge in daten.get('edges', []):
                    if 'end' in edge and 'label' in edge['end'] and edge['end'].get('language') == 'de':
                        konzept = edge['end']['label']
                        if konzept.lower() != begriff.lower() and konzept not in ergebnis['verwandte_konzepte']:
                            ergebnis['verwandte_konzepte'].append(konzept)
                
                # Sortiere nach Gewichtung
                ergebnis['beziehungen'].sort(key=lambda x: x['gewicht'], reverse=True)
                ergebnis['verwandte_konzepte'] = list(set(ergebnis['verwandte_konzepte']))[:limit]
        
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der ConceptNet-Daten: {e}")
        
        self.cache[cache_key] = ergebnis
        return ergebnis
    
    def formatiere_antwort(self, daten: Dict[str, Any], quelle: str) -> str:
        """
        Formatiert die API-Antwort in eine lesbare Textform.
        
        Args:
            daten: Die zu formatierenden Daten
            quelle: Die Datenquelle (z.B. 'wikipedia', 'openlibrary', 'dbpedia_person', 'dbpedia_ort', 'conceptnet')
            
        Returns:
            Formatierter Text
        """
        if not daten.get('erfolg'):
            return "Entschuldigung, ich konnte dazu leider keine Informationen finden."
        
        if quelle == 'wikipedia':
            antwort = f"🔍 **{daten['titel']}**\n\n"
            antwort += f"{daten['zusammenfassung']}\n\n"
            if 'kategorien' in daten and daten['kategorien']:
                kategorien = ", ".join([k.replace("Kategorie:", "") for k in daten['kategorien']])
                antwort += f"Kategorien: {kategorien}\n"
            antwort += f"\nQuelle: [Wikipedia]({daten['url']})"
            return antwort
            
        elif quelle == 'openlibrary' and 'ergebnisse' in daten:
            if daten['anzahl'] == 0:
                return "Keine passenden Bücher gefunden."
                
            antwort = f"📚 **Suchergebnisse für Bücher**\n\n"
            
            for i, buch in enumerate(daten['ergebnisse'], 1):
                autoren = ", ".join(buch.get('autoren', ['Unbekannter Autor']))
                erscheinungsjahr = buch.get('erscheinungsjahr', 'Unbekanntes Jahr')
                
                antwort += f"{i}. **{buch['titel']}**\n"
                antwort += f"   von {autoren} ({erscheinungsjahr})\n"
                
                if buch.get('cover_url'):
                    antwort += f"   [Bild]({buch['cover_url']})\n"
                
                if buch.get('isbn'):
                    antwort += f"   ISBN: {buch['isbn']}\n"
                
                antwort += "\n"
                
            return antwort
            
        elif quelle == 'openlibrary' and 'titel' in daten:
            autoren = ", ".join(daten.get('autoren', ['Unbekannter Autor']))
            
            antwort = f"📖 **{daten['titel']}**\n"
            antwort += f"von {autoren}\n\n"
            antwort += f"{daten.get('beschreibung', 'Keine Beschreibung verfügbar.')}\n\n"
            
            if daten.get('veroeffentlichung'):
                antwort += f"Erscheinungsjahr: {daten['veroeffentlichung']}\n"
                
            if daten.get('cover_url'):
                antwort += f"\n[Cover anzeigen]({daten['cover_url']})\n"
                
            if daten.get('url'):
                antwort += f"\n[Mehr auf Open Library]({daten['url']})"
                
            return antwort
                
        elif quelle == 'dbpedia_person':
            antwort = f"👤 **{daten['name']}**\n\n"
            
            if daten.get('beschreibung'):
                antwort += f"{daten['beschreibung']}\n\n"
            
            if daten.get('beruf'):
                berufe = ", ".join(daten['beruf'])
                antwort += f"**Beruf(e):** {berufe}\n"
            
            if daten.get('geburtsort'):
                antwort += f"**Geburtsort:** {daten['geburtsort']}"
                if daten.get('geburtsdatum'):
                    antwort += f" ({daten['geburtsdatum'].split('T')[0]})"
                antwort += "\n"
            
            if daten.get('organisationen'):
                orgs = ", ".join(daten['organisationen'])
                antwort += f"**Organisationen:** {orgs}\n"
            
            antwort += f"\n[Mehr auf DBpedia]({daten['url']})"
            return antwort
            
        elif quelle == 'dbpedia_ort':
            antwort = f"🏙️ **{daten['name']}**\n\n"
            
            if daten.get('beschreibung'):
                antwort += f"{daten['beschreibung']}\n\n"
            
            if daten.get('land'):
                antwort += f"**Land:** {daten['land']}\n"
            
            if daten.get('einwohner'):
                antwort += f"**Einwohner:** {int(float(daten['einwohner'])):,} (Stand: 2023)\n".replace(',', '.')
            
            if daten.get('flaeche'):
                antwort += f"**Fläche:** {daten['flaeche']}\n"
            
            antwort += f"\n[Mehr auf DBpedia]({daten['url']})"
            return antwort
            
        elif quelle == 'conceptnet':
            antwort = f"🧠 **Konzepte zu: {daten['begriff']}**\n\n"
            
            if daten.get('beziehungen'):
                antwort += "**Zusammenhänge:**\n"
                for beziehung in daten['beziehungen'][:5]:  # Zeige die 5 stärksten Beziehungen
                    antwort += f"- {beziehung['text']} (Gewicht: {beziehung['gewicht']:.2f})\n"
            
            if daten.get('verwandte_konzepte'):
                antwort += "\n**Verwandte Konzepte:**\n"
                for konzept in daten['verwandte_konzepte']:
                    antwort += f"- {konzept}\n"
            
            return antwort
                
        return "Die Antwort konnte nicht formatiert werden."
