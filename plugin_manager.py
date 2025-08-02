import importlib
import inspect
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Type, Callable, Union
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class PluginType(str, Enum):
    """Enum für verschiedene Plugin-Typen."""
    COMMAND = "command"
    EVENT = "event"
    INTEGRATION = "integration"
    TOOL = "tool"
    CUSTOM = "custom"

class PluginStatus(str, Enum):
    """Status eines Plugins."""
    LOADED = "loaded"
    ERROR = "error"
    DISABLED = "disabled"
    LOADING = "loading"

@dataclass
class PluginInfo:
    """Metadaten über ein Plugin."""
    name: str
    version: str = "1.0.0"
    author: str = ""
    description: str = ""
    plugin_type: PluginType = PluginType.CUSTOM
    dependencies: List[str] = None
    settings_schema: Dict = None
    icon: str = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.settings_schema is None:
            self.settings_schema = {}

@dataclass
class Plugin:
    """Repräsentiert ein geladenes Plugin."""
    name: str
    module: Any
    info: PluginInfo
    status: PluginStatus = PluginStatus.LOADING
    error: Optional[str] = None
    settings: Dict = None
    
    def __post_init__(self):
        if self.settings is None:
            self.settings = {}

class PluginManager:
    """Verwaltet das Laden und Ausführen von Plugins."""
    
    def __init__(self, config):
        """Initialisiert den PluginManager.
        
        Args:
            config: Eine ConfigManager-Instanz
        """
        self.config = config
        self.plugins: Dict[str, Plugin] = {}
        self.plugin_dirs = self._get_plugin_dirs()
        self.event_handlers = {}
        
        # Erstelle das Plugin-Verzeichnis, falls es nicht existiert
        for plugin_dir in self.plugin_dirs:
            plugin_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_plugin_dirs(self) -> List[Path]:
        """Gibt eine Liste von Plugin-Verzeichnissen zurück.
        
        Returns:
            Eine Liste von Path-Objekten zu den Plugin-Verzeichnissen
        """
        # Standard-Plugin-Verzeichnisse
        dirs = [
            Path("plugins"),  # Im Arbeitsverzeichnis
            Path.home() / ".config" / "jarvis" / "plugins",  # Benutzer-spezifisch
            Path("/etc/jarvis/plugins")  # Systemweit
        ]
        
        # Benutzerdefinierte Verzeichnisse aus der Konfiguration hinzufügen
        if self.config.has_section('PLUGINS') and 'plugin_dirs' in self.config['PLUGINS']:
            custom_dirs = self.config['PLUGINS']['plugin_dirs'].split(':')
            dirs.extend([Path(d) for d in custom_dirs])
        
        return [d for d in dirs if d.exists() or d == Path("plugins")]
    
    def discover_plugins(self) -> List[str]:
        """Durchsucht die Plugin-Verzeichnisse nach verfügbaren Plugins.
        
        Returns:
            Eine Liste von gefundenen Plugin-Namen
        """
        plugin_names = set()
        
        for plugin_dir in self.plugin_dirs:
            if not plugin_dir.exists():
                continue
                
            for item in plugin_dir.iterdir():
                # Überprüfe auf Python-Module
                if item.is_file() and item.suffix == '.py' and not item.name.startswith('_'):
                    plugin_name = item.stem
                    plugin_names.add(plugin_name)
                # Oder Verzeichnisse mit __init__.py
                elif item.is_dir() and (item / '__init__.py').exists():
                    plugin_name = item.name
                    plugin_names.add(plugin_name)
        
        return sorted(plugin_names)
    
    def load_plugin(self, plugin_name: str) -> Optional[Plugin]:
        """Lädt ein Plugin.
        
        Args:
            plugin_name: Der Name des Plugins
            
        Returns:
            Das geladene Plugin oder None bei Fehlern
        """
        # Prüfe, ob das Plugin bereits geladen ist
        if plugin_name in self.plugins:
            return self.plugins[plugin_name]
        
        # Suche nach dem Plugin in den Verzeichnissen
        plugin_path = None
        for plugin_dir in self.plugin_dirs:
            # Prüfe auf Python-Datei
            py_file = plugin_dir / f"{plugin_name}.py"
            if py_file.exists():
                plugin_path = py_file
                break
                
            # Oder Verzeichnis mit __init__.py
            pkg_dir = plugin_dir / plugin_name
            if pkg_dir.exists() and (pkg_dir / '__init__.py').exists():
                plugin_path = pkg_dir
                break
        
        if not plugin_path:
            logger.error(f"Plugin {plugin_name} nicht gefunden")
            return None
        
        # Erstelle einen eindeutigen Modulnamen
        module_name = f"plugins.{plugin_name}"
        
        try:
            # Füge das Plugin-Verzeichnis zum Python-Pfad hinzu
            if str(plugin_path.parent) not in sys.path:
                sys.path.insert(0, str(plugin_path.parent))
            
            # Importiere das Modul
            if plugin_path.is_file():
                spec = importlib.util.spec_from_file_location(module_name, plugin_path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
            else:
                module = importlib.import_module(f"{module_name}")
            
            # Erstelle Plugin-Info
            plugin_info = self._extract_plugin_info(module, plugin_name)
            
            # Erstelle das Plugin-Objekt
            plugin = Plugin(
                name=plugin_name,
                module=module,
                info=plugin_info,
                status=PluginStatus.LOADED
            )
            
            # Rufe die setup-Funktion auf, falls vorhanden
            if hasattr(module, 'setup'):
                try:
                    setup_result = module.setup(plugin)
                    if setup_result is False:
                        raise RuntimeError("Setup-Funktion hat False zurückgegeben")
                except Exception as e:
                    raise RuntimeError(f"Fehler beim Setup von {plugin_name}: {e}")
            
            # Registriere Event-Handler
            self._register_event_handlers(plugin)
            
            # Speichere das Plugin
            self.plugins[plugin_name] = plugin
            logger.info(f"Plugin {plugin_name} erfolgreich geladen")
            
            return plugin
            
        except Exception as e:
            logger.error(f"Fehler beim Laden des Plugins {plugin_name}: {e}", exc_info=True)
            
            # Erstelle ein Plugin-Objekt mit Fehlerstatus
            plugin = Plugin(
                name=plugin_name,
                module=None,
                info=PluginInfo(name=plugin_name, description=f"Fehler: {str(e)}"),
                status=PluginStatus.ERROR,
                error=str(e)
            )
            
            self.plugins[plugin_name] = plugin
            return None
    
    def _extract_plugin_info(self, module: Any, plugin_name: str) -> PluginInfo:
        """Extrahiert Plugin-Metadaten aus einem Modul.
        
        Args:
            module: Das importierte Modul
            plugin_name: Der Name des Plugins
            
        Returns:
            Ein PluginInfo-Objekt mit den extrahierten Metadaten
        """
        # Standardwerte
        info = PluginInfo(
            name=plugin_name,
            version=getattr(module, '__version__', '1.0.0'),
            author=getattr(module, '__author__', ''),
            description=getattr(module, '__doc__', '').strip() if module.__doc__ else '',
            plugin_type=getattr(module, 'PLUGIN_TYPE', PluginType.CUSTOM),
            dependencies=getattr(module, 'DEPENDENCIES', []),
            settings_schema=getattr(module, 'SETTINGS_SCHEMA', {})
        )
        
        return info
    
    def _register_event_handlers(self, plugin: Plugin) -> None:
        """Registriert Event-Handler aus einem Plugin.
        
        Args:
            plugin: Das Plugin, aus dem die Handler registriert werden sollen
        """
        if not hasattr(plugin.module, 'register_handlers'):
            return
        
        try:
            # Rufe die register_handlers-Funktion auf
            handlers = plugin.module.register_handlers()
            
            if not isinstance(handlers, dict):
                logger.warning(f"Ungültige Handler-Registrierung in {plugin.name}")
                return
            
            # Registriere jeden Handler
            for event_name, handler in handlers.items():
                if not callable(handler):
                    logger.warning(f"Handler für {event_name} in {plugin.name} ist nicht aufrufbar")
                    continue
                
                if event_name not in self.event_handlers:
                    self.event_handlers[event_name] = []
                
                self.event_handlers[event_name].append((plugin.name, handler))
                logger.debug(f"Handler für {event_name} aus {plugin.name} registriert")
                
        except Exception as e:
            logger.error(f"Fehler beim Registrieren der Handler für {plugin.name}: {e}")
    
    def load_all_plugins(self) -> List[str]:
        """Lädt alle verfügbaren Plugins.
        
        Returns:
            Eine Liste der Namen der erfolgreich geladenen Plugins
        """
        plugin_names = self.discover_plugins()
        loaded = []
        
        for plugin_name in plugin_names:
            plugin = self.load_plugin(plugin_name)
            if plugin and plugin.status == PluginStatus.LOADED:
                loaded.append(plugin_name)
        
        return loaded
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """Entlädt ein Plugin.
        
        Args:
            plugin_name: Der Name des Plugins
            
        Returns:
            True, wenn das Plugin erfolgreich entladen wurde, sonst False
        """
        if plugin_name not in self.plugins:
            logger.warning(f"Kann Plugin {plugin_name} nicht entladen: Nicht geladen")
            return False
        
        plugin = self.plugins[plugin_name]
        
        # Rufe die cleanup-Funktion auf, falls vorhanden
        if hasattr(plugin.module, 'cleanup'):
            try:
                plugin.module.cleanup()
            except Exception as e:
                logger.error(f"Fehler beim Aufräumen von Plugin {plugin_name}: {e}")
        
        # Entferne Event-Handler
        self._unregister_event_handlers(plugin_name)
        
        # Entferne das Modul aus sys.modules
        module_name = f"plugins.{plugin_name}"
        if module_name in sys.modules:
            del sys.modules[module_name]
        
        # Entferne das Plugin aus der Verwaltung
        del self.plugins[plugin_name]
        
        logger.info(f"Plugin {plugin_name} wurde entladen")
        return True
    
    def _unregister_event_handlers(self, plugin_name: str) -> None:
        """Entfernt alle Event-Handler eines Plugins.
        
        Args:
            plugin_name: Der Name des Plugins
        """
        for event_name in list(self.event_handlers.keys()):
            # Filtere die Handler des Plugins heraus
            self.event_handlers[event_name] = [
                (p, h) for p, h in self.event_handlers[event_name] 
                if p != plugin_name
            ]
            
            # Entferne leere Event-Listen
            if not self.event_handlers[event_name]:
                del self.event_handlers[event_name]
    
    def unload_all_plugins(self) -> None:
        """Entlädt alle geladenen Plugins."""
        for plugin_name in list(self.plugins.keys()):
            self.unload_plugin(plugin_name)
    
    def reload_plugin(self, plugin_name: str) -> Optional[Plugin]:
        """Lädt ein Plugin neu.
        
        Args:
            plugin_name: Der Name des Plugins
            
        Returns:
            Das neu geladene Plugin oder None bei Fehlern
        """
        if plugin_name in self.plugins:
            self.unload_plugin(plugin_name)
        
        return self.load_plugin(plugin_name)
    
    def get_plugin(self, plugin_name: str) -> Optional[Plugin]:
        """Gibt ein geladenes Plugin zurück.
        
        Args:
            plugin_name: Der Name des Plugins
            
        Returns:
            Das Plugin oder None, wenn nicht gefunden
        """
        return self.plugins.get(plugin_name)
    
    def get_plugins_by_type(self, plugin_type: Union[PluginType, str]) -> List[Plugin]:
        """Gibt alle Plugins eines bestimmten Typs zurück.
        
        Args:
            plugin_type: Der Plugin-Typ (als String oder PluginType-Enum)
            
        Returns:
            Eine Liste von Plugins des angegebenen Typs
        """
        if isinstance(plugin_type, str):
            plugin_type = PluginType(plugin_type.lower())
            
        return [p for p in self.plugins.values() if p.info.plugin_type == plugin_type]
    
    def emit_event(self, event_name: str, *args, **kwargs) -> List[Any]:
        """Löst ein Event aus und ruft alle registrierten Handler auf.
        
        Args:
            event_name: Der Name des Events
            *args: Positionsargumente für die Handler
            **kwargs: Schlüsselwortargumente für die Handler
            
        Returns:
            Eine Liste der Rückgabewerte der Handler (in der Reihenfolge der Ausführung)
        """
        results = []
        
        if event_name not in self.event_handlers:
            return results
        
        for plugin_name, handler in self.event_handlers[event_name]:
            try:
                result = handler(*args, **kwargs)
                results.append(result)
                logger.debug(f"Event {event_name} von {plugin_name} verarbeitet")
            except Exception as e:
                logger.error(f"Fehler im Event-Handler {event_name} von {plugin_name}: {e}", exc_info=True)
        
        return results
    
    def get_plugin_status(self) -> Dict[str, Dict[str, Any]]:
        """Gibt den Status aller Plugins zurück.
        
        Returns:
            Ein Dictionary mit Plugin-Namen als Schlüssel und Status-Informationen als Werte
        """
        status = {}
        
        for name, plugin in self.plugins.items():
            status[name] = {
                'status': plugin.status.value,
                'type': plugin.info.plugin_type.value,
                'version': plugin.info.version,
                'error': plugin.error,
                'description': plugin.info.description,
                'dependencies': plugin.info.dependencies,
                'settings': plugin.settings
            }
        
        return status
    
    def install_plugin(self, source: str, plugin_name: str = None) -> Optional[Plugin]:
        """Installiert ein Plugin aus einer Quelle.
        
        Args:
            source: Die Quelle des Plugins (Pfad, URL, Paketname)
            plugin_name: Optional: Der gewünschte Name des Plugins
            
        Returns:
            Das installierte Plugin oder None bei Fehlern
        """
        # TODO: Implementiere die Plugin-Installation
        # Dies könnte das Herunterladen von einer URL, das Klonen eines Git-Repos,
        # das Installieren eines Python-Pakets usw. umfassen
        raise NotImplementedError("Die Plugin-Installation ist noch nicht implementiert")
    
    def uninstall_plugin(self, plugin_name: str) -> bool:
        """Deinstalliert ein Plugin.
        
        Args:
            plugin_name: Der Name des zu deinstallierenden Plugins
            
        Returns:
            True, wenn das Plugin erfolgreich deinstalliert wurde, sonst False
        """
        # TODO: Implementiere die Plugin-Deinstallation
        # Dies könnte das Entfernen von Dateien, das Aufräumen von Abhängigkeiten usw. umfassen
        raise NotImplementedError("Die Plugin-Deinstallation ist noch nicht implementiert")
