from typing import Dict, Type, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class Registry:
    """Generic registry for models, relations, etc. supporting metadata and variants."""
    
    def __init__(self):
        # Store as list of candidates: {name: [{'obj': obj, 'meta': meta}, ...]}
        self._registry: Dict[str, List[Dict[str, Any]]] = {}

    def register(self, name: str, **metadata):
        """Decorator to register a class or function with optional metadata."""
        def decorator(obj):
            if name not in self._registry:
                self._registry[name] = []
            
            # Check for duplicates? or just append?
            # Append allows variants.
            entry = {'obj': obj, 'meta': metadata}
            self._registry[name].append(entry)
            logger.debug(f"Registered {name} with metadata {metadata}")
            return obj
        return decorator

    def get(self, name: str, **filters) -> Optional[Any]:
        """
        Get object by name, optionally filtering by metadata.
        Example: get("lorentzian", nfreqs=3)
        """
        candidates = self._registry.get(name, [])
        if not candidates:
            return None
        
        matches = []
        for cand in candidates:
            meta = cand['meta']
            # Check if all filters match metadata
            # If filter key not in meta, mismatch? 
            # Or if meta has key and filter matches.
            is_match = True
            for k, v in filters.items():
                if k not in meta or meta[k] != v:
                    is_match = False
                    break
            if is_match:
                matches.append(cand)
        
        if len(matches) == 0:
            logger.warning(f"No match found for {name} with filters {filters} among {len(candidates)} candidates.")
            return None
        elif len(matches) > 1:
            # Ambiguous? Return last registered or first?
            # Original code seemed to rely on overwriting or specific key.
            # We return the last one (most recently registered)
            logger.debug(f"Multiple matches for {name}, returning last one.")
            return matches[-1]['obj']
        else:
            return matches[0]['obj']

    def get_metadata(self, name: str, **filters) -> Dict[str, Any]:
        candidates = self._registry.get(name, [])
        # Same logic as get, but return meta
        if not candidates: return {}
        
        matches = []
        for cand in candidates:
            meta = cand['meta']
            is_match = True
            for k, v in filters.items():
                if k not in meta or meta[k] != v:
                    is_match = False
                    break
            if is_match:
                matches.append(cand)
        
        if matches:
            return matches[-1]['meta']
        return {}

    def list_available(self) -> list:
        return list(self._registry.keys())

# Global Registries
ModelRegistry = Registry()
RelationRegistry = Registry()
