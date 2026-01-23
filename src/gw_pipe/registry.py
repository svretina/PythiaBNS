import logging

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Registry to manage gravitational wave source models."""
    _models = {}

    @classmethod
    def register(cls, name, nfreqs=None, conversion_func=None):
        """Decorator to register a model function."""
        def decorator(func):
            model_key = (name.lower(), nfreqs)
            cls._models[model_key] = {
                'func': func,
                'conversion_func': conversion_func
            }
            logger.info(f"Registered model: {name} (nfreqs={nfreqs})")
            return func
        return decorator

    @classmethod
    def get_model(cls, name, nfreqs=None):
        """Retrieve a model from the registry."""
        model_entry = cls._get_entry(name, nfreqs)
        return model_entry['func'] if model_entry else None

    @classmethod
    def get_conversion_func(cls, name, nfreqs=None):
        """Retrieve a conversion function from the registry."""
        model_entry = cls._get_entry(name, nfreqs)
        return model_entry['conversion_func'] if model_entry else None

    @classmethod
    def _get_entry(cls, name, nfreqs=None):
        """Internal helper to get model entry."""
        # Try exact match (name, nfreqs)
        entry = cls._models.get((name.lower(), nfreqs))
        if entry:
            return entry
        
        # Try name only
        entry = cls._models.get((name.lower(), None))
        if entry:
            return entry
        
        return None

    @classmethod
    def list_models(cls):
        """List all registered models."""
        return list(cls._models.keys())

def register_model(name, nfreqs=None, conversion_func=None):
    return ModelRegistry.register(name, nfreqs, conversion_func)
