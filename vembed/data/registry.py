class CollatorRegistry:
    """Registry for batch collator functions.

    Manages registration and retrieval of custom data collators for training.
    """

    _registry = {}

    @classmethod
    def register(cls, name):
        """Register a collator class.

        Args:
            name: Unique identifier for the collator.

        Returns:
            Decorator function that registers the collator class.
        """
        def decorator(collator_cls):
            cls._registry[name] = collator_cls
            return collator_cls

        return decorator

    @classmethod
    def get(cls, name):
        """Retrieve a registered collator class.

        Args:
            name: Name of the collator to retrieve.

        Returns:
            The collator class if found, None otherwise.
        """
        return cls._registry.get(name)

    @classmethod
    def list_collators(cls):
        """List all registered collator names.

        Returns:
            List of registered collator names.
        """
        return list(cls._registry.keys())
