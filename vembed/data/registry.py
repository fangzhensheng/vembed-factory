class CollatorRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(collator_cls):
            cls._registry[name] = collator_cls
            return collator_cls

        return decorator

    @classmethod
    def get(cls, name):
        return cls._registry.get(name)

    @classmethod
    def list_collators(cls):
        return list(cls._registry.keys())
