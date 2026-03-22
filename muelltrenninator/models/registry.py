# Global model dictionary

MODEL_REGISTRY = {}

def register_model(model_name):
    def decorator(model_cls):
        MODEL_REGISTRY[model_name] = model_cls
        return model_cls
    return decorator