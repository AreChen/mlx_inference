from mlx_lm import load as load_text_model
from mlx_vlm import load as vlm_load
from mlx_vlm.utils import load_config
from .config import settings
from .logger import logger

# 启动时一次性加载所有模型到全局变量
loaded_text_models = {}
print("DEBUG: preload_text_models =", settings.preload_text_models)
for name in settings.preload_text_models:
    try:
        loaded_text_models[name] = load_text_model(name)
        logger.info(f"Preloaded text model: {name}")
    except Exception as e:
        logger.warning(f"Preload failed for text model {name}: {e}")

loaded_vision_models = {}
for name in settings.preload_vision_models:
    try:
        model, processor = vlm_load(name, trust_remote_code=True)
        config = load_config(name, trust_remote_code=True)
        loaded_vision_models[name] = (model, processor, config)
        logger.info(f"Preloaded vision model: {name}")
    except Exception as e:
        logger.warning(f"Preload failed for vision model {name}: {e}")

# 禁止推理时动态加载模型，推理时只能用已加载模型
    else:
        logger.info('Initial prompt cache setup complete, no saving attempted due to compatibility.')
        logger.info("Skipping saving prompt cache due to compatibility issues.")