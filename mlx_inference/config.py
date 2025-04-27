from dotenv import dotenv_values
import json
from typing import Any, Optional

class Config:
    def __init__(self):
        env = dotenv_values(".env")

        # 读取字符串配置
        self.api_key: str = env.get("API_KEY", "eapil_api_key")
        self.max_tokens_text: int = int(env.get("MAX_TOKENS_TEXT", 64000))
        self.max_tokens_vision: int = int(env.get("MAX_TOKENS_VISION", 4096))
        self.skip_special_tokens: bool = str(env.get("SKIP_SPECIAL_TOKENS", "False")).lower() in ("true", "1", "yes")

        # 解析模型别名映射
        model_aliases_raw = env.get("MODEL_ALIASES", None)
        self.model_aliases: Optional[Any] = None
        if model_aliases_raw:
            try:
                self.model_aliases = json.loads(model_aliases_raw)
            except Exception as e:
                print(f"[配置错误] MODEL_ALIASES 解析失败: {e}，内容为: {model_aliases_raw}")
                self.model_aliases = {}

        # 解析模型列表
        def parse_list(key, default=None):
            val = env.get(key)
            if not val:
                return default if default is not None else []
            try:
                return json.loads(val)
            except Exception:
                # 兼容逗号分隔
                return [i.strip() for i in val.split(",") if i.strip()]

        self.preload_text_models = parse_list("PRELOAD_TEXT_MODELS")
        self.preload_vision_models = parse_list("PRELOAD_VISION_MODELS")

settings = Config()