def postprocess_output(text: str, model_name: str) -> str:
    """
    针对不同模型名，对输出字符串进行定制化后处理（如字符替换）。
    支持未来多模型多规则扩展。

    :param text: 原始输出字符串
    :param model_name: 当前模型名称
    :return: 处理后的字符串
    """
    rules = {
        "mlx-community/Kimi-VL-A3B-Thinking-8bit": [
            ("◁", "<"),
            ("▷", ">"),
        ],
        # 未来可在此扩展更多模型和规则
    }
    for old, new in rules.get(model_name, []):
        text = text.replace(old, new)
    return text