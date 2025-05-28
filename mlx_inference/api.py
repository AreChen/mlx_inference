import asyncio
from fastapi import FastAPI, HTTPException, Request
from .models import ChatCompletionRequest, ResponseRequest
from .config import settings
from .logger import logger
from .model_loader import loaded_text_models, loaded_vision_models
# from .queue import ...  # 并发队列相关已移除，无需导入
from .utils import get_session_id
from .inference import infer_text_model, infer_vision_model

app = FastAPI()

def standardize_content(content):
    """
    递归标准化 content 列表，兼容 OpenAI 官方结构。
    - type: input_text → text
    - type: input_image → image_url
    - image_url 字段标准化为 dict(url=...)
    """
    if isinstance(content, list):
        new_content = []
        for c in content:
            if isinstance(c, dict):
                if c.get("type") == "input_text":
                    c["type"] = "text"
                    # 保持 text 字段
                elif c.get("type") == "input_image":
                    c["type"] = "image_url"
                    # 兼容 image_url 为字符串或 dict
                    url_val = c.get("image_url")
                    if isinstance(url_val, str):
                        c["image_url"] = {"url": url_val}
                    elif isinstance(url_val, dict):
                        # 已经是 dict，保持不变
                        pass
                # 可扩展更多兼容性映射
                # 递归处理嵌套 content
                if "content" in c:
                    c["content"] = standardize_content(c["content"])
            new_content.append(c)
        return new_content
    return content
async def worker_func(request, body, session_id, endpoint_func, endpoint_name, response_future):
    logger.info(f"[队列] 开始处理 {endpoint_name} session_id={session_id}")
    try:
        result = await endpoint_func(request, body, session_id)
        response_future.set_result(result)
    except Exception as e:
        logger.error(f"[队列] 处理 {endpoint_name} session_id={session_id} 异常: {e}")
        response_future.set_exception(e)

# 全局推理队列和 worker
inference_queue = asyncio.Queue()

@app.on_event("startup")
async def startup_event():
    async def inference_worker():
        while True:
            req, body, session_id, fut = await inference_queue.get()
            try:
                result = await handle_chat_completions(req, body, session_id)
                fut.set_result(result)
            except Exception as e:
                fut.set_exception(e)
            finally:
                inference_queue.task_done()
    asyncio.create_task(inference_worker())
    logger.info("应用启动完成，已启用全局推理队列串行化。")

async def handle_chat_completions(request, body, session_id):
    model_name = getattr(body, "model", None)
    if not model_name:
        raise HTTPException(status_code=400, detail="No model specified")
    # 别名映射，仅用于请求阶段
    alias_map = getattr(settings, "model_aliases", {}) or {}
    real_model_name = alias_map.get(model_name, model_name)
    # 禁止推理时动态加载模型，只能用已加载模型
    is_text_model = real_model_name in loaded_text_models
    is_vision_model = real_model_name in loaded_vision_models
    if not (is_text_model or is_vision_model):
        raise HTTPException(status_code=400, detail="Model not found or not preloaded")
    # API Key 校验
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    api_key = auth_header.split(' ')[1]
    if api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    if not body.messages:
        raise HTTPException(status_code=400, detail="No messages provided")
    # 推理分流
    if is_text_model:
        return infer_text_model(real_model_name, body, session_id)
    elif is_vision_model:
        return infer_vision_model(real_model_name, body, session_id)
    else:
        raise HTTPException(status_code=400, detail="Model type not supported")

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionRequest):
    logger.info(f"Received request for /v1/chat/completions with body: {body}")
    # 参数校验
    if body.temperature < 0 or body.temperature > 1:
        raise HTTPException(status_code=400, detail="Temperature must be between 0 and 1")
    if body.top_p < 0 or body.top_p > 1:
        raise HTTPException(status_code=400, detail="Top_p must be between 0 and 1")
    if body.n < 1:
        raise HTTPException(status_code=400, detail="N must be at least 1")
    session_id = get_session_id(request.headers)
    logger.info(f'Assigned session ID: {session_id}')
    # 递归标准化 messages 内 content 字段，兼容 OpenAI 多模态结构
    if hasattr(body, "messages") and isinstance(body.messages, list):
        for msg in body.messages:
            if isinstance(msg, dict) and "content" in msg:
                msg["content"] = standardize_content(msg["content"])
    # 推理请求入队，串行化执行
    fut = asyncio.get_event_loop().create_future()
    await inference_queue.put((request, body, session_id, fut))
    result = await fut
    return result

@app.post("/v1/responses")
async def responses(request: Request, body: ResponseRequest):
    logger.info(f"Received request for /v1/responses with body: {body}")
    if body.temperature < 0 or body.temperature > 1:
        raise HTTPException(status_code=400, detail="Temperature must be between 0 and 1")
    if body.top_p < 0 or body.top_p > 1:
        raise HTTPException(status_code=400, detail="Top_p must be between 0 and 1")
    if body.n < 1:
        raise HTTPException(status_code=400, detail="N must be at least 1")
    # input 兼容 str 或 list，转为 messages
    if isinstance(body.input, str):
        messages = [{"role": "user", "content": body.input}]
    elif isinstance(body.input, list):
        # 对 input 列表每个元素的 content 字段做标准化
        messages = []
        for item in body.input:
            # 若 item 是 dict 且有 content 字段，递归标准化
            if isinstance(item, dict) and "content" in item:
                item = dict(item)  # 拷贝，避免副作用
                item["content"] = standardize_content(item["content"])
            messages.append({"role": "user", "content": item.get("content", item) if isinstance(item, dict) else item})
    else:
        raise HTTPException(status_code=400, detail="Invalid input type for 'input'")
    # 构造 ChatCompletionRequest
    chat_body = ChatCompletionRequest(
        messages=messages,
        model=body.model,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
        n=body.n,
        stop=body.stop,
        stream=body.stream,
        functions=body.functions,
        from_responses=True  # 标记为 responses 端点
    )
    session_id = get_session_id(request.headers)
    logger.info(f'Assigned session ID: {session_id}')
    # 复用 chat.completions 推理分流逻辑
    return await handle_chat_completions(request, chat_body, session_id)

@app.get("/v1/models")
async def list_models():
    data = []
    alias_map = getattr(settings, "model_aliases", {}) or {}
    # 反向映射：真实名->别名
    reverse_alias = {v: k for k, v in alias_map.items()}

    for name in settings.preload_text_models:
        display_name = reverse_alias.get(name, name)
        data.append({
            "id": display_name,
            "object": "model",
            "created": 0,
            "owned_by": "local",
            "type": "text",
            "real_name": name if display_name != name else None
        })
    for name in settings.preload_vision_models:
        display_name = reverse_alias.get(name, name)
        data.append({
            "id": display_name,
            "object": "model",
            "created": 0,
            "owned_by": "local",
            "type": "vision",
            "real_name": name if display_name != name else None
        })
    return {
        "object": "list",
        "data": data
    }