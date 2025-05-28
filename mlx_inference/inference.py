from fastapi import HTTPException
from mlx_inference.postprocess import postprocess_output
from fastapi.responses import StreamingResponse, JSONResponse
import time
import json
import base64
import threading

# 全局互斥锁，保护 Metal/MLX 资源串行访问
global_infer_lock = threading.Lock()
from io import BytesIO
from PIL import Image
import numpy as np

from .models import ChatCompletionRequest

def pad_to_patch_size(img, patch_size=14):
    """自动填充PIL图片到patch_size的整数倍，使用反射填充"""
    if not isinstance(img, Image.Image):
        return img
    w, h = img.size
    pad_w = (patch_size - w % patch_size) % patch_size
    pad_h = (patch_size - h % patch_size) % patch_size
    if pad_w == 0 and pad_h == 0:
        return img
    arr = np.array(img)
    arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    return Image.fromarray(arr)
from .model_loader import loaded_text_models, loaded_vision_models
from .logger import logger
from .config import settings

def _normalize_messages(messages):
    """确保每个 message 的 content 为 str，若为 list，则拼接所有 type=='text' 的内容"""
    normalized = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            normalized.append(msg)
        elif isinstance(content, list):
            # 拼接所有 type=='text' 的内容
            texts = []
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text" and c.get("text"):
                    texts.append(c["text"])
            if texts:
                new_msg = dict(msg)
                new_msg["content"] = "\n".join(texts)
                normalized.append(new_msg)
        # 其他情况忽略
    return normalized

def infer_text_model(model_name, body: ChatCompletionRequest, session_id: str):
    if model_name not in loaded_text_models:
        raise HTTPException(status_code=400, detail="Model not found")
    model, tokenizer = loaded_text_models[model_name]
    # 处理 system prompt
    system_prompt = "You are a helpful AI assistant. Please provide detailed and comprehensive responses to user queries, avoiding short or generic replies like 'OK' or '好的'. Ensure your answers are informative and relevant to the user's request."
    for message in body.messages:
        if message.get('role') == 'system':
            system_prompt += "\n" + message['content']
    # 构建完整消息，确保 content 为 str
    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(_normalize_messages(body.messages))
    prompt = tokenizer.apply_chat_template(full_messages, add_generation_prompt=True)
    try:
        if isinstance(prompt, str):
            prompt_tokens = tokenizer.encode(prompt)
            prompt_preview = prompt[:200]
        elif isinstance(prompt, list):
            prompt_tokens = prompt
            prompt_preview = str(prompt[:40])
        else:
            logger.error(f'Unknown prompt type: {type(prompt)} for session: {session_id}')
            raise HTTPException(
                status_code=500,
                detail=f"apply_chat_template 返回未知类型: {type(prompt)}"
            )
        logger.info(f'Generated prompt length: {len(prompt_tokens)} tokens for session: {session_id}')
        logger.info(f'full_messages count: {len(full_messages)}, prompt preview: {prompt_preview}')
        MAX_CONTEXT_TOKENS = 16384
        if len(prompt_tokens) > MAX_CONTEXT_TOKENS:
            logger.error(f'Prompt token length {len(prompt_tokens)} exceeds model max context {MAX_CONTEXT_TOKENS} for session: {session_id}')
            raise HTTPException(
                status_code=400,
                detail=f"Prompt token length {len(prompt_tokens)} exceeds model max context {MAX_CONTEXT_TOKENS}. 请缩短输入内容。"
            )
    except Exception as e:
        logger.error(f'Error calculating prompt length: {str(e)}, prompt type: {type(prompt)} for session: {session_id}')
        logger.info(f'Prompt content (first 100 chars): {str(prompt)[:100]}...')
    # 非流式响应
    if not body.stream:
        from mlx_lm import generate
        import time
        logger.info(f"[推理开始] session_id={session_id} time={time.time()}")
        # Metal/MLX 资源串行访问保护
        with global_infer_lock:
            output_text = generate(
                model, tokenizer, prompt,
                max_tokens=body.max_tokens if body.max_tokens > 0 else 64000
            )
        logger.info(f"[推理结束] session_id={session_id} time={time.time()}")
        # 判断是否为 /v1/responses 端点调用，返回 output/output_text 结构
        if getattr(body, "from_responses", False):
            output_message = {
                "id": f"msg-{session_id[:8]}",
                "role": "assistant",
                "type": "message",
                "status": "completed",
                "content": [{
                    "type": "output_text",
                    "text": output_text,
                    "annotations": []
                }]
            }
            output_text_val = ""
            try:
                output_text_val = output_message["content"][0]["text"]
            except Exception:
                output_text_val = ""
            resp = {
                "id": f"response-{session_id[:8]}",
                "object": "response",
                "created": int(time.time()),
                "model": model_name,
                "output": [output_message],
                "output_text": output_text_val,
                "status": "completed"
            }
            return JSONResponse(content=resp, media_type="application/json")
        response = {
            "id": f"chatcmpl-{session_id[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": output_text},
                "finish_reason": "stop"
            }]
        }
        return JSONResponse(content=response, media_type="application/json")
    # 流式响应
    def generation_generator():
        import time
        # print("[DEBUG] generation_generator 启动，time 已导入")
        from mlx_lm import stream_generate
        generator = stream_generate(
            model, tokenizer, prompt,
            max_tokens=body.max_tokens if body.max_tokens > 0 else 64000
        )
        # print("[DEBUG] stream_generate 已启动")
        try:
            global_infer_lock.acquire()
            for response in generator:
                if response.text:
                    for char in response.text:
                        # print(f"[DEBUG] 生成字符: {char}")
                        chunk_data = json.dumps({
                            "id": f"chatcmpl-{session_id[:8]}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": char}
                            }]
                        })
                        yield f"data: {chunk_data}\r\n\r\n"
            final_chunk_data = json.dumps({
                "id": f"chatcmpl-{session_id[:8]}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            })
            yield f"data: {final_chunk_data}\r\n\r\n"
            yield "data: [DONE]\r\n\r\n"
        finally:
            global_infer_lock.release()
    return StreamingResponse(generation_generator(), media_type='text/event-stream')

def infer_vision_model(model_name, body: ChatCompletionRequest, session_id: str):
    if model_name not in loaded_vision_models:
        raise HTTPException(status_code=400, detail="Model not found")
    model, processor, config = loaded_vision_models[model_name]
    from mlx_vlm import apply_chat_template as vlm_apply_chat_template, generate as vlm_generate
    # 解析 OpenAI 多模态 messages，提取图片和文本
    images = []
    prompt_messages = []
    # vision模型同样需要全局锁保护
    for msg in body.messages:
        content = msg.get("content")
        # 新增：合并同一 user 消息下的 text 和 image_url，生成一条 user 消息，内容为 text + "\n<image>"
        if isinstance(content, list):
            text_parts = []
            has_image = False
            for c in content:
                if isinstance(c, dict):
                    if c.get("type") == "image_url" and isinstance(c.get("image_url"), dict):
                        url = c["image_url"].get("url")
                        if url:
                            has_image = True
                            if url.startswith("data:image"):
                                try:
                                    header, b64data = url.split(",", 1)
                                    img_bytes = base64.b64decode(b64data)
                                    img = Image.open(BytesIO(img_bytes))
                                    img = pad_to_patch_size(img, patch_size=14)
                                    images.append(img)
                                except Exception as e:
                                    print(f"Failed to decode base64 image: {e}")
                            else:
                                images.append(url)
                    elif c.get("type") == "image" and c.get("image"):
                        has_image = True
                        images.append(c["image"])
                    elif c.get("type") == "text" and c.get("text"):
                        text_parts.append(c.get("text"))
            # 合并文本和图片占位符
            if text_parts or has_image:
                merged_content = "\n".join(text_parts)
                if has_image:
                    merged_content += "\n<image>"
                prompt_messages.append({"role": msg.get("role"), "content": merged_content.strip()})
        elif isinstance(content, str):
            prompt_messages.append(msg)
    # 允许只有文本输入时也能推理
    # 允许只传图片时自动补充中文默认 prompt
    # 修复：只要 images 不为空，且 prompt_messages 没有任何 user 消息，就补充一条 user 消息
    has_user_message = any(m.get("role") == "user" for m in prompt_messages)
    if images and not has_user_message:
        # 补充 user 消息内容为 <image>，确保模板能插入图片占位符
        prompt_messages.append({"role": "user", "content": "<image>\n图片如下"})
        logger.info(f"[VLM DEBUG] 自动补充 prompt_messages: {prompt_messages}")
    else:
        logger.info(f"[VLM DEBUG] 原始 prompt_messages: {prompt_messages}")
    logger.info(f"[VLM DEBUG] images 数量: {len(images)}")
    # 允许 images 为空，仅文本推理
    formatted_prompt = vlm_apply_chat_template(
        processor, config, prompt_messages, num_images=len(images)
    )
    # 统计 formatted_prompt 里 image 占位符数量
    image_token_count = str(formatted_prompt).count('<image>')
    logger.info(f"[VLM DEBUG] formatted_prompt: {formatted_prompt}")

    # Metal/MLX 资源串行访问保护
    with global_infer_lock:
        output_text = vlm_generate(
            model, processor, config, formatted_prompt, images,
            max_tokens=body.max_tokens if body.max_tokens > 0 else 64000
        )
    # ...（后续返回逻辑不变）
    logger.info(f"[VLM DEBUG] image_token_count: {image_token_count}, images_len: {len(images)}")
    if image_token_count != len(images):
        logger.warning(f"Image token count ({image_token_count}) does not match images list length ({len(images)}). Continuing with processing.")
    # 非流式
    if not body.stream:
        logger.info(f"[VLM DEBUG] Checking body.from_responses: {getattr(body, 'from_responses', 'AttributeNotFound')}") # 移到 if 块内
        output = vlm_generate(
            model,
            processor,
            formatted_prompt,
            images,
            max_tokens=getattr(body, "max_tokens", settings.max_tokens_vision),
            skip_special_tokens=settings.skip_special_tokens
        )
        # 后处理：根据模型名做字符替换
        output = postprocess_output(output, model_name)
        logger.info(f"[VLM DEBUG] output type: {type(output)}, value: {repr(output)}")
        response = {
            "id": f"chatcmpl-{session_id[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": output},
                "finish_reason": "stop"
            }]
        }
        # 判断是否为 /v1/responses 端点调用，返回 output/output_text 结构
        if getattr(body, "from_responses", False):
            logger.info(f"[VLM DEBUG] Entered from_responses branch")
            # 严格仿照 OpenAI responses 端点结构
            output_message = {
                "id": f"msg-{session_id[:8]}",
                "role": "assistant",
                "type": "message",
                "status": "completed",
                "content": [{
                    "type": "output_text",
                    "text": output,
                    "annotations": []
                }]
            }
            # output_text 取 output[0].content[0].text
            output_text_val = ""
            try:
                output_text_val = output_message["content"][0]["text"]
            except Exception:
                output_text_val = ""
            resp = {
                "id": f"response-{session_id[:8]}",
                "object": "response",
                "created": int(time.time()),
                "model": model_name,
                "output": [output_message],
                "output_text": output_text_val,
                "status": "completed"
            }
            logger.info(f"[VLM DEBUG] resp to return: {resp}")
            return JSONResponse(content=resp, media_type="application/json")
        return JSONResponse(content=response, media_type="application/json")
    # 流式
    from mlx_vlm.utils import stream_generate
    async def event_stream():
        idx = 0
        for chunk in stream_generate(
            model,
            processor,
            formatted_prompt,
            images,
            max_tokens=getattr(body, "max_tokens", settings.max_tokens_vision),
            stream=True,
            skip_special_tokens=settings.skip_special_tokens
        ):
            text = getattr(chunk, "text", str(chunk))
            # 后处理：根据模型名做字符替换
            text = postprocess_output(text, model_name)
            data = {
                "id": f"chatcmpl-{session_id[:8]}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {"content": text}
                }]
            }
            yield f"data: {json.dumps(data, ensure_ascii=False)}\r\n\r\n"
            idx += 1
        yield "data: [DONE]\r\n\r\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")