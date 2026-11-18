"""PRINTFILM 自用直连渠道及火山方舟、MiniMax 的原生协议。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote, urlsplit

import httpx

from app.schemas_routing import DefaultModels, LogicalModel, LogicalModelBinding, ResolvedModelRoute, SystemModelChannel
from app.services import storage
from app.services.tasks.upstream import observed_request
from app.services.upstream_sanitize import sanitize_upstream_value

if TYPE_CHECKING:
    from app.config import Settings

# 以渠道 ID 区分直连和聚合协议，任务快照沿用现有凭据隔离。
DIRECT_SEEDREAM = "direct-seedream"
DIRECT_MINIMAX = "direct-minimax"
# MiniMax H3 官方标准版的首轮验收分辨率。
H3_RESOLUTION = "768p"


def direct_routing(settings: Settings) -> tuple[list[SystemModelChannel], list[LogicalModel], DefaultModels]:
    """构建独立渠道及默认模型，密钥仅从本机运行环境读取。"""
    defaults = DefaultModels(text_model="deepseek-v4-pro", image_model="seedream-5.0", video_model="MiniMax-H3")
    minimax_base = "https://api.minimax.cn" if settings.direct_minimax_region == "cn" else "https://api.minimax.io"
    definitions = (
        ("direct-deepseek", "DeepSeek 官方", "https://api.deepseek.com", settings.direct_deepseek_api_key,
         "text", defaults.text_model, defaults.text_model),
        (DIRECT_SEEDREAM, "火山方舟", "https://ark.cn-beijing.volces.com/api/v3", settings.direct_seedream_api_key,
         "image", defaults.image_model, settings.direct_seedream_model),
        (DIRECT_MINIMAX, "MiniMax 官方", minimax_base, settings.direct_minimax_api_key,
         "video", defaults.video_model, defaults.video_model),
    )
    channels = []
    models = []
    for channel, name, base, key, capability, model, upstream in definitions:
        channels.append(SystemModelChannel(
            id=channel, name=name, base_url=base, api_key=key.strip(), has_api_key=bool(key.strip()),
            protocol="openai", models=[upstream],
        ))
        models.append(LogicalModel(
            id=model, name=model, capability=capability,
            bindings=[LogicalModelBinding(channel_id=channel, upstream_model=upstream)],
        ))
    return channels, models, defaults


def local_reference_path(url: str) -> Path | None:
    """仅解析站点自身静态目录内的文件，防止参考素材读取越界。"""
    if storage.is_local_static_url(url):
        path = storage.local_path_from_url(url)
        if path and path.resolve().is_relative_to(storage.STATIC_ROOT.resolve()) and path.is_file():
            return path
    return None


async def image_reference(url: str) -> str:
    """将回环地址的图片以内联形式交给云端，保留原有公网链接。"""
    if path := local_reference_path(url):
        # 每幅参考图有限定大小，读取前检查以免放大任务内存。
        if path.stat().st_size > 30_000_000:
            raise ValueError("本机参考图超过 30 MB，请压缩后重试")
        return await asyncio.to_thread(storage.file_to_data_uri, path)
    if url.startswith(("https://", "http://", "data:image/")):
        return url
    raise ValueError("参考图文件不可读，请重新选择图片")


async def seedream_body(route: ResolvedModelRoute, prompt: str, size: str, ratio: str, refs: list[str]) -> dict:
    """生成火山方舟单图请求，按画幅补充提示词并允许参考图。"""
    body = {
        "model": route.upstream_model, "prompt": prompt if ratio == "auto" else f"{prompt}\n画面宽高比：{ratio}",
        "size": size, "response_format": "url", "output_format": "png", "watermark": False,
    }
    if refs:
        body["image"] = [await image_reference(url) for url in dict.fromkeys(refs)]
    return body


async def h3_submit(route: ResolvedModelRoute, body: dict) -> str:
    """发送一次 H3 视频任务，成功时返回可持久化的任务号。"""
    if body["resolution"] != H3_RESOLUTION:
        raise ValueError("本机 MiniMax H3 直连请选 768p")
    content = []
    for item in body["content"]:
        if item["type"] == "image_url":
            url = await image_reference(item["image_url"]["url"])
            content.append({**item, "image_url": {"url": url}})
        else:
            content.append(item)
    # H3 首尾帧采用自适应画幅。普通参考模式沿用项目画幅。
    has_frame = any(item.get("role") in {"first_frame", "last_frame"} for item in content)
    # 预留协议包装空间，避免超过官方 64 MB 请求体限制。
    if sum(len(item.get("image_url", {}).get("url", "")) for item in content) > 60_000_000:
        raise ValueError("参考图总大小过大，请压缩或使用公网图片链接")
    payload = {
        "model": route.upstream_model, "content": content, "duration": body["duration"],
        "resolution": body["resolution"].upper(), "ratio": "adaptive" if has_frame else body.get("ratio") or "16:9",
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await observed_request(client, "POST", f"{route.base_url}/v2/video_generation",
                                          provider="minimax", headers={"Authorization": f"Bearer {route.api_key}"}, json=payload)
    data = _h3_response(response)
    task_id = str(data.get("task_id") or "").strip()
    if not task_id:
        raise RuntimeError("MiniMax H3 未返回任务号，请在官方后台核对是否受理")
    return task_id


async def h3_query(route: ResolvedModelRoute, task_id: str) -> dict:
    """查询已受理任务，读取官方封装的状态与成品地址。"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await observed_request(client, "GET", f"{route.base_url}/v2/query/video_generation/{quote(task_id, safe='')}",
                                          provider="minimax", headers={"Authorization": f"Bearer {route.api_key}"})
    if response.status_code == 429 or response.status_code >= 500:
        return {"status": "running"}
    data = _h3_response(response)
    task = data.get("task", data)
    if not isinstance(task, dict) or not task.get("status"):
        raise RuntimeError("MiniMax H3 未返回任务状态")
    status = str(task.get("status") or "").lower()
    if status in {"success", "succeeded"}:
        url = (task.get("content") or {}).get("url")
        if not isinstance(url, str) or urlsplit(url).scheme != "https":
            raise RuntimeError("MiniMax H3 未返回可下载的视频地址")
        return {"status": "succeeded", "url": url}
    if status in {"failed", "fail", "cancelled", "canceled", "expired"}:
        error = task.get("error")
        message = error.get("message") if isinstance(error, dict) else error
        return {"status": "failed", "error": sanitize_upstream_value(str(message or status))}
    return {"status": "running"}


def _h3_response(response: httpx.Response) -> dict:
    """校验 MiniMax 响应并脱敏错误，避免日志记录原始上游凭据。"""
    if response.is_error:
        raise RuntimeError(f"MiniMax H3 HTTP {response.status_code}: {sanitize_upstream_value(response.text[:400])}")
    data = response.json()
    if not isinstance(data, dict) or data.get("error"):
        raise RuntimeError("MiniMax H3 返回异常响应")
    return data
