"""直连冒烟测试从实际模型入口取证，仅用离线供应商响应。"""

from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from app.config import get_settings
from app.services import ark, direct_models, llm_client, model_settings, storage
from app.services.media_models.runtime import pin_route, restore_route
from app.services.tasks.upstream import _video_request


@pytest.fixture
def direct_config(monkeypatch):
    """三路测试凭据只用于验证路由和请求头的隔离。"""
    settings = get_settings().model_copy(update={
        "direct_models_enabled": True, "direct_deepseek_api_key": "test-text", "direct_seedream_api_key": "test-image",
        "direct_seedream_model": "ep-test", "direct_minimax_api_key": "test-video", "direct_minimax_region": "cn",
        "ark_mock": False, "local_media_mock": False, "ark_video_poll_interval": 0,
    })
    monkeypatch.setattr(model_settings, "get_settings", lambda: settings)
    monkeypatch.setattr(model_settings, "reload_settings", lambda: settings)
    monkeypatch.setattr(model_settings, "_overlay", {})
    monkeypatch.setattr(model_settings, "_routing_snapshot", model_settings.RoutingSnapshot(
        [], [], model_settings.DefaultModels(),
    ))
    monkeypatch.setattr("app.services.media_models.runtime.get_settings", lambda: settings)
    return settings


async def test_direct_routes_and_media_requests(direct_config, monkeypatch, tmp_path):
    """验证文字、方舟图片、H3 视频分别走官方请求并读取本地分镜图。"""
    await model_settings.load_model_settings_cache(None)
    gateway = ark.ArkGateway(direct_config)
    text_url, headers, body = llm_client.prepare_chat_request("写故事", "雨夜车站")
    assert text_url == "https://api.deepseek.com/chat/completions"
    assert headers["Authorization"] == "Bearer test-text"
    assert body["model"] == "deepseek-v4-pro"
    assert body["thinking"] == {"type": "disabled"}

    fixture = Path(__file__).resolve().parents[1] / "static/mock/local/default.jpg"
    monkeypatch.setattr(storage, "STATIC_ROOT", tmp_path)
    monkeypatch.setattr(storage, "GENERATED_ROOT", tmp_path / "generated")
    monkeypatch.setattr("app.services.oss.oss_enabled", lambda: False)
    (tmp_path / "style.jpg").write_bytes(fixture.read_bytes())

    async def download(url, dest, **kwargs):
        """在离线测试中落盘供应商生成的图片。"""
        assert url == "https://images.example.test/frame.png"
        assert not kwargs.get("headers")
        dest.write_bytes(fixture.read_bytes())

    monkeypatch.setattr(storage, "download_to", download)
    image_http = AsyncMock(return_value=httpx.Response(200, json={"data": [{"url": "https://images.example.test/frame.png"}]}))
    monkeypatch.setattr(ark, "observed_request", image_http)
    plan = pin_route("image", "seedream-5.0")
    assert "api_key" not in plan
    assert restore_route(plan).api_key == "test-image"
    image = await gateway.gen_image("雨夜车站", "", ["/static/style.jpg"], project_id=5, shot_no=1,
                                    size="2K", model="seedream-5.0", aspect_ratio="9:16")
    call = image_http.await_args
    assert call.args[2] == "https://ark.cn-beijing.volces.com/api/v3/images/generations"
    assert call.kwargs["headers"]["Authorization"] == "Bearer test-image"
    assert call.kwargs["json"]["model"] == "ep-test"
    assert call.kwargs["json"]["image"][0].startswith("data:image/jpeg;base64,")

    video_http = AsyncMock(side_effect=[
        httpx.Response(200, json={"task_id": "h3-test"}),
        httpx.Response(200, json={"task": {"status": "queued"}}),
        httpx.Response(200, json={"task": {"status": "succeeded", "content": {"url": "https://video.example.test/result.mp4"}}}),
    ])
    monkeypatch.setattr(direct_models, "observed_request", video_http)
    task_id = await gateway.gen_video_i2v(image.local_url, "让列车缓缓进站", 5, resolution="768p", ratio="9:16", model="MiniMax-H3")
    result = await gateway.poll_task(task_id)
    assert result.status == "succeeded"
    assert result.url == "https://video.example.test/result.mp4"
    submit, first_poll, _ = video_http.await_args_list
    assert submit.args[2] == "https://api.minimax.cn/v2/video_generation"
    assert submit.kwargs["headers"]["Authorization"] == "Bearer test-video"
    assert submit.kwargs["json"]["resolution"] == "768P"
    assert submit.kwargs["json"]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert _video_request("POST", submit.args[2]) == (True, False, None)
    assert _video_request("GET", first_poll.args[2]) == (False, True, task_id)


async def test_missing_direct_key_never_falls_back_to_gateway(direct_config):
    """缺少任一路 Key 时拒绝提交，不借用其它模型或聚合凭据。"""
    direct_config.direct_seedream_api_key = ""
    await model_settings.load_model_settings_cache(None)
    with pytest.raises(ValueError, match="可用渠道"):
        pin_route("image", "seedream-5.0")
