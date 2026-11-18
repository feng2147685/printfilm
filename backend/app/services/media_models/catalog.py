from .prices import active_book, resolve_strategy
from .runtime import pin_route


async def priced_catalog(db, catalog: dict) -> dict:
    """按实际渠道、策略和价格组装前台可选模型与分辨率。"""
    from app.config import get_settings
    from app.services.direct_models import DIRECT_MINIMAX, H3_RESOLUTION
    from app.services.model_availability import is_model_enabled
    from app.services.model_settings import get_routing_snapshot
    models = get_routing_snapshot().logical_models
    catalog["generation_defaults"] = {"image_resolution": "2K", "video_resolution": get_settings().ark_video_resolution}
    book = await active_book(db)
    catalog["price_version"] = book["version"] if book else None
    catalog["billing_mode"] = "fixed_media" if book else "legacy"
    for key in ("image_models", "video_models"):
        rows = []
        for row in catalog[key]:
            if not is_model_enabled("image" if key == "image_models" else "video", row["id"], models):
                continue
            try:
                strategy = resolve_strategy(row["id"])
            except ValueError:
                continue
            row.update(strategy.describe())
            prices = (book["prices"].get(strategy.model_id) or {}) if book else {}
            row["prices"] = prices
            row["price_version"] = catalog["price_version"]
            row["available_resolutions"] = [x for x in strategy.resolutions if x in prices] if book else list(strategy.resolutions)
            try:
                route = pin_route(strategy.capability, row["id"])
                if route["channel_id"] == DIRECT_MINIMAX:
                    row["available_resolutions"] = [value for value in row["available_resolutions"] if value == H3_RESOLUTION]
                row["available"] = bool(row["available_resolutions"])
            except ValueError:
                row["available"] = False
            row["pricing_hint"] = "按张计费" if strategy.capability == "image" else "按提交时长计费"
            rows.append(row)
        catalog[key] = rows
    return catalog
