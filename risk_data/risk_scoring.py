from __future__ import annotations

# 保持 risk_scoring.py 作为统一入口：
# - 支持 `python -m risk_data.risk_scoring`
# - 兼容 `from risk_scoring import ...` 的历史导入
try:
    from . import risk_engine as _engine
except ImportError:
    import risk_engine as _engine  # type: ignore


for _name in dir(_engine):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_engine, _name)


if __name__ == "__main__":
    raise SystemExit(_engine._cli())