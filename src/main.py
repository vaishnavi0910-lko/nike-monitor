from src.api_realtime import app  # noqa: F401 — exposes `app` for `uvicorn main:app`

if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
