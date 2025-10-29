from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.route import router
import asyncio
from app.model_handler import load_model_and_scaler , ensure_directories

app = FastAPI(title="Delivery Time Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    ensure_directories()
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_model_and_scaler)

app.include_router(router)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, workers=1)
    