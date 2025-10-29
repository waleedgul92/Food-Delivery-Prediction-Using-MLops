from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .route import router
import asyncio
from .model_handler import load_model_and_scaler, ensure_directories
from contextlib import asynccontextmanager

# Use the 'lifespan' context manager instead of on_event
@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code runs on startup
    print("Starting up...")
    ensure_directories()
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_model_and_scaler)
    print("Model loaded. Application startup complete.")
    
    yield
    
    # This code runs on shutdown (if needed)
    print("Shutting down...")

app = FastAPI(title="Delivery Time Predictor API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)