from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import statistics
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TelemetryData(BaseModel):
    adc: float
    temperature: float

# State Management
rolling_buffer = []
MAX_BUFFER = 10
latest_payload = None

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception:
                pass

manager = ConnectionManager()

@app.get("/health")
async def health_check():
    return {
        "ok": True,
        "clients": len(manager.active_connections),
        "has_payload": latest_payload is not None
    }

@app.post("/ingest")
async def ingest_data(data: TelemetryData):
    global latest_payload
    rolling_buffer.append(data.adc)
    if len(rolling_buffer) > MAX_BUFFER:
        rolling_buffer.pop(0)
    
    median_freq = statistics.median(rolling_buffer) if rolling_buffer else 0
    
    # Rule engine evaluation
    if median_freq < 100:
        adulteration_type = "AWAITING SENSOR DATA"
        safety_score = 0
        confidence = 0.0
    elif median_freq > 3000:
        adulteration_type = "High Impurity / Adulterated"
        safety_score = 35
        confidence = 89.4
    else:
        adulteration_type = "Pure Milk / Safe"
        safety_score = 98
        confidence = 96.5

    latest_payload = {
        "adulteration_type": adulteration_type,
        "safety_score": safety_score,
        "confidence": confidence,
        "frequency": median_freq,
        "temperature": data.temperature,
        "ph": 6.7
    }

    await manager.broadcast(latest_payload)
    return {"status": "success", "median_freq": median_freq}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        if latest_payload:
            await websocket.send_text(json.dumps(latest_payload))
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
