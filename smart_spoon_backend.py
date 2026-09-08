from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import statistics
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

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

# --- MULTI-MODEL ENSEMBLE INITIALIZATION ---
# Training baseline data: [Frequency (Hz), Temperature (°C)]
# Labels: 0 = Awaiting/Invalid, 1 = Adulterated / Impure, 2 = Pure / Safe
X_train = np.array([
    [0, 25.0],       # Inactive / No sensor
    [50, 26.0],      # Low signal / Noise
    [3500, 24.5],    # High impurity / electrolyte spike
    [4000, 28.0],    # Adulterated sample
    [1500, 25.0],    # Pure milk baseline
    [1800, 26.5],    # Pure milk typical range
    [2200, 24.0]     # Pure milk safe range
])
y_train = np.array([0, 0, 1, 1, 2, 2, 2])

# Define individual models
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=50, random_state=42)

# Combine them into a Soft Voting Ensemble for consensus prediction
ensemble_model = VotingClassifier(
    estimators=[('rf', rf_model), ('gb', gb_model)],
    voting='soft'
)
ensemble_model.fit(X_train, y_train)

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
    
    # --- MULTI-MODEL ENSEMBLE EVALUATION ---
    input_features = np.array([[median_freq, data.temperature]])
    prediction_class = ensemble_model.predict(input_features)[0]
    probabilities = ensemble_model.predict_proba(input_features)[0]
    confidence = float(max(probabilities) * 100)

    # Map consensus model output classes to domain metrics
    if median_freq < 100 or prediction_class == 0:
        adulteration_type = "AWAITING SENSOR DATA"
        safety_score = 0
        confidence = 0.0
    elif prediction_class == 1:
        adulteration_type = "High Impurity / Adulterated"
        safety_score = int(np.clip(100 - (confidence * 0.7), 10, 45))
    else:
        adulteration_type = "Pure Milk / Safe"
        safety_score = int(np.clip(85 + (confidence * 0.15), 90, 99))

    latest_payload = {
        "adulteration_type": adulteration_type,
        "safety_score": safety_score,
        "confidence": round(confidence, 1),
        "frequency": median_freq,
        "temperature": data.temperature,
        "ph": 6.7
    }

    await manager.broadcast(latest_payload)
    return {"status": "success", "median_freq": median_freq, "ensemble_prediction": int(prediction_class)}

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
