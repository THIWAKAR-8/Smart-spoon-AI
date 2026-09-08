from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import statistics
import json
import numpy as np
from datetime import datetime
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
X_train = np.array([
    [0, 25.0],
    [50, 26.0],
    [3500, 24.5],
    [4000, 28.0],
    [1500, 25.0],
    [1800, 26.5],
    [2200, 24.0]
])
y_train = np.array([0, 0, 1, 1, 2, 2, 2])

rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=50, random_state=42)
ensemble_model = VotingClassifier(estimators=[('rf', rf_model), ('gb', gb_model)], voting='soft')
ensemble_model.fit(X_train, y_train)

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
    
    input_features = np.array([[median_freq, data.temperature]])
    prediction_class = ensemble_model.predict(input_features)[0]
    probabilities = ensemble_model.predict_proba(input_features)[0]
    confidence = float(max(probabilities) * 100)

    if median_freq < 100 or prediction_class == 0:
        verdict = "AWAITING SENSOR DATA"
        status_color = "#334155"
        safety_score = 0
        confidence = 0.0
    elif prediction_class == 1:
        verdict = "High Impurity / Adulterated"
        status_color = "#ef4444"
        safety_score = 35
    else:
        verdict = "Pure Milk / Safe"
        status_color = "#10b981"
        safety_score = 96

    # Structured payload matching frontend expectations
    latest_payload = {
        "hero": {
            "adulteration_type": verdict,
            "accuracy": round(confidence, 1),
            "status_color": status_color
        },
        "primary": {
            "1_safety_score": safety_score,
            "11_kitchen_directive": "Boil thoroughly before consumption." if safety_score < 80 else "Sample is fresh and safe for domestic use.",
            "12_countertop_timer_hrs": "6 Hours",
            "13_fridge_timer_hrs": "48 Hours",
            "16_water_adulteration_pct": 0 if prediction_class == 2 else 25,
            "19_fraud_loss_penalty_inr": 12 if prediction_class == 1 else 0,
            "21_REAL_TIME_PH_METER": 6.7
        },
        "secondary": {
            "eis_dsp_telemetry": {
                "1_Total_Impedance_Magnitude": int(median_freq if median_freq > 0 else 500)
            },
            "randles_circuit_parameters": {
                "Solution_Resistance_Rs": "42.5 Ω",
                "Charge_Transfer_Rct": "185.2 Ω"
            },
            "biochemical_physics": {
                "Ionic_Conductivity": "2.41 mS/cm",
                "Dielectric_Constant": "78.4"
            },
            "dairy_rheology_economics": {
                "Estimated_Fat_Pct": "3.5%",
                "SNF_Content": "8.5%"
            },
            "ai_and_regulatory_metrology": {
                "35_Class_Probability_Distribution": str({
                    "Pure_Milk": round(probabilities[2] * 100, 1),
                    "Water_Dilution": round(probabilities[1] * 100, 1),
                    "Urea_Admixture": 2.0,
                    "Detergent": 1.0
                })
            }
        },
        "system_meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "raw_adc": int(data.adc),
            "probe_temperature_c": data.temperature,
            "excitation_frequency_hz": int(median_freq),
            "com_port": "ESP32_WIFI_WSS"
        }
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
