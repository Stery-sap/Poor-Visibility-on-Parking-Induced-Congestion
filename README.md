# ParkWatch AI 🚦
### AI-Driven Parking Violation Intelligence for Bengaluru Traffic Management

> Detects illegal parking hotspots, quantifies their congestion impact, and enables targeted enforcement — powered by LightGBM trained on 298,450 real police violation records.

🔗 **[Live Demo](https://profound-yeot-3ac6c7.netlify.app/)**

---

## The Problem

On-street illegal parking near commercial areas, metro stations, and busy junctions chokes carriageways and intersections across Bengaluru. Enforcement today is entirely **reactive and patrol-based** — officers respond after congestion has already formed, with no data on where parking violations cause the most traffic damage or when to prioritise which zones.

**ParkWatch AI answers three questions enforcement teams couldn't answer before:**
- Which parking violations will cause the highest congestion impact?
- Which grid zones are chronic hotspots that need permanent enforcement?
- When (hour, day, vehicle type) does risk peak at each location?

---

## What's in the Dashboard

**Hotspot map** — Predicted congestion zones rendered as risk-coloured circles on a live Bengaluru basemap, filterable by CRITICAL / HIGH / MEDIUM risk level.

**Live prediction panel** — Enter any violation's coordinates, time, vehicle type, and violation category. The LightGBM model returns a congestion probability, risk level, and the specific factors that drove the score in under 5ms.

**Model comparison** — F1 scores for all four trained models displayed side-by-side so you can see why LightGBM was chosen.

**Hourly violation chart** — Violations by hour of day, colour-coded to show peak congestion windows (7–9 AM and 5–8 PM).

---

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| **LightGBM** ✅ | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 1.0000 |
| XGBoost | 0.9999 | 1.0000 | 0.9999 | 0.9999 | 1.0000 |
| Random Forest | 0.9994 | 1.0000 | 0.9992 | 0.9996 | 1.0000 |
| Logistic Regression | 0.9041 | 0.9693 | 0.9111 | 0.9393 | 0.9666 |

LightGBM was selected — same accuracy as XGBoost, 3× smaller on disk (1.2 MB vs 13 MB for Random Forest), faster inference, and native feature importance.

**Top predictive features:** location (lat/lon), grid cell violation density, police station zone, main road obstruction, heavy vehicle type, and peak hour.

---

## How the Congestion Score Works

No ground-truth congestion label existed in the raw data, so the target variable was engineered from domain knowledge. Each violation receives a weighted score across seven factors:

| Factor | Weight |
|---|---|
| Grid cell in top 25% density | +3 |
| Main road obstruction | +2 |
| Heavy vehicle (LGV / HGV / Bus) | +2 |
| High-traffic junction (top 30) | +2 |
| Double parking | +2 |
| Peak hour (7–9 AM, 5–8 PM) | +1 |
| Footpath blocked | +1 |

Violations scoring ≥ 5 are labelled **high congestion impact** — 81.4% of the 298,450-record dataset. The model learns to predict this from 18 engineered features spanning location, time, violation type, and vehicle category.

---

## API

**Example request:**
```json
POST /predict
{
  "latitude": 12.9716,
  "longitude": 77.5946,
  "created_datetime": "2024-03-15T08:30:00+00:00",
  "violation_type": "PARKING IN A MAIN ROAD",
  "vehicle_type": "LGV",
  "police_station": "Upparpet",
  "grid_violation_count": 420
}
```

**Example response:**
```json
{
  "high_congestion_impact": 1,
  "congestion_probability": 0.9874,
  "risk_level": "CRITICAL",
  "key_risk_factors": [
    "Main road obstruction",
    "Heavy vehicle",
    "Peak hour violation",
    "Dense violation hotspot"
  ]
}
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML pipeline | Python · scikit-learn · LightGBM · XGBoost · pandas · joblib |
| Backend API | FastAPI · Uvicorn · Pydantic |
| Frontend | HTML · CSS · JavaScript · Leaflet.js · Chart.js |
| Deployment | Render (API) · Netlify (frontend) |

---

## Project Structure

```
Poor-Visibility-on-Parking-Induced-Congestion/
├── parking_intelligence_complete.py   # ML pipeline + inference module
├── api.py                             # FastAPI backend
├── index.html                         # Frontend dashboard
├── requirements.txt
└── model_artifacts/
    ├── best_model.pkl                 # Trained LightGBM model
    ├── metadata.json                  # Feature list, thresholds, label classes
    ├── hotspot_report.csv             # Top-50 predicted hotspot cells
    ├── feature_importance.csv
    ├── le_vehicle.pkl                 # LabelEncoder — vehicle_type
    ├── le_station.pkl                 # LabelEncoder — police_station
    └── scaler.pkl                     # StandardScaler
```

---

## Dataset

**Source:** Bengaluru Traffic Police violation records  
**Period:** January – May 2024  
**Size:** 298,450 records  
**Fields:** GPS coordinates, violation type, vehicle type, junction name, police station, timestamp  
**Note:** Dataset is anonymised. Raw data is not included in this repo.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

Built as part of a hackathon