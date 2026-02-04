# Fraud Detection API - Deploy to Render

This folder contains everything needed to deploy your fraud detection API to Render.com.

## Quick Deploy (5 Steps)

### Step 1: Copy Your Trained Models

After running the training notebook (L6.1), copy your models:

```bash
# From Part1_Training folder
cp -r models/* ../Deploy_Render/models/
```

Or manually copy these files to the `models/` folder:
- `xgboost_pipeline.pkl`
- `random_forest_pipeline.pkl`
- `model_metadata.json`
- `feature_stats.json`

### Step 2: Create GitHub Repository

```bash
# Initialize git
cd Deploy_Render
git init
git add .
git commit -m "Initial commit - Fraud Detection API"

# Create a new repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/fraud-detection-api.git
git branch -M main
git push -u origin main
```

### Step 3: Deploy on Render

1. Go to [render.com](https://render.com) and sign up (free)
2. Click **New** → **Web Service**
3. Connect your GitHub account
4. Select your `fraud-detection-api` repository
5. Configure:
   - **Name**: `fraud-detection-api`
   - **Region**: Singapore (or nearest)
   - **Branch**: `main`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Click **Create Web Service**

### Step 4: Wait for Deployment (~5 minutes)

Render will:
1. Clone your repository
2. Install dependencies
3. Start your API

### Step 5: Test Your API

Your API will be live at: `https://fraud-detection-api-xxxx.onrender.com`

Test it:
```bash
curl https://fraud-detection-api-xxxx.onrender.com/
```

Or open in browser to see the API docs:
`https://fraud-detection-api-xxxx.onrender.com/docs`

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Detailed status |
| `/model-info` | GET | Model metadata |
| `/predict` | POST | Predict fraud |
| `/docs` | GET | Interactive API docs |

---

## Example Prediction Request

```bash
curl -X POST "https://YOUR-URL.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "category": "Grocery",
    "amount": 150.50,
    "age_at_transaction": 35.5,
    "days_until_card_expires": 365.0,
    "loc_delta": 0.05,
    "trans_volume_mavg": 120.0,
    "trans_volume_mstd": 45.0,
    "trans_freq": 3.0,
    "loc_delta_mavg": 0.03
  }'
```

---

## Troubleshooting

### "Service unavailable"
- Wait 30-60 seconds (cold start)
- Check Render logs for errors

### "Model not found"
- Ensure models/ folder has all .pkl files
- Commit and push: `git add . && git commit -m "Add models" && git push`

### "Memory error"
- Free tier has 512MB RAM limit
- Pipeline files should be under this limit

---

## Folder Structure

```
Deploy_Render/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── Procfile            # Render start command
├── README.md           # This file
└── models/             # Model files (add after training)
    ├── xgboost_pipeline.pkl
    ├── random_forest_pipeline.pkl
    ├── model_metadata.json
    └── feature_stats.json
```
