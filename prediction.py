import numpy as np
import pandas as pd
import joblib
import os
import time

# ===========================
# Lazy-loaded global models
# ===========================
npk_model = None
temp_hum_model = None
yield_model = None
rainfall_model = None

le_state = None
le_soil = None
le_crop = None


def load_models():
    global npk_model, temp_hum_model, yield_model, rainfall_model
    global le_state, le_soil, le_crop

    if npk_model is None:
        print("Loading models...")

        npk_model = joblib.load("models/predict_npk_xgb_model.pkl")
        temp_hum_model = joblib.load("models/predict_temp_hum_model.pkl")
        yield_model = joblib.load("models/predict_xgb_yield_model.pkl")
        rainfall_model = joblib.load("models/predict_monthly_rainfall_xgb_model.pkl")

        le_state = joblib.load("models/predict_state_encoder.pkl")
        le_soil = joblib.load("models/predict_soil_encoder.pkl")
        le_crop = joblib.load("models/predict_crop_encoder.pkl")

        print("Models loaded successfully")


def full_prediction(user_state, user_soil, user_crop, user_area, predict_year):

    load_models()

    try:

        state_enc = le_state.transform([user_state])[0]
        soil_enc = le_soil.transform([user_soil])[0]
        crop_enc = le_crop.transform([user_crop])[0]

        # ----------------------------
        # Rainfall Prediction
        # ----------------------------
        rainfall_input = pd.DataFrame(
            [[state_enc, predict_year]],
            columns=["state_enc", "year"]
        )

        start = time.time()
        monthly_rainfall = rainfall_model.predict(rainfall_input)[0]
        print(f"Rainfall prediction: {time.time()-start:.3f} sec")

        annual_rainfall = float(np.sum(monthly_rainfall))

        months = [
            "JAN","FEB","MAR","APR","MAY","JUN",
            "JUL","AUG","SEP","OCT","NOV","DEC"
        ]

        rainfall_df = pd.DataFrame({
            "Month": months,
            "Rainfall_mm": monthly_rainfall
        })

        top_months = rainfall_df.sort_values(
            by="Rainfall_mm",
            ascending=False
        ).head(3)

        best_months_text = ", ".join(top_months["Month"])

        # ----------------------------
        # Temperature & Humidity
        # ----------------------------
        temp_hum_input = pd.DataFrame(
            [[state_enc, soil_enc, crop_enc, annual_rainfall]],
            columns=[
                "STATE_ENC",
                "SOIL_ENC",
                "CROP_ENC",
                "RAINFALL"
            ]
        )

        start = time.time()
        pred_temp, pred_hum = temp_hum_model.predict(temp_hum_input)[0]
        print(f"Temp/Humidity: {time.time()-start:.3f} sec")

        # ----------------------------
        # NPK
        # ----------------------------
        npk_input = pd.DataFrame(
            [[
                state_enc,
                soil_enc,
                crop_enc,
                pred_temp,
                pred_hum,
                annual_rainfall,
                6.5
            ]],
            columns=[
                "STATE_ENC",
                "SOIL_ENC",
                "CROP_ENC",
                "TEMPERATURE",
                "HUMIDITY",
                "RAINFALL",
                "ph"
            ]
        )

        start = time.time()
        pred_N, pred_P, pred_K = npk_model.predict(npk_input)[0]
        print(f"NPK: {time.time()-start:.3f} sec")

        # ----------------------------
        # Yield
        # ----------------------------
        yield_input = pd.DataFrame([{
            "crop_enc": crop_enc,
            "year": predict_year,
            "season_enc": 1,
            "state_enc": state_enc,
            "area": user_area,
            "annual rainfall": annual_rainfall
        }])

        start = time.time()
        predicted_yield = float(yield_model.predict(yield_input)[0])
        print(f"Yield: {time.time()-start:.3f} sec")

        return {
            "user_state": user_state,
            "user_soil": user_soil,
            "user_crop": user_crop,
            "user_area": user_area,
            "annual_rainfall": annual_rainfall,
            "pred_temp": pred_temp,
            "pred_hum": pred_hum,
            "pred_N": pred_N,
            "pred_P": pred_P,
            "pred_K": pred_K,
            "predicted_yield": predicted_yield,
            "best_months_text": best_months_text,
            "rainfall_data": {
                "months": months,
                "values": monthly_rainfall.tolist()
            },
            "best_months": top_months["Month"].tolist()
        }

    except Exception as e:
        print("Prediction Error:", str(e))
        raise