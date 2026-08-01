import numpy as np
import pandas as pd
import joblib
import os

# ✅ Load Models and Encoders Once
npk_model = joblib.load('models/predict_npk_xgb_model.pkl')
temp_hum_model = joblib.load('models/predict_temp_hum_model.pkl')
yield_model = joblib.load('models/predict_xgb_yield_model.pkl')
rainfall_model = joblib.load('models/predict_monthly_rainfall_xgb_model.pkl')

le_state = joblib.load('models/predict_state_encoder.pkl')
le_soil = joblib.load('models/predict_soil_encoder.pkl')
le_crop = joblib.load('models/predict_crop_encoder.pkl')


def full_prediction(user_state, user_soil, user_crop, user_area, predict_year):
    # Encode categorical inputs
    state_enc = le_state.transform([user_state])[0]
    soil_enc = le_soil.transform([user_soil])[0]
    crop_enc = le_crop.transform([user_crop])[0]

    # Predict Monthly Rainfall
    rainfall_input = pd.DataFrame([[state_enc, predict_year]], columns=['state_enc', 'year'])
    monthly_rainfall = rainfall_model.predict(rainfall_input)[0]
    annual_rainfall = np.sum(monthly_rainfall)

    # Rainfall DataFrame
    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
              'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    rainfall_df = pd.DataFrame({'Month': months, 'Rainfall_mm': monthly_rainfall})

    # Best cropping months
    top_months = rainfall_df.sort_values(by='Rainfall_mm', ascending=False).head(3)
    best_months_text = ", ".join(top_months['Month'].values)

    # Predict Temperature & Humidity
    temp_hum_input = pd.DataFrame([[state_enc, soil_enc, crop_enc, annual_rainfall]],
                                  columns=['STATE_ENC', 'SOIL_ENC', 'CROP_ENC', 'RAINFALL'])
    pred_temp, pred_hum = temp_hum_model.predict(temp_hum_input)[0]

    # Predict NPK
    ph_value = 6.5
    npk_input = pd.DataFrame([[state_enc, soil_enc, crop_enc, pred_temp, pred_hum, annual_rainfall, ph_value]],
                             columns=['STATE_ENC', 'SOIL_ENC', 'CROP_ENC', 'TEMPERATURE', 'HUMIDITY', 'RAINFALL', 'ph'])
    pred_N, pred_P, pred_K = npk_model.predict(npk_input)[0]

    # Predict Yield
    yield_input = pd.DataFrame([{
        'crop_enc': crop_enc,
        'year': predict_year,
        'season_enc': 1,  # assuming Kharif season
        'state_enc': state_enc,
        'area': user_area,
        'annual rainfall': annual_rainfall
    }])
    predicted_yield = yield_model.predict(yield_input)[0]

    # Prepare logging data
    log_data = {
        'State': user_state,
        'Soil': user_soil,
        'Crop': user_crop,
        'Area (hectares)': user_area,
        'Prediction Year': predict_year,
        'Predicted Nitrogen (%)': round(pred_N, 2),
        'Predicted Phosphorous (%)': round(pred_P, 2),
        'Predicted Potassium (%)': round(pred_K, 2),
        'Temperature (°C)': round(pred_temp, 2),
        'Humidity (%)': round(pred_hum, 2),
        'Annual Rainfall (mm)': round(annual_rainfall, 2),
        'Predicted Yield (tons/hectare)': round(predicted_yield, 2),
        'Best Cropping Months': best_months_text
    }

    log_df = pd.DataFrame([log_data])
    log_file = 'prediction_log.csv'

    # ✅ Save to CSV safely
    if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
        existing_log = pd.read_csv(log_file)
        updated_log = pd.concat([existing_log, log_df], ignore_index=True)
        updated_log.to_csv(log_file, index=False)
    else:
        log_df.to_csv(log_file, index=False)

    # ✅ Final result return
    return {
        'user_state': user_state,
        'user_soil': user_soil,
        'user_crop': user_crop,
        'user_area': user_area,
        'annual_rainfall': annual_rainfall,
        'pred_temp': pred_temp,
        'pred_hum': pred_hum,
        'pred_N': pred_N,
        'pred_P': pred_P,
        'pred_K': pred_K,
        'predicted_yield': predicted_yield,
        'best_months_text': best_months_text,
        'rainfall_data': {
            'months': months,
            'values': monthly_rainfall.tolist()
        },
        'best_months': top_months['Month'].tolist()
    }
