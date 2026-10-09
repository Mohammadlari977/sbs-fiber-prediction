"""
SpinLab: solution blow spinning fiber-diameter prediction.

The eight machine-learning models are frozen (trained on the literature cross-validation set). Predictions can be
used zero-shot or after a local post-hoc calibration: the frozen models predict the user's anchor conditions and an
affine correction y_log,corrected = a * y_log,predicted + b is fitted to the measured ln(diameter). The base models
are not retrained and the literature data are not combined with user data. Inputs are checked against the domain
of the training conditions, and a warning is returned when a query lies outside it.
"""

import json
import math
import os
import pickle

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

BASE = os.path.dirname(os.path.abspath(__file__))
FEATURES = ['Polymer_conc_wt', 'intrinsic_viscosity_dLg', 'c_over_cstar', 'Solvent_ST_mNm', 'Solvent_BP_C',
            'Air_pressure_bar_std', 'Feed_rate_mLh_std', 'Working_distance_cm']
LABEL = {'ExtraTrees': 'Extra Trees', 'RandomForest': 'Random Forest', 'XGBoost': 'XGBoost', 'LightGBM': 'LightGBM',
         'CatBoost': 'CatBoost', 'SVR': 'SVR', 'ElasticNet': 'ElasticNet', 'KNN': 'KNN'}

# ------------------------------------------------------------------ models (scaling is part of each saved model)
with open(os.path.join(BASE, 'final_models_ml.pkl'), 'rb') as f:
    MODELS = pickle.load(f)
MODEL_NAMES = [n for n in LABEL if n in MODELS]

# ------------------------------------------------------------------ applicability domain
with open(os.path.join(BASE, 'applicability_domain.json')) as f:
    AD = json.load(f)
AD_MEAN, AD_SCALE = np.array(AD['scaler_mean']), np.array(AD['scaler_scale'])
AD_REF = np.array(AD['reference_points_standardized'])
AD_K, AD_THR = int(AD['k']), float(AD['knn_threshold_95pct'])
CV_MIN, CV_MAX = np.array(AD['cv_min']), np.array(AD['cv_max'])
SHORT = ['concentration', 'intrinsic viscosity', 'c/c*', 'surface tension', 'boiling point', 'air pressure',
         'feed rate', 'working distance']


def domain(X):
    """Mean distance to the 5 nearest unique training conditions (standardized inputs) and range check."""
    X = np.atleast_2d(np.asarray(X, float))
    Z = (X - AD_MEAN) / AD_SCALE
    d = np.sqrt(((Z[:, None, :] - AD_REF[None, :, :]) ** 2).sum(-1))
    dk = np.sort(d, axis=1)[:, :AD_K].mean(1)
    out = [[SHORT[j] for j in range(len(FEATURES)) if row[j] < CV_MIN[j] or row[j] > CV_MAX[j]] for row in X]
    return [{'inside': bool(dk[i] <= AD_THR), 'distance': round(float(dk[i]), 3), 'outside_range': out[i]}
            for i in range(len(X))]


# ------------------------------------------------------------------ lookups for the feature calculator
POLYS = {}
for _, r in pd.read_csv(os.path.join(BASE, 'mark_houwink_parameters.csv')).iterrows():
    POLYS[str(r['Polymer'])] = {'K': float(r['K']), 'a': float(r['a'])}
SOLVS = {}
for _, r in pd.read_csv(os.path.join(BASE, 'solvent_properties.csv'), keep_default_na=False).iterrows():
    try:
        st, bp, rho = float(r['Solvent_ST_mNm']), float(r['Solvent_BP_C']), float(r['Solvent_density_gmL'])
    except (TypeError, ValueError):
        continue
    if all(math.isfinite(v) for v in (st, bp, rho)):
        SOLVS[str(r['Solvent_system'])] = {'ST': st, 'BP': bp, 'rho': rho}


# ------------------------------------------------------------------ prediction and calibration
def predict_log(name, X):
    return np.asarray(MODELS[name].predict(np.atleast_2d(np.asarray(X, float))), float)


def fit_affine(p, y):
    """Least-squares a, b for y = a*p + b; offset only (a = 1) when the predictions do not vary."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    if len(p) >= 2 and np.ptp(p) > 1e-9:
        a, b = np.linalg.lstsq(np.c_[p, np.ones_like(p)], y, rcond=None)[0]
        return float(a), float(b)
    return 1.0, float(np.mean(y - p))


def parse_user(rows):
    u = np.asarray(rows, float)
    return u[:, :8], np.log(u[:, 8]), u[:, 8]


app = FastAPI(title='SpinLab')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])


@app.get('/', response_class=HTMLResponse)
async def index():
    with open(os.path.join(BASE, 'index.html'), encoding='utf-8') as f:
        return HTMLResponse(f.read())


@app.get('/api/health')
async def health():
    return {'status': 'ok', 'models': MODEL_NAMES}


@app.get('/api/config')
async def config():
    return {
        'polymers': sorted(POLYS),
        'solvents': dict(sorted(SOLVS.items())),
        'models': [{'key': n, 'label': LABEL[n], 'type': 'ML', 'available': True} for n in MODEL_NAMES],
        'training_ranges': {f: [float(lo), float(hi)] for f, lo, hi in zip(FEATURES, CV_MIN, CV_MAX)},
    }


@app.get('/api/polymer/{name}')
async def polymer(name: str):
    if name not in POLYS:
        return JSONResponse({'error': f'Not found: {name}'}, status_code=404)
    return POLYS[name]


@app.post('/api/predict')
async def predict(request: Request):
    """Zero-shot prediction. Body: {"features": [8 values]}"""
    body = await request.json()
    X = np.asarray(body['features'], float).reshape(1, -1)
    preds = {n: round(float(np.exp(predict_log(n, X)[0])), 1) for n in MODEL_NAMES}
    return {'predictions': preds, 'domain': domain(X)[0]}


@app.post('/api/calibrate')
async def calibrate(request: Request):
    """Local calibration of the frozen models.
    Body: {"user_data": [[conc, eta, c/c*, ST, BP, pressure, feed, WD, diameter_nm], ...]} (at least 3 rows).
    For each model: zero-shot predictions, leave-one-out calibrated predictions (affine correction fitted on the
    other anchors), and the affine correction fitted on all anchors."""
    body = await request.json()
    X, y_log, y_nm = parse_user(body['user_data'])
    n = len(X)
    if n < 3:
        return JSONResponse({'error': 'At least 3 calibration points are needed.'}, status_code=400)
    results = {}
    for name in MODEL_NAMES:
        p = predict_log(name, X)
        loo = np.empty(n)
        for i in range(n):
            m = np.arange(n) != i
            a, b = fit_affine(p[m], y_log[m])
            loo[i] = a * p[i] + b
        loo_nm = np.exp(loo)
        err = np.abs(y_nm - loo_nm) / y_nm * 100
        zs_nm = np.exp(p)
        a, b = fit_affine(p, y_log)
        results[name] = {
            'label': LABEL[name],
            'mape': round(float(err.mean()), 1),
            'loo_preds': [round(float(v), 1) for v in loo_nm],
            'errors': [round(float(v), 1) for v in err],
            'zeroshot_preds': [round(float(v), 1) for v in zs_nm],
            'zeroshot_mape': round(float((np.abs(y_nm - zs_nm) / y_nm * 100).mean()), 1),
            'slope': round(a, 3), 'intercept': round(b, 3),
        }
    best = min(results, key=lambda k: results[k]['mape'])
    dom = domain(X)
    return {
        'best_model': best, 'best_mape': results[best]['mape'], 'model_results': results,
        'actuals': [round(float(v), 1) for v in y_nm], 'n_models': len(results), 'n_user': n,
        'domain': {'points': dom, 'n_outside': int(sum(not d['inside'] for d in dom))},
    }


def _range(lo, hi, step):
    v, out = lo, []
    while v <= hi + 1e-9:
        out.append(round(v, 2))
        v += step
    return out


@app.post('/api/predict_grid')
async def predict_grid(request: Request):
    """Concentration x pressure grid. Zero-shot, or calibrated with the affine correction when user_data is given."""
    body = await request.json()
    eta, st, bp = body['eta'], body['solvent_st'], body['solvent_bp']
    feed, wd, cstar = body['feed_rate'], body['working_distance'], body['cstar']
    concs = _range(*body.get('conc_range', [2, 8, 0.5]))
    press = _range(*body.get('pressure_range', [0.5, 5, 0.5]))
    X = np.array([[c, eta, c / cstar, st, bp, p, feed, wd] for c in concs for p in press], float)
    name = body.get('model_name')
    if name not in MODELS:
        name = MODEL_NAMES[0]
    pred = predict_log(name, X)
    user = body.get('user_data')
    mode = 'zero-shot'
    if user and len(user) >= 2:
        Xu, yu, _ = parse_user(user)
        a, b = fit_affine(predict_log(name, Xu), yu)
        pred = a * pred + b
        mode = 'calibrated'
    nm = np.exp(pred)
    dom = domain(X)
    k = len(press)
    return {
        'model': LABEL[name], 'mode': mode, 'concentrations': concs, 'pressures': press,
        'grid': [[round(float(v), 1) for v in nm[i * k:(i + 1) * k]] for i in range(len(concs))],
        'inside': [[d['inside'] for d in dom[i * k:(i + 1) * k]] for i in range(len(concs))],
        'n_outside': int(sum(not d['inside'] for d in dom)), 'n_cells': len(dom),
    }


@app.post('/api/predict_calibrated')
async def predict_calibrated(request: Request):
    """Predictions for new conditions with one model and the affine correction fitted on user_data."""
    body = await request.json()
    name = body.get('model_name')
    if name not in MODELS:
        return JSONResponse({'error': f'Model {name} not found'}, status_code=400)
    Xu, yu, _ = parse_user(body['user_data'])
    a, b = fit_affine(predict_log(name, Xu), yu)
    Xp = np.asarray(body['predict_features'], float)
    nm = np.exp(a * predict_log(name, Xp) + b)
    return {'model': LABEL[name], 'predictions': [round(float(v), 1) for v in nm], 'slope': round(a, 3),
            'intercept': round(b, 3), 'domain': domain(Xp)}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=7860)
