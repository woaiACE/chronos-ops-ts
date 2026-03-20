import numpy as np
import pandas as pd
import time
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


STATE_TO_CODE = {
    'workday': 0,
    'weekend': 1,
    'holiday_or_makeup': 2,
}


def calculate_metrics(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred)
    smape_array = np.zeros_like(diff, dtype=float)
    nonzero_mask = denominator > 0
    smape_array[nonzero_mask] = diff[nonzero_mask] / denominator[nonzero_mask]

    smape = np.mean(smape_array)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return smape, rmse


def get_operational_state(dt, holiday_fn=None, makeup_fn=None):
    dt = pd.to_datetime(dt)
    if holiday_fn is not None and bool(holiday_fn(dt)):
        return 'holiday_or_makeup'
    if makeup_fn is not None and bool(makeup_fn(dt)):
        return 'holiday_or_makeup'
    if dt.dayofweek >= 5:
        return 'weekend'
    return 'workday'


def build_holiday_proximity_features(date_series, holiday_fn=None, makeup_fn=None, prefix=None):
    dates = pd.Series(pd.to_datetime(date_series), copy=False).reset_index(drop=True)
    feature_prefix = f'{prefix}_' if prefix else ''
    n_rows = len(dates)

    holiday_mask = np.zeros(n_rows, dtype=bool)
    makeup_mask = np.zeros(n_rows, dtype=bool)
    for idx, dt in enumerate(dates):
        holiday_mask[idx] = bool(holiday_fn(dt)) if holiday_fn is not None else False
        makeup_mask[idx] = bool(makeup_fn(dt)) if makeup_fn is not None else False

    # Workday includes weekdays plus official makeup workdays.
    weekday_mask = (dates.dt.dayofweek < 5).to_numpy()
    workday_mask = np.logical_or(weekday_mask, makeup_mask)

    pre_holiday_n_day = np.zeros(n_rows, dtype=int)
    post_holiday_workday_n = np.zeros(n_rows, dtype=int)

    start = 0
    while start < n_rows:
        if not holiday_mask[start]:
            start += 1
            continue

        end = start
        while end + 1 < n_rows and holiday_mask[end + 1]:
            end += 1

        holiday_len = end - start + 1
        if holiday_len >= 2:
            for n in range(1, 3):
                idx = start - n
                if idx >= 0 and not holiday_mask[idx]:
                    pre_holiday_n_day[idx] = n

            found = 0
            probe = end + 1
            while probe < n_rows and found < 3:
                if (not holiday_mask[probe]) and workday_mask[probe]:
                    found += 1
                    post_holiday_workday_n[probe] = found
                probe += 1

        start = end + 1

    return pd.DataFrame({
        f'{feature_prefix}pre_holiday_n_day': pre_holiday_n_day,
        f'{feature_prefix}is_pre_holiday_1d': (pre_holiday_n_day == 1).astype(int),
        f'{feature_prefix}is_pre_holiday_2d': (pre_holiday_n_day == 2).astype(int),
        f'{feature_prefix}post_holiday_workday_n': post_holiday_workday_n,
        f'{feature_prefix}is_post_holiday_workday_1': (post_holiday_workday_n == 1).astype(int),
        f'{feature_prefix}is_post_holiday_workday_2': (post_holiday_workday_n == 2).astype(int),
        f'{feature_prefix}is_post_holiday_workday_3': (post_holiday_workday_n == 3).astype(int),
    })


def build_calendar_features(date_series, prefix=None, holiday_fn=None, makeup_fn=None):
    dates = pd.Series(pd.to_datetime(date_series), copy=False).reset_index(drop=True)
    feature_prefix = f'{prefix}_' if prefix else ''

    day_of_week = dates.dt.dayofweek
    day_of_year = dates.dt.dayofyear
    states = dates.apply(lambda dt: get_operational_state(dt, holiday_fn=holiday_fn, makeup_fn=makeup_fn))

    features = pd.DataFrame(index=dates.index)
    features[f'{feature_prefix}day_of_week'] = day_of_week
    features[f'{feature_prefix}is_weekend'] = (day_of_week >= 5).astype(int)
    features[f'{feature_prefix}day_of_month'] = dates.dt.day
    features[f'{feature_prefix}month'] = dates.dt.month
    features[f'{feature_prefix}week_of_year'] = dates.dt.isocalendar().week.astype(int)
    features[f'{feature_prefix}dow_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    features[f'{feature_prefix}dow_cos'] = np.cos(2 * np.pi * day_of_week / 7)
    features[f'{feature_prefix}doy_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
    features[f'{feature_prefix}doy_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)
    features[f'{feature_prefix}is_holiday'] = dates.apply(
        lambda dt: int(bool(holiday_fn(dt))) if holiday_fn is not None else 0
    )
    features[f'{feature_prefix}is_makeup_workday'] = dates.apply(
        lambda dt: int(bool(makeup_fn(dt))) if makeup_fn is not None else 0
    )
    features[f'{feature_prefix}state_code'] = states.map(STATE_TO_CODE).astype(int)

    month_end_day = dates.dt.days_in_month
    features[f'{feature_prefix}is_month_end_settlement'] = (dates.dt.day >= 26).astype(int)
    features[f'{feature_prefix}days_to_month_end'] = (month_end_day - dates.dt.day).astype(int)
    billing_days = {1, 5, 10, 15, 20, 25}
    features[f'{feature_prefix}is_billing_cycle_day'] = dates.dt.day.apply(
        lambda d: int((int(d) in billing_days) or (int(d) >= 26))
    )

    proximity_features = build_holiday_proximity_features(
        dates,
        holiday_fn=holiday_fn,
        makeup_fn=makeup_fn,
        prefix=prefix,
    )
    features = pd.concat([features, proximity_features], axis=1)
    return features


def build_supervised_features(df, target_col, reference_col=None, holiday_fn=None, makeup_fn=None):
    features = pd.DataFrame(index=df.index)

    date_series = pd.to_datetime(df['date']).reset_index(drop=True)
    day_of_week = date_series.dt.dayofweek
    day_of_year = date_series.dt.dayofyear

    features['day_of_week'] = day_of_week
    features['is_weekend'] = (day_of_week >= 5).astype(int)
    features['day_of_month'] = date_series.dt.day
    features['month'] = date_series.dt.month
    features['week_of_year'] = date_series.dt.isocalendar().week.astype(int)
    features['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    features['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7)
    features['doy_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
    features['doy_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)
    features['is_holiday'] = date_series.apply(
        lambda dt: int(bool(holiday_fn(dt))) if holiday_fn is not None else 0
    )
    features['is_makeup_workday'] = date_series.apply(
        lambda dt: int(bool(makeup_fn(dt))) if makeup_fn is not None else 0
    )
    features['operational_state_code'] = date_series.apply(
        lambda dt: STATE_TO_CODE[get_operational_state(dt, holiday_fn=holiday_fn, makeup_fn=makeup_fn)]
    )
    month_end_day = date_series.dt.days_in_month
    features['is_month_end_settlement'] = (date_series.dt.day >= 26).astype(int)
    features['days_to_month_end'] = (month_end_day - date_series.dt.day).astype(int)
    billing_days = {1, 5, 10, 15, 20, 25}
    features['is_billing_cycle_day'] = date_series.dt.day.apply(
        lambda d: int((int(d) in billing_days) or (int(d) >= 26))
    )

    proximity_features = build_holiday_proximity_features(
        date_series,
        holiday_fn=holiday_fn,
        makeup_fn=makeup_fn,
        prefix=None,
    )
    for column in proximity_features.columns:
        features[column] = proximity_features[column].to_numpy()

    target = df[target_col].astype(float)

    for lag in range(1, 15):
        features[f'{target_col}_lag_{lag}'] = target.shift(lag)

    for window in [3, 7, 14, 30]:
        features[f'{target_col}_roll_mean_{window}'] = target.shift(1).rolling(window).mean()
        features[f'{target_col}_roll_std_{window}'] = target.shift(1).rolling(window).std()

    features[f'{target_col}_diff_1'] = target.shift(1) - target.shift(2)
    features[f'{target_col}_diff_7'] = target.shift(1) - target.shift(8)

    if reference_col and reference_col in df.columns:
        ref = df[reference_col].astype(float)
        features[f'{reference_col}_lag_1'] = ref.shift(1)
        features[f'{reference_col}_lag_7'] = ref.shift(7)
        features[f'{reference_col}_lag_14'] = ref.shift(14)

        for window in [3, 7, 14, 30]:
            features[f'{reference_col}_roll_mean_{window}'] = ref.shift(1).rolling(window).mean()
            features[f'{reference_col}_roll_std_{window}'] = ref.shift(1).rolling(window).std()

        ratio_raw = ref / np.clip(target, 1e-6, None)
        features['call_to_ticket_ratio_lag1'] = ratio_raw.shift(1)
        features['call_to_ticket_ratio_lag7'] = ratio_raw.shift(7)

    if target_col == 'tickets_received' and 'tickets_resolved' in df.columns:
        resolved = df['tickets_resolved'].astype(float)
        backlog = target - resolved
        features['estimated_backlog_lag1'] = backlog.shift(1)
        features['estimated_backlog_roll_mean_7'] = backlog.shift(1).rolling(7).mean()

    return features


def build_direct_multistep_feature_frame(
    df,
    target_col,
    horizon,
    reference_col=None,
    holiday_fn=None,
    makeup_fn=None,
):
    base_features = build_supervised_features(
        df,
        target_col,
        reference_col=reference_col,
        holiday_fn=holiday_fn,
        makeup_fn=makeup_fn,
    ).reset_index(drop=True)
    origin_dates = pd.Series(pd.to_datetime(df['date'])).reset_index(drop=True)
    target = df[target_col].astype(float).reset_index(drop=True)

    historical_frames = []
    for lead in range(1, horizon + 1):
        forecast_dates = origin_dates + pd.to_timedelta(lead, unit='D')
        lead_frame = base_features.copy()
        lead_frame.insert(0, 'origin_date', origin_dates)
        lead_frame['forecast_date'] = forecast_dates
        lead_frame['lead'] = lead
        lead_frame['row_type'] = 'train'
        lead_frame['target'] = target.shift(-lead)
        forecast_calendar = build_calendar_features(
            forecast_dates,
            prefix='forecast',
            holiday_fn=holiday_fn,
            makeup_fn=makeup_fn,
        )
        lead_frame = pd.concat([lead_frame, forecast_calendar], axis=1)
        historical_frames.append(lead_frame)

    future_frames = []
    base_last = base_features.iloc[[-1]].reset_index(drop=True)
    last_date = origin_dates.iloc[-1]
    for lead in range(1, horizon + 1):
        forecast_date = last_date + pd.Timedelta(days=lead)
        future_frame = base_last.copy()
        future_frame.insert(0, 'origin_date', pd.Series([last_date]))
        future_frame['forecast_date'] = pd.Series([forecast_date])
        future_frame['lead'] = lead
        future_frame['row_type'] = 'future'
        future_frame['target'] = np.nan
        forecast_calendar = build_calendar_features(
            pd.Series([forecast_date]),
            prefix='forecast',
            holiday_fn=holiday_fn,
            makeup_fn=makeup_fn,
        )
        future_frame = pd.concat([future_frame, forecast_calendar], axis=1)
        future_frames.append(future_frame)

    historical_frame = pd.concat(historical_frames, ignore_index=True)
    future_frame = pd.concat(future_frames, ignore_index=True)
    return historical_frame, future_frame


def _build_candidate_models():
    return {
        'ridge': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Ridge(alpha=2.0)),
        ]),
        'hgb': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', HistGradientBoostingRegressor(
                max_iter=220,
                learning_rate=0.02,
                max_depth=3,
                min_samples_leaf=8,
                l2_regularization=2.0,
                random_state=42,
            )),
        ]),
    }


def _fallback_point_prediction(feature_row, target_col):
    candidate_columns = [
        f'{target_col}_lag_7',
        f'{target_col}_roll_mean_7',
        f'{target_col}_lag_1',
        f'{target_col}_roll_mean_14',
        f'{target_col}_roll_mean_30',
    ]
    for column in candidate_columns:
        value = feature_row.get(column)
        if pd.notna(value):
            return max(0.0, float(value))
    return 0.0


def _fit_direct_model(lead_frame, feature_cols):
    clean_frame = lead_frame.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols + ['target'])
    if clean_frame.empty:
        return None, 'fallback', None

    X = clean_frame[feature_cols]
    y = clean_frame['target'].astype(float)

    if len(clean_frame) < 36:
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Ridge(alpha=2.0)),
        ])
        model.fit(X, np.log1p(np.clip(y, 0.0, None)))
        return model, 'ridge_small_sample', None

    # On long histories HGB selection can be very slow on Windows and block the whole pipeline.
    # Use a fast ridge path to keep end-to-end forecasting (including plotting) responsive.
    if len(clean_frame) >= 500:
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Ridge(alpha=2.0)),
        ])
        model.fit(X, np.log1p(np.clip(y, 0.0, None)))
        return model, 'ridge_large_sample', None

    val_size = min(28, max(7, len(clean_frame) // 8))
    train_size = len(clean_frame) - val_size
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    X_val = X.iloc[train_size:]
    y_val = y.iloc[train_size:]

    y_train_log = np.log1p(np.clip(y_train, 0.0, None))

    best_model = None
    best_name = None
    best_smape = float('inf')

    for name, candidate in _build_candidate_models().items():
        model = clone(candidate)
        model.fit(X_train, y_train_log)
        val_pred = np.maximum(0.0, np.expm1(model.predict(X_val)))
        val_smape, _ = calculate_metrics(y_val.to_numpy(), val_pred)
        if val_smape < best_smape:
            best_smape = val_smape
            best_model = model
            best_name = name

    y_full_log = np.log1p(np.clip(y, 0.0, None))
    best_model.fit(X, y_full_log)
    return best_model, best_name, float(best_smape)


def train_feature_model(train_df, target_col, reference_col=None, holiday_fn=None, makeup_fn=None):
    features = build_supervised_features(
        train_df,
        target_col,
        reference_col=reference_col,
        holiday_fn=holiday_fn,
        makeup_fn=makeup_fn,
    )
    target = train_df[target_col].astype(float)

    supervised = features.copy()
    supervised[target_col] = target.values
    supervised = supervised.replace([np.inf, -np.inf], np.nan).dropna()

    if len(supervised) < 120:
        raise ValueError(
            f"Not enough samples to train feature model for {target_col}. "
            "Need at least 120 rows after feature construction."
        )

    feature_cols = [col for col in supervised.columns if col != target_col]
    X = supervised[feature_cols]
    y = supervised[target_col]

    selector_scaler = StandardScaler()
    X_scaled = selector_scaler.fit_transform(X)
    selector_estimator = Lasso(alpha=0.001, max_iter=20000)
    selector_estimator.fit(X_scaled, y)

    selected_features = [
        name for name, coef in zip(feature_cols, np.abs(selector_estimator.coef_))
        if coef > 1e-6
    ]

    if len(selected_features) < 8:
        corr_scores = X.corrwith(y).abs().fillna(0.0)
        selected_features = corr_scores.sort_values(ascending=False).head(min(20, len(corr_scores))).index.tolist()

    X_selected = X[selected_features]
    val_size = min(28, max(7, len(X_selected) // 8))
    train_size = len(X_selected) - val_size

    X_train = X_selected.iloc[:train_size]
    y_train = y.iloc[:train_size]
    X_val = X_selected.iloc[train_size:]
    y_val = y.iloc[train_size:]

    y_train_log = np.log1p(np.clip(y_train, 0.0, None))

    candidate_models = {
        'ridge': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Ridge(alpha=2.0)),
        ]),
        'hgb': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', HistGradientBoostingRegressor(
                max_iter=220,
                learning_rate=0.02,
                max_depth=3,
                min_samples_leaf=8,
                l2_regularization=2.0,
                random_state=42,
            )),
        ]),
    }

    best_name = None
    best_model = None
    best_smape = float('inf')

    for name, candidate in candidate_models.items():
        candidate.fit(X_train, y_train_log)
        val_pred_log = candidate.predict(X_val)
        val_pred = np.maximum(0.0, np.expm1(val_pred_log))
        val_smape, _ = calculate_metrics(y_val.to_numpy(), val_pred)
        if val_smape < best_smape:
            best_smape = val_smape
            best_name = name
            best_model = candidate

    y_full_log = np.log1p(np.clip(y, 0.0, None))
    best_model.fit(X_selected, y_full_log)

    return {
        'model': best_model,
        'selected_features': selected_features,
        'selected_model_name': best_name,
        'use_log_transform': True,
    }


def forecast_direct_multistep(
    train_df,
    target_col,
    horizon,
    reference_col=None,
    holiday_fn=None,
    makeup_fn=None,
    return_feature_frame=False,
    progress_prefix=None,
    log_every=5,
    enable_progress_log=True,
):
    start_ts = time.perf_counter()
    if progress_prefix is None:
        progress_prefix = target_col

    if enable_progress_log:
        print(
            f"[Direct] {progress_prefix}: 开始多步监督分支，horizon={horizon}，"
            f"reference_col={reference_col if reference_col else 'None'}"
        )

    historical_frame, future_frame = build_direct_multistep_feature_frame(
        train_df,
        target_col,
        horizon,
        reference_col=reference_col,
        holiday_fn=holiday_fn,
        makeup_fn=makeup_fn,
    )

    meta_columns = {'origin_date', 'forecast_date', 'lead', 'row_type', 'target'}
    feature_cols = [column for column in historical_frame.columns if column not in meta_columns]

    predictions = []
    model_summaries = []
    log_every = max(1, int(log_every))

    for lead in range(1, horizon + 1):
        lead_train = historical_frame[historical_frame['lead'] == lead].reset_index(drop=True)
        lead_future = future_frame[future_frame['lead'] == lead].reset_index(drop=True)

        model, model_name, val_smape = _fit_direct_model(lead_train, feature_cols)
        if model is None:
            pred_value = _fallback_point_prediction(lead_future.iloc[0], target_col)
        else:
            future_features = lead_future[feature_cols].replace([np.inf, -np.inf], np.nan)
            if future_features.isna().any().any():
                pred_value = _fallback_point_prediction(lead_future.iloc[0], target_col)
            else:
                pred_value = max(0.0, float(np.expm1(model.predict(future_features)[0])))

        predictions.append(pred_value)
        model_summaries.append({
            'lead': lead,
            'model_name': model_name,
            'val_smape': val_smape,
        })

        if enable_progress_log and (lead == 1 or lead % log_every == 0 or lead == horizon):
            elapsed = time.perf_counter() - start_ts
            print(
                f"[Direct] {progress_prefix}: lead {lead}/{horizon} 已完成，"
                f"当前模型={model_name}，累计耗时={elapsed:.1f}s"
            )

    result = {
        'predictions': np.asarray(predictions, dtype=float),
        'model_summaries': pd.DataFrame(model_summaries),
    }
    if return_feature_frame:
        result['feature_frame'] = pd.concat([historical_frame, future_frame], ignore_index=True)

    if enable_progress_log:
        elapsed = time.perf_counter() - start_ts
        print(f"[Direct] {progress_prefix}: 多步监督分支完成，总耗时={elapsed:.1f}s")

    return result
