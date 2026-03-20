import os
import pandas as pd


def ensure_output_dirs(base_dir="outputs"):
    csv_dir = os.path.join(base_dir, "csv")
    png_dir = os.path.join(base_dir, "png")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    return csv_dir, png_dir


def export_forecast_csv(fut_call, fut_ticket, output_csv_path):
    export_rows = []
    for d, p10, p50, p90 in zip(fut_call['future_dates'], fut_call['p10'], fut_call['p50'], fut_call['p90']):
        export_rows.append({
            'date': d.strftime('%Y-%m-%d'),
            'target_name': 'call_volume',
            'p10': round(p10, 2),
            'p50': round(p50, 2),
            'p90': round(p90, 2),
        })

    for d, p10, p50, p90 in zip(fut_ticket['future_dates'], fut_ticket['p10'], fut_ticket['p50'], fut_ticket['p90']):
        export_rows.append({
            'date': d.strftime('%Y-%m-%d'),
            'target_name': 'tickets_received',
            'p10': round(p10, 2),
            'p50': round(p50, 2),
            'p90': round(p90, 2),
        })

    pd.DataFrame(export_rows).to_csv(output_csv_path, index=False)
    return output_csv_path


def export_dataframe_csv(df, output_csv_path):
    df.to_csv(output_csv_path, index=False)
    return output_csv_path
