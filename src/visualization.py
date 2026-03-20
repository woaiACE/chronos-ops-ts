import matplotlib.pyplot as plt
import pandas as pd


def plot_evaluation(bt_results_call, bt_results_ticket, output_path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    axes[0].plot(bt_results_call['dates'], bt_results_call['actual'], label='Actual', marker='o')
    axes[0].plot(bt_results_call['dates'], bt_results_call['predicted'], label='Predicted (p50)', marker='x')
    axes[0].set_title(
        f"Rolling Backtest ({bt_results_call['windows_used']} windows x {bt_results_call['horizon']} days) - Call Volume"
    )
    axes[0].set_ylabel('Volume')
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(bt_results_ticket['dates'], bt_results_ticket['actual'], label='Actual', marker='o', color='green')
    axes[1].plot(bt_results_ticket['dates'], bt_results_ticket['predicted'], label='Predicted (p50)', marker='x', color='orange')
    axes[1].set_title(
        f"Rolling Backtest ({bt_results_ticket['windows_used']} windows x {bt_results_ticket['horizon']} days) - Tickets Received"
    )
    axes[1].set_ylabel('Tickets')
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_future_from_csv(historical_csv, forecast_csv, output_path):
    hist_df = pd.read_csv(historical_csv)
    hist_df['date'] = pd.to_datetime(hist_df['date'])

    fc_df = pd.read_csv(forecast_csv)
    fc_df['date'] = pd.to_datetime(fc_df['date'])

    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    for i, (target, name, color) in enumerate([
        ('call_volume', 'Call Volume', 'blue'),
        ('tickets_received', 'Tickets Received', 'green')
    ]):
        ax = axes[i]
        target_hist = hist_df[['date', target]].dropna().tail(90)
        ax.plot(target_hist['date'], target_hist[target], label='Historical Actuals', color='black', alpha=0.5, marker='o', markersize=3)

        target_fc = fc_df[fc_df['target_name'] == target]
        if not target_fc.empty:
            ax.plot(target_fc['date'], target_fc['p50'], label='Predicted (p50)', color=color, linestyle='--', marker='x')
            ax.fill_between(
                target_fc['date'],
                target_fc['p10'],
                target_fc['p90'],
                color=color,
                alpha=0.2,
                label='80% Confidence Interval (p10-p90)\n*p90 is the business baseline for scheduling redundancy'
            )

        ax.set_title(f'Future Forecast - {name}')
        ax.set_ylabel('Count')
        ax.grid(True)
        ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_forecast_export_csv(forecast_csv, output_path):
    fc_df = pd.read_csv(forecast_csv)
    fc_df['date'] = pd.to_datetime(fc_df['date'])

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    for i, (target, name, color) in enumerate([
        ('call_volume', 'Call Volume (Export)', 'royalblue'),
        ('tickets_received', 'Tickets Received (Export)', 'seagreen')
    ]):
        ax = axes[i]
        target_fc = fc_df[fc_df['target_name'] == target].sort_values('date')
        if target_fc.empty:
            ax.set_title(f'{name} - 无数据')
            ax.grid(True)
            continue

        ax.plot(target_fc['date'], target_fc['p50'], label='P50 预测', color=color, linewidth=2)
        ax.fill_between(
            target_fc['date'],
            target_fc['p10'],
            target_fc['p90'],
            color=color,
            alpha=0.2,
            label='区间: P10-P90'
        )
        ax.set_title(name)
        ax.set_ylabel('数量')
        ax.grid(True)
        ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
