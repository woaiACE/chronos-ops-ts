import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
import numpy as np
import pandas as pd


BUCKET_DISPLAY_NAMES = {
    'workday_normal': '普通工作日',
    'weekend': '周末',
    'holiday': '法定节假日',
    'post_holiday_workday_1_3': '节后1-3个工作日',
    'makeup_workday': '补班日',
    'other': '补班日',
}

SERIES_DISPLAY_NAMES = {
    'call_volume': '电话量',
    'tickets_received': '工单量',
}

STAGE_DISPLAY_NAMES = {
    'direct_fusion': '基线融合',
    'residual': '偏差修正',
    'leadwise': '预测步长修正',
}

_FONT_CONFIGURED = False


def _configure_chinese_font():
    global _FONT_CONFIGURED
    if _FONT_CONFIGURED:
        return

    preferred_fonts = [
        'Microsoft YaHei',
        'SimHei',
        'Noto Sans CJK SC',
        'Source Han Sans SC',
        'WenQuanYi Zen Hei',
        'Arial Unicode MS',
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    selected_fonts = [font_name for font_name in preferred_fonts if font_name in available_fonts]
    fallback_fonts = ['DejaVu Sans']

    if selected_fonts:
        rcParams['font.sans-serif'] = selected_fonts + fallback_fonts
    else:
        rcParams['font.sans-serif'] = fallback_fonts
    rcParams['font.family'] = 'sans-serif'
    rcParams['axes.unicode_minus'] = False
    _FONT_CONFIGURED = True


def _normalize_bucket_name(bucket_name):
    bucket_name = str(bucket_name)
    if bucket_name == 'other':
        return 'makeup_workday'
    return bucket_name


def _series_display_name(series_name):
    return SERIES_DISPLAY_NAMES.get(str(series_name), str(series_name))


def _flatten_backtest_series(bt_results):
    windows = bt_results.get('all_windows', []) if isinstance(bt_results, dict) else []
    if not windows:
        return bt_results['dates'], bt_results['actual'], bt_results['predicted']

    ordered_windows = sorted(windows, key=lambda w: int(w.get('window_index', 0)), reverse=True)
    dates = np.concatenate([np.asarray(window['dates']) for window in ordered_windows])
    actual = np.concatenate([np.asarray(window['actual'], dtype=float) for window in ordered_windows])
    predicted = np.concatenate([np.asarray(window['predicted'], dtype=float) for window in ordered_windows])
    return dates, actual, predicted


def plot_evaluation(bt_results_call, bt_results_ticket, output_path):
    _configure_chinese_font()
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    call_dates, call_actual, call_predicted = _flatten_backtest_series(bt_results_call)
    ticket_dates, ticket_actual, ticket_predicted = _flatten_backtest_series(bt_results_ticket)

    axes[0].plot(call_dates, call_actual, label='实际值', marker='o')
    axes[0].plot(call_dates, call_predicted, label='预测值(P50)', marker='x')
    axes[0].set_title(
        f"电话量回测结果（{bt_results_call['windows_used']}个窗口 x {bt_results_call['horizon']}天）"
    )
    axes[0].set_ylabel('数量')
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(ticket_dates, ticket_actual, label='实际值', marker='o', color='green')
    axes[1].plot(ticket_dates, ticket_predicted, label='预测值(P50)', marker='x', color='orange')
    axes[1].set_title(
        f"工单量回测结果（{bt_results_ticket['windows_used']}个窗口 x {bt_results_ticket['horizon']}天）"
    )
    axes[1].set_ylabel('数量')
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_future_from_csv(historical_csv, forecast_csv, output_path):
    _configure_chinese_font()
    hist_df = pd.read_csv(historical_csv)
    hist_df['date'] = pd.to_datetime(hist_df['date'])

    fc_df = pd.read_csv(forecast_csv)
    fc_df['date'] = pd.to_datetime(fc_df['date'])

    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    for i, (target, name, color) in enumerate([
        ('call_volume', '电话量', 'blue'),
        ('tickets_received', '工单量', 'green')
    ]):
        ax = axes[i]
        target_hist = hist_df[['date', target]].dropna().tail(90)
        ax.plot(target_hist['date'], target_hist[target], label='历史实际值', color='black', alpha=0.5, marker='o', markersize=3)

        target_fc = fc_df[fc_df['target_name'] == target]
        if not target_fc.empty:
            ax.plot(target_fc['date'], target_fc['p50'], label='预测值(P50)', color=color, linestyle='--', marker='x')
            ax.fill_between(
                target_fc['date'],
                target_fc['p10'],
                target_fc['p90'],
                color=color,
                alpha=0.2,
                label='80%预测区间(P10-P90)\nP90可作为业务排班冗余参考线'
            )

        ax.set_title(f'{name}未来预测')
        ax.set_ylabel('数量')
        ax.grid(True)
        ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_forecast_export_csv(forecast_csv, output_path):
    _configure_chinese_font()
    fc_df = pd.read_csv(forecast_csv)
    fc_df['date'] = pd.to_datetime(fc_df['date'])

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    for i, (target, name, color) in enumerate([
        ('call_volume', '电话量导出视图', 'royalblue'),
        ('tickets_received', '工单量导出视图', 'seagreen')
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


def plot_tuning_report(report_df, output_path):
    _configure_chinese_font()
    if report_df is None or report_df.empty:
        return

    report_df = report_df.copy()
    series_col = 'series_name' if 'series_name' in report_df.columns else 'target_name'
    report_df['label'] = report_df[series_col].map(_series_display_name) + ':' + report_df['stage'].map(lambda value: STAGE_DISPLAY_NAMES.get(str(value), str(value)))
    report_df = report_df.sort_values([series_col, 'stage']).reset_index(drop=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    x = range(len(report_df))

    axes[0].bar(x, report_df['smape_before'], label='调整前', alpha=0.75)
    axes[0].bar(x, report_df['smape_after'], label='调整后', alpha=0.75)
    axes[0].set_ylabel('sMAPE')
    axes[0].set_title('调参效果对比 - sMAPE前后变化')
    axes[0].grid(True, axis='y')
    axes[0].legend()

    axes[1].bar(x, report_df['rmse_before'], label='调整前', alpha=0.75)
    axes[1].bar(x, report_df['rmse_after'], label='调整后', alpha=0.75)
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('调参效果对比 - RMSE前后变化')
    axes[1].grid(True, axis='y')
    axes[1].legend()

    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(report_df['label'], rotation=35, ha='right')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_monitor_bucket_report(report_df, output_path):
    _configure_chinese_font()
    if report_df is None or report_df.empty:
        return

    scoped_df = report_df.loc[report_df['scope'] == 'recent'].copy()
    if scoped_df.empty:
        scoped_df = report_df.copy()

    if scoped_df.empty:
        return

    bucket_order = [
        'workday_normal',
        'weekend',
        'holiday',
        'post_holiday_workday_1_3',
        'makeup_workday',
    ]
    scoped_df['bucket'] = scoped_df['bucket'].map(_normalize_bucket_name)
    scoped_df['bucket_label'] = scoped_df['bucket'].map(lambda value: BUCKET_DISPLAY_NAMES.get(str(value), str(value)))
    scoped_df['bucket'] = pd.Categorical(scoped_df['bucket'], categories=bucket_order, ordered=True)
    scoped_df = scoped_df.sort_values(['series_name', 'bucket']).reset_index(drop=True)
    series_names = list(scoped_df['series_name'].dropna().unique())
    if not series_names:
        return

    fig, axes = plt.subplots(len(series_names), 2, figsize=(16, max(5 * len(series_names), 6)))
    if len(series_names) == 1:
        axes = [axes]

    for row_idx, series_name in enumerate(series_names):
        series_df = scoped_df.loc[scoped_df['series_name'] == series_name].copy()
        labels = [str(item) for item in series_df['bucket_label']]

        smape_ax, bias_ax = axes[row_idx]
        smape_ax.bar(labels, series_df['smape'], color='#4472c4', alpha=0.85)
        smape_ax.set_title(f'{_series_display_name(series_name)} - 最近分场景sMAPE')
        smape_ax.set_ylabel('sMAPE')
        smape_ax.grid(True, axis='y', alpha=0.25)
        smape_ax.tick_params(axis='x', rotation=25)

        bias_ax.bar(labels, series_df['bias'], color='#c55a11', alpha=0.85)
        bias_ax.axhline(0.0, color='black', linewidth=1)
        bias_ax.set_title(f'{_series_display_name(series_name)} - 最近分场景偏差')
        bias_ax.set_ylabel('Bias')
        bias_ax.grid(True, axis='y', alpha=0.25)
        bias_ax.tick_params(axis='x', rotation=25)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_monitor_bucket_sample_scope(monitor_report_df, history_profile_df, output_path):
    _configure_chinese_font()
    if monitor_report_df is None or monitor_report_df.empty:
        return
    if history_profile_df is None or history_profile_df.empty:
        return

    bt_recent_df = monitor_report_df.loc[monitor_report_df['scope'] == 'recent'].copy()
    if bt_recent_df.empty:
        bt_recent_df = monitor_report_df.copy()

    hist_recent_df = history_profile_df.loc[history_profile_df['scope'] == 'history_recent'].copy()
    hist_all_df = history_profile_df.loc[history_profile_df['scope'] == 'history_all'].copy()
    if hist_recent_df.empty:
        hist_recent_df = hist_all_df.copy()
    if hist_all_df.empty:
        hist_all_df = hist_recent_df.copy()

    if bt_recent_df.empty or hist_recent_df.empty or hist_all_df.empty:
        return

    bt_recent_df['bucket'] = bt_recent_df['bucket'].map(_normalize_bucket_name)
    hist_recent_df['bucket'] = hist_recent_df['bucket'].map(_normalize_bucket_name)
    hist_all_df['bucket'] = hist_all_df['bucket'].map(_normalize_bucket_name)

    merged_df = pd.merge(
        bt_recent_df[['series_name', 'bucket', 'sample_size']],
        hist_recent_df[['series_name', 'bucket', 'hist_sample_size']].rename(
            columns={'hist_sample_size': 'hist_sample_size_recent'}
        ),
        on=['series_name', 'bucket'],
        how='outer',
    )
    merged_df = pd.merge(
        merged_df,
        hist_all_df[['series_name', 'bucket', 'hist_sample_size']].rename(
            columns={'hist_sample_size': 'hist_sample_size_all'}
        ),
        on=['series_name', 'bucket'],
        how='outer',
    )
    if merged_df.empty:
        return

    merged_df['sample_size'] = merged_df['sample_size'].fillna(0).astype(int)
    merged_df['hist_sample_size_recent'] = merged_df['hist_sample_size_recent'].fillna(0).astype(int)
    merged_df['hist_sample_size_all'] = merged_df['hist_sample_size_all'].fillna(0).astype(int)
    merged_df['bucket'] = pd.Categorical(
        merged_df['bucket'],
        categories=['workday_normal', 'weekend', 'holiday', 'post_holiday_workday_1_3', 'makeup_workday'],
        ordered=True,
    )
    merged_df = merged_df.sort_values(['series_name', 'bucket']).reset_index(drop=True)

    series_names = list(merged_df['series_name'].dropna().unique())
    if not series_names:
        return

    fig, axes = plt.subplots(len(series_names), 1, figsize=(16, max(5 * len(series_names), 6)), sharex=False)
    if len(series_names) == 1:
        axes = [axes]

    for row_idx, series_name in enumerate(series_names):
        ax = axes[row_idx]
        series_df = merged_df.loc[merged_df['series_name'] == series_name].copy()
        if series_df.empty:
            continue

        labels = [BUCKET_DISPLAY_NAMES.get(str(item), str(item)) for item in series_df['bucket']]
        x = np.arange(len(series_df), dtype=float)
        width = 0.34

        bar_recent_bt = ax.bar(
            x - width / 2,
            series_df['sample_size'],
            width=width,
            label='回测样本数',
            color='#4472c4',
            alpha=0.85,
        )
        bar_recent_hist = ax.bar(
            x + width / 2,
            series_df['hist_sample_size_recent'],
            width=width,
            label='最近历史样本数',
            color='#70ad47',
            alpha=0.85,
        )

        ax2 = ax.twinx()
        line_all_hist, = ax2.plot(
            x,
            series_df['hist_sample_size_all'],
            label='全历史样本数',
            color='#ed7d31',
            linewidth=2.2,
            marker='o',
        )

        ax.set_title(f'{_series_display_name(series_name)} - 样本口径对照')
        ax.set_xlabel('说明：蓝色=最近回测样本，绿色=最近历史样本，橙线=全历史样本')
        ax.set_ylabel('样本量（最近窗口）')
        ax2.set_ylabel('样本量（全历史）')
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=25)
        ax.grid(True, axis='y', alpha=0.25)

        handles = [bar_recent_bt, bar_recent_hist, line_all_hist]
        labels_legend = [h.get_label() for h in handles]
        ax.legend(handles, labels_legend, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_stage_transition(stage_df, output_path):
    _configure_chinese_font()
    if stage_df is None or stage_df.empty:
        return

    df = stage_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    series_names = list(df['series_name'].dropna().unique())
    if not series_names:
        return

    fig, axes = plt.subplots(len(series_names), 1, figsize=(16, max(5 * len(series_names), 6)), sharex=False)
    if len(series_names) == 1:
        axes = [axes]

    stage_styles = [
        ('chronos_pred', 'Chronos', '#7f7f7f', 1.6),
        ('direct_pred', 'Direct', '#1f77b4', 1.6),
        ('fused_pred', '融合后', '#2ca02c', 2.0),
        ('residual_pred', 'Residual后', '#ff7f0e', 2.0),
        ('final_pred', 'Lead-wise后(最终)', '#d62728', 2.3),
    ]

    for row_idx, series_name in enumerate(series_names):
        ax = axes[row_idx]
        series_df = df.loc[df['series_name'] == series_name].sort_values('date').copy()
        if series_df.empty:
            continue

        ax.plot(series_df['date'], series_df['actual'], label='实际值', color='black', linewidth=2.4, marker='o', markersize=3)
        for col, label, color, width in stage_styles:
            if col not in series_df.columns:
                continue
            ax.plot(series_df['date'], series_df[col], label=label, color=color, linewidth=width, alpha=0.9)

        ax.set_title(f'{series_name} - 分阶段预测变化（最近回测窗口）')
        ax.set_ylabel('数值')
        ax.grid(True, alpha=0.25)
        ax.legend(loc='upper left', ncol=3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
