import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_mock_data(filepath="data.csv", days=1100):
    np.random.seed(42)
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=days - 1)

    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Base volumes
    base_call = 1000
    base_ticket = 300

    call_volume = []
    tickets_received = []
    tickets_resolved = []

    for dt in dates:
        # Weekend seasonality
        if dt.weekday() >= 5: # Saturday or Sunday
            daily_call = int(np.random.normal(base_call * 0.3, base_call * 0.05))
            daily_ticket = int(np.random.normal(base_ticket * 0.3, base_ticket * 0.05))
        else:
            daily_call = int(np.random.normal(base_call, base_call * 0.1))
            daily_ticket = int(np.random.normal(base_ticket, base_ticket * 0.1))

        # Annual holiday (e.g., Spring Festival in February)
        if dt.month == 2 and 1 <= dt.day <= 7:
            daily_call = 0
            daily_ticket = 0

        # Ensure non-negative
        daily_call = max(0, daily_call)
        daily_ticket = max(0, daily_ticket)
        daily_resolved = max(0, daily_ticket - int(np.random.normal(10, 5))) # roughly resolving most

        call_volume.append(daily_call)
        tickets_received.append(daily_ticket)
        tickets_resolved.append(daily_resolved)

    df = pd.DataFrame({
        'date': dates,
        'call_volume': call_volume,
        'tickets_received': tickets_received,
        'tickets_resolved': tickets_resolved
    })

    # Format date as YYYY/M/D to match the user's example
    df['date'] = df['date'].dt.strftime('%Y/%-m/%-d')

    df.to_csv(filepath, index=False)
    print(f"Mock data generated at {filepath} with {len(df)} rows.")

if __name__ == "__main__":
    generate_mock_data()
