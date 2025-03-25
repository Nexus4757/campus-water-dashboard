
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import base64

# NOAA precipitation data
noaa_df = pd.read_csv("noaa_monthly_rainfall_datatest.pdf.csv", skiprows=2)
noaa_df = noaa_df.rename(columns={noaa_df.columns[0]: 'date', noaa_df.columns[7]: 'precipitation'})
noaa_df['date'] = pd.to_datetime(noaa_df['date'], errors='coerce')
noaa_df = noaa_df.dropna(subset=['date'])

# Water sensor data
df = pd.read_csv("bq-results-20250225-063801-1740465499494.csv")
df['record_time'] = pd.to_datetime(df['record_time'])
df['month'] = df['record_time'].dt.to_period('M')
df['date'] = df['record_time'].dt.date
numeric_columns = [col for col in df.select_dtypes(include=['number']).columns if col != 'sensor_Id']
agg_df = df.groupby(['date', 'sensor_Id'])[numeric_columns].mean().reset_index()
unique_dates = sorted(agg_df['date'].unique())

# Logo
image_path = "umd_loog.webp"
encoded_image = base64.b64encode(open(image_path, 'rb').read()).decode('ascii')

# App setup
app = dash.Dash(__name__)
app.title = "UMD Campus Water Dashboard"

app.layout = html.Div([
    html.Div([
        html.Img(src='data:image/png;base64,{}'.format(encoded_image), style={'height': '80px'}),
        html.H1("UMD Campus Water Quantity & Quality Dashboard", style={
            'textAlign': 'center', 'color': '#005A9C', 'fontWeight': 'bold'
        })
    ], style={'textAlign': 'center', 'marginBottom': '20px'}),

    dcc.Graph(id="precipitation-time-series"),

    html.Div([
        html.Label("Select Variable:"),
        dcc.Dropdown(
            id="variable-dropdown",
            options=[{"label": var.title(), "value": var} for var in numeric_columns],
            value="depth",
            clearable=False,
            style={'width': '300px', 'margin': '0 auto'}
        )
    ], style={'textAlign': 'center', 'marginBottom': '20px'}),

    dcc.Graph(id="time-series-plot"),

    html.Div([
        html.Label("Select Time Range (for time series):"),
        dcc.RangeSlider(
            id="date-slider",
            min=0,
            max=len(unique_dates) - 1,
            value=[0, len(unique_dates) - 1],
            marks={i: str(date) for i, date in enumerate(unique_dates) if i % max(1, len(unique_dates)//10) == 0},
            tooltip={"placement": "bottom", "always_visible": True}
        ),
        html.Button("▶️ Play", id="play-button", n_clicks=0, style={'marginTop': '10px'}),
        dcc.Interval(id="interval", interval=1000, n_intervals=0, disabled=True)
    ], style={'padding': '20px'}),

    html.Div([
        html.Label("Select Month (for gauges only):"),
        dcc.Dropdown(
            id="month-dropdown",
            options=[{'label': str(m), 'value': str(m)} for m in df['month'].unique()],
            value=str(df['month'].max()),
            clearable=False,
            style={'width': '40%', 'margin': 'auto'}
        )
    ], style={'marginBottom': '30px'}),

    html.Div([
        html.Div([dcc.Graph(id="gauge-mean")], style={'flex': '1'}),
        html.Div([dcc.Graph(id="gauge-max")], style={'flex': '1'}),
        html.Div([dcc.Graph(id="gauge-min")], style={'flex': '1'})
    ], style={'display': 'flex', 'justifyContent': 'space-around', 'padding': '20px'})
], style={'maxWidth': '1200px', 'margin': 'auto'})


@app.callback(
    Output("time-series-plot", "figure"),
    [Input("variable-dropdown", "value"),
     Input("date-slider", "value")]
)
def update_time_series(selected_var, date_range):
    start, end = date_range
    start_date = unique_dates[start]
    end_date = unique_dates[end]
    mask = (agg_df['date'] >= start_date) & (agg_df['date'] <= end_date)
    df_filtered = agg_df[mask]

    fig = go.Figure()
    for sid in df_filtered['sensor_Id'].unique():
        sensor_df = df_filtered[df_filtered['sensor_Id'] == sid]
        fig.add_trace(go.Scatter(x=sensor_df['date'], y=sensor_df[selected_var], name=f"Sensor {sid}", line=dict(width=2)))

    fig.update_layout(
        title=f"{selected_var.title()} Trends Across Sensors",
        xaxis_title="Date", yaxis_title=selected_var.title(),
        transition=dict(duration=500, easing="cubic-in-out"),
        template="plotly_white"
    )
    return fig


@app.callback(
    Output("gauge-mean", "figure"),
    Output("gauge-max", "figure"),
    Output("gauge-min", "figure"),
    Input("variable-dropdown", "value"),
    Input("month-dropdown", "value")
)
def update_gauges(selected_variable, selected_month):
    selected_month = pd.Period(selected_month)
    filtered = df[df['month'] == selected_month]

    if selected_variable not in filtered.columns or filtered[selected_variable].dropna().empty:
        return [go.Figure()] * 3

    mean_val = filtered[selected_variable].mean()
    max_val = filtered[selected_variable].max()
    min_val = filtered[selected_variable].min()

    def gauge(title, val, vmin, vmax):
        return go.Figure(go.Indicator(
            mode="gauge+number",
            value=val,
            title={'text': title},
            gauge={'axis': {'range': [vmin, vmax]}, 'bar': {'color': "#1f77b4"}}
        ))

    return (
        gauge(f"Mean {selected_variable.title()} in {selected_month}", mean_val, min_val, max_val),
        gauge(f"Max {selected_variable.title()} in {selected_month}", max_val, min_val, max_val),
        gauge(f"Min {selected_variable.title()} in {selected_month}", min_val, min_val, max_val)
    )


@app.callback(
    Output("precipitation-time-series", "figure"),
    Input("variable-dropdown", "value")
)
def update_precip(_):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=noaa_df['date'], y=noaa_df['precipitation'],
        mode='lines+markers', name="Precipitation", line=dict(color='blue')
    ))
    fig.update_layout(
        title="🌧️ Daily Precipitation Over Time",
        xaxis_title="Date", yaxis_title="Precipitation (mm)",
        template="plotly_white", transition=dict(duration=500)
    )
    return fig

@app.callback(
    Output("interval", "disabled"),
    Output("play-button", "children"),
    Input("play-button", "n_clicks"),
    State("interval", "disabled")
)
def toggle_play(n, is_disabled):
    if n % 2 == 1:
        return False, "⏸️ Pause"
    else:
        return True, "▶️ Play"

@app.callback(
    Output("date-slider", "value"),
    Input("interval", "n_intervals"),
    State("date-slider", "value")
)
def auto_slide(n, current_range):
    start, end = current_range
    if end + 1 < len(unique_dates):
        return [start + 1, end + 1]
    else:
        return [0, 10]

if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port=10000)