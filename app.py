import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from prophet import Prophet

df = pd.read_csv('data/merged_data.csv')


st.title('Fitness & Activity Trend Analyzer')
st.markdown('Analyzing Fitbit data from 30 users over 31 days')
st.subheader('Key Metrics')
col1, col2, col3, col4 = st.columns(4)

col1.metric('Avg Daily Steps', f"{int(df['TotalSteps'].mean()):,}")
col2.metric('Avg Calories', f"{int(df['Calories'].mean()):,}")
col3.metric('Total Users', df['Id'].nunique())
col4.metric('Avg Sleep (mins)', f"{int(df['TotalMinutesAsleep'].mean()):,}")

st.write('---')

st.subheader('Steps Distribution')
fig, ax = plt.subplots()
ax.hist(df['TotalSteps'], bins=20, color='steelblue', edgecolor='white')
ax.set_xlabel('Total Steps')
ax.set_ylabel('Frequency')
st.pyplot(fig)

st.subheader('Calories Burned vs Steps Taken')
fig, ax = plt.subplots()
ax.scatter(df['TotalSteps'],df['Calories'],alpha=0.4,color='coral')
ax.set_xlabel('TotalSteps')
ax.set_ylabel('Calories')
st.pyplot(fig)

st.subheader('Sleep vs Active Minutes')
fig, ax = plt.subplots()
ax.scatter(df['VeryActiveMinutes'], df['TotalMinutesAsleep'], alpha=0.4, color='mediumpurple')
ax.set_xlabel('Very Active Minutes')
ax.set_ylabel('Sleep (mins)')
st.pyplot(fig)

st.subheader('User Segmentation (K-Means Clustering)')
user_summary = df.groupby('Id').agg(
    AvgSteps=('TotalSteps', 'mean'),
    AvgCalories=('Calories', 'mean'),
    AvgSleep=('TotalMinutesAsleep', 'mean'),
    AvgVeryActiveMinutes=('VeryActiveMinutes', 'mean'),
    AvgSedentaryMinutes=('SedentaryMinutes', 'mean')
).reset_index()
user_summary = user_summary.fillna(0)

features = ['AvgSteps', 'AvgCalories', 'AvgSleep', 'AvgVeryActiveMinutes', 'AvgSedentaryMinutes']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(user_summary[features])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
user_summary['Cluster'] = kmeans.fit_predict(X_scaled)

cluster_avg_steps = user_summary.groupby('Cluster')['AvgSteps'].mean().sort_values(ascending=False)

label_map = {
    cluster_avg_steps.index[0]: 'Active',
    cluster_avg_steps.index[1]: 'Moderate',
    cluster_avg_steps.index[2]: 'Sedentary'
}

user_summary['Segment'] = user_summary['Cluster'].map(label_map)

st.subheader('Steps vs Calories by Segment')

color_map = {'Active': '#2ecc71', 'Moderate': '#3498db', 'Sedentary': '#e74c3c'}

fig, ax = plt.subplots()
for segment, grp in user_summary.groupby('Segment'):
    ax.scatter(grp['AvgSteps'], grp['AvgCalories'],
               label=segment, color=color_map[segment], s=80, alpha=0.8)

ax.set_xlabel('Avg Daily Steps')
ax.set_ylabel('Avg Calories Burned')
ax.set_title('User Clusters: Steps vs Calories')
ax.legend()
st.pyplot(fig)

st.subheader('Segment Summary')
summary_table = user_summary.groupby('Segment')[features].mean().round(1).reset_index()
st.dataframe(summary_table, use_container_width=True)

st.write('---')
st.subheader('30-Day Steps Forecast (Prophet)')

daily_steps = df.groupby('ActivityDate')['TotalSteps'].mean().reset_index()
daily_steps.columns = ['ds', 'y']
daily_steps['ds'] = pd.to_datetime(daily_steps['ds'])

model = Prophet(
    yearly_seasonality=False,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.1
)
model.fit(daily_steps)

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(daily_steps['ds'], daily_steps['y'], color='steelblue', label='Actual Steps', linewidth=2)

forecast_only = forecast[forecast['ds'] > daily_steps['ds'].max()]
ax.plot(forecast_only['ds'], forecast_only['yhat'], color='coral', linestyle='--', label='Forecast', linewidth=2)

ax.fill_between(forecast_only['ds'], forecast_only['yhat_lower'], forecast_only['yhat_upper'], alpha=0.2, color='coral', label='Confidence Interval')

ax.set_xlabel('Date')
ax.set_ylabel('Average Steps')
ax.set_title('Actual vs Forecasted Daily Steps')
ax.legend()
st.pyplot(fig)

st.subheader('Forecast Values (Next 30 Days)')
forecast_display = forecast_only[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
forecast_display.columns = ['Date', 'Predicted Steps', 'Lower Bound', 'Upper Bound']
forecast_display['Date'] = forecast_display['Date'].dt.strftime('%Y-%m-%d')
forecast_display['Predicted Steps'] = forecast_display['Predicted Steps'].round(0).astype(int)
forecast_display['Lower Bound'] = forecast_display['Lower Bound'].round(0).astype(int)
forecast_display['Upper Bound'] = forecast_display['Upper Bound'].round(0).astype(int)
forecast_display = forecast_display.reset_index(drop=True)
st.dataframe(forecast_display, use_container_width=True)