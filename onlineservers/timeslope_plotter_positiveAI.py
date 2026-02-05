import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

folder_path = "./outputs"
days_to_check = 90

today = datetime.now()
expected_dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, days_to_check + 1)]

try:
    files = os.listdir(folder_path)
except FileNotFoundError:
    print(f"Error: Folder '{folder_path}' does not exist.")
    sys.exit(1)

existing_dates = []
for file in files:
    if file.startswith("output_") and file.endswith(".xlsx"):
        try:
            date_part = file.split("_")[1].split(".")[0]
            datetime.strptime(date_part, "%Y-%m-%d")
            existing_dates.append(date_part)
        except ValueError:
            print(f"Error: Invalid date format in file '{file}'.")
            sys.exit(1)

missing_dates = [date for date in expected_dates if date not in existing_dates]

if missing_dates:
    print("Warning: Missing files for the following dates (skipping them):")
    for missing_date in missing_dates:
        print(missing_date)

directory = os.path.join(os.getcwd(), 'outputs')
excel_files = [f for f in os.listdir(directory) if f.startswith('output_') and f.endswith('.xlsx')]

data = {'Date': [], 'Average Slope': []}

for file in sorted(excel_files):
    date_str = file.split('_')[1].split('.xlsx')[0]
    df = pd.read_excel(os.path.join(directory, file))
    q1 = df['Slope'].quantile(0.05)
    q3 = df['Slope'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df_filtered = df[(df['Slope'] >= lower_bound) & (df['Slope'] <= upper_bound)]
    avg_slope = df_filtered['Slope'].mean()
    if not pd.isna(avg_slope):
        data['Date'].append(date_str)
        data['Average Slope'].append(avg_slope)

df_results = pd.DataFrame(data)
df_results['Date'] = pd.to_datetime(df_results['Date'])
df_results = df_results.sort_values('Date')
df_results['Previous Slope'] = df_results['Average Slope'].shift(1)

def custom_pct_change(current, previous):
    if previous == 0 or pd.isna(previous):
        return 0
    return (abs(current - previous) / abs(previous)) * 100

df_results['Slope Change (%)'] = df_results.apply(
    lambda row: custom_pct_change(row['Average Slope'], row['Previous Slope']), axis=1)

df_results['High Change (Window 3)'] = df_results['Slope Change (%)'] > (
        df_results['Slope Change (%)'].rolling(window=3).mean() +
        df_results['Slope Change (%)'].rolling(window=3).std()
)
df_results['High Change (Window 5)'] = df_results['Slope Change (%)'] > (
        df_results['Slope Change (%)'].rolling(window=5).mean() +
        df_results['Slope Change (%)'].rolling(window=5).std()
)

last_green_slope = None
last_green_date = None
pending_yellow_check = False
last_red_circle_date = None
yellow_circle_data = []

for i in range(len(df_results)):
    current_date = df_results['Date'].iloc[i]
    current_slope = df_results['Average Slope'].iloc[i]
    if pending_yellow_check and current_slope > last_green_slope:
        future_slope_values = df_results['Average Slope'].iloc[i + 1:i + 4]
        max_slope_value = future_slope_values.max() if not future_slope_values.empty else None
        days_since_last_red = (
                current_date - pd.to_datetime(last_red_circle_date)).days if last_red_circle_date else None
        intercepted = any(
            pd.to_datetime(yc[0]) < current_date and pd.to_datetime(yc[0]) > pd.to_datetime(last_red_circle_date) for yc
            in yellow_circle_data) if last_red_circle_date else False
        probability_above_10 = 1 if max_slope_value and max_slope_value > 10 else 0
        yellow_circle_data.append((current_date.strftime('%Y-%m-%d'), max_slope_value, last_red_circle_date,
                                   days_since_last_red, current_slope, intercepted, probability_above_10))
        pending_yellow_check = False
    if df_results['High Change (Window 3)'].iloc[i]:
        last_green_slope = current_slope
        last_green_date = current_date
        pending_yellow_check = True
    if df_results['High Change (Window 5)'].iloc[i]:
        last_red_circle_date = current_date.strftime('%Y-%m-%d')

yellow_circle_df = pd.DataFrame(yellow_circle_data, columns=['Yellow Circle Dates', 'Max Slope Value After 3 Days',
                                                             'Previous Red Circle Date', 'Days Since Last Red Circle',
                                                             'Current Yellow Circle Slope',
                                                             'Intercepted By Another Yellow Circle',
                                                             'Actually Above 10% Slope'])
X = yellow_circle_df[['Max Slope Value After 3 Days', 'Days Since Last Red Circle',
                      'Current Yellow Circle Slope', 'Intercepted By Another Yellow Circle']].copy()
y = yellow_circle_df['Actually Above 10% Slope']
X.fillna(X.median(), inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
model.fit(X_train, y_train)
yellow_circle_df['Probability Above 10%'] = model.predict_proba(X)[:, 1] * 100
yellow_circle_df=yellow_circle_df[['Yellow Circle Dates','Actually Above 10% Slope','Probability Above 10%']]
print(yellow_circle_df)


yellow_circle_df.to_csv("./csvs/yellow_circle_data.csv", index=False)
body = "Hello,\n\nPlease find attached the daily slope analysis report.\n\nBest regards,\nYour Automated System"

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import smtplib

def send_email_with_attachment(subject, body, recipient, attachment_path, sender_email, sender_password, smtp_server, smtp_port):
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    with open(attachment_path, 'rb') as attachment:
        mime_base = MIMEBase('application', 'octet-stream')
        mime_base.set_payload(attachment.read())
        encoders.encode_base64(mime_base)
        mime_base.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(attachment_path)}"')
        msg.attach(mime_base)
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()
    server.login(sender_email, sender_password)
    server.send_message(msg)
    server.quit()

send_email_with_attachment("Daily Slope Analysis", body, "mina.moussa@hotmail.com", "./csvs/yellow_circle_data.csv", "minamoussa903@gmail.com", "gxay qryy jsgg nnqn", "smtp.gmail.com", 587)

