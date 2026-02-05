import os
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

import os
from datetime import datetime, timedelta
import sys


#modify this to go back in time
#today = datetime.strptime("2025-10-25", "%Y-%m-%d")
today = datetime.now()


# Email configuration
subject = "Daily Slope Analysis"
recipient = "mina.moussa@hotmail.com"
sender_email = "minamoussa903@gmail.com"
sender_password = "thjj eryc yzym dylb"
smtp_server = "smtp.gmail.com"
smtp_port = 587


# Set the folder path and the time range
folder_path = "./outputs"
days_to_check = 90  # Number of past days to check

# Get today's date

# Generate a list of all dates in the past `days_to_check` days, excluding today
expected_dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, days_to_check + 1)]

# Get the list of files in the folder
try:
    files = os.listdir(folder_path)
except FileNotFoundError:
    print(f"Error: Folder '{folder_path}' does not exist.")
    sys.exit(1)

# Extract dates from filenames
existing_dates = []
for file in files:
    if file.startswith("output_") and file.endswith(".xlsx"):
        try:
            date_part = file.split("_")[1].split(".")[0]
            # Ensure the date format is valid
            datetime.strptime(date_part, "%Y-%m-%d")
            existing_dates.append(date_part)
        except ValueError:
            print(f"Error: Invalid date format in file '{file}'.")
            sys.exit(1)

# Find missing dates
missing_dates = [date for date in expected_dates if date not in existing_dates]

# Check if any dates are missing
if missing_dates:
    print("Error: Missing files for the following dates:")
    for missing_date in missing_dates:
        print(missing_date)
    sys.exit(1)
else:
    print("All files are present for the past 90 days, excluding today.")


# Get all Excel files in the current directory
#directory = os.getcwd() + '/outputs/'
#excel_files = [f for f in os.listdir(directory) if f.startswith('output_') and f.endswith('.xlsx')]
# Get all Excel files in the current directory
# Get all Excel files in the current directory
directory = os.getcwd() + '/outputs/'
all_excel_files = [f for f in os.listdir(directory) if f.startswith('output_') and f.endswith('.xlsx')]

# Only keep files up to and including the 'today' date you set above
def get_date_from_filename(filename):
    try:
        date_str = filename.split('_')[1].split('.xlsx')[0]  # e.g. "2025-11-16"
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except Exception:
        return None

cutoff_date = today.date()

excel_files = [
    f for f in all_excel_files
    if (file_date := get_date_from_filename(f)) is not None and file_date <= cutoff_date
]

# Optional: print to confirm which ones are used
print("Excel files being processed (<= cutoff date):")
for f in sorted(excel_files):
    print(f)


# List to store big dip days
big_dip_days = []


# Helper function to filter out major outliers based on the IQR method
def filter_outliers(df, column):
    q1 = df[column].quantile(0.05)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


# Data storage
data = {
    'Date': [],
    'Average Slope': []
}

# Loop through all Excel files
for file in sorted(excel_files):
    print(file)
    # Extract date from filename (e.g., output_2024-09-01.xlsx)
    date_str = file.split('_')[1].split('.xlsx')[0]

    # Read the Excel file
    df = pd.read_excel('./outputs/' + file)

    # Filter out outliers from the 'Slope' column
    df_filtered = filter_outliers(df, 'Slope')

    # Calculate the average slope after filtering out outliers
    avg_slope = df_filtered['Slope'].mean()

    # Append to data dictionary if slope data is valid
    if not pd.isna(avg_slope):
        data['Date'].append(date_str)
        data['Average Slope'].append(avg_slope)

# Convert dictionary to DataFrame
df_results = pd.DataFrame(data)

# Convert 'Date' column to datetime for proper sorting and plotting
df_results['Date'] = pd.to_datetime(df_results['Date'])

# Sort by Date to ensure the timeline is correct
df_results = df_results.sort_values('Date')


# The rest of the code remains unchanged, handling slope calculations, plotting, and finding buy/sell days
# (The rest of your original code goes here, working on the cleaned `df_results` DataFrame)


# Custom percentage change function to handle negative values properly
def custom_pct_change(current, previous):
    if previous == 0 or pd.isna(previous):  # Handle division by zero or NaN values
        return 0
    return (abs(current - previous) / abs(previous)) * 100


# Shift the 'Average Slope' to get previous values
df_results['Previous Slope'] = df_results['Average Slope'].shift(1)

# Calculate percentage changes
df_results['Slope Change (%)'] = df_results.apply(
    lambda row: custom_pct_change(row['Average Slope'], row['Previous Slope']), axis=1)


# Function to check if the current slope is lower than the previous two slopes
def check_slope_below_previous_two(series):
    if len(series) < 3:
        return False
    return series.iloc[-1] < series.iloc[-2] and series.iloc[-1] < series.iloc[-3]


# Rolling calculations for both rolling windows 3 and 5
df_results['Slope Below Previous Two (Window 3)'] = df_results['Average Slope'].rolling(window=3).apply(
    check_slope_below_previous_two, raw=False)
df_results['Slope Below Previous Two (Window 5)'] = df_results['Average Slope'].rolling(window=5).apply(
    check_slope_below_previous_two, raw=False)

# Add a column to check if the percentage change is significantly high compared to the previous rolling windows
df_results['High Change (Window 3)'] = df_results['Slope Change (%)'] > df_results['Slope Change (%)'].rolling(
    window=3).mean() + df_results['Slope Change (%)'].rolling(window=3).std()
df_results['High Change (Window 5)'] = df_results['Slope Change (%)'] > df_results['Slope Change (%)'].rolling(
    window=5).mean() + df_results['Slope Change (%)'].rolling(window=5).std()

# Plot Average Timetostop
# plt.plot(df_results['Date'], df_results['Average Timetostop'], label='Average Timetostop', marker='o')

# Annotate slope points that meet the conditions for both rolling windows
for i in range(1, len(df_results)):
    if df_results['High Change (Window 3)'].iloc[i] and df_results['Slope Below Previous Two (Window 3)'].iloc[i]:
        big_dip_days.append((df_results['Date'].iloc[i] + timedelta(days=1)).strftime('%Y-%m-%d'))

    if df_results['High Change (Window 5)'].iloc[i] and df_results['Slope Below Previous Two (Window 5)'].iloc[i]:
        big_dip_days.append((df_results['Date'].iloc[i] + timedelta(days=1)).strftime('%Y-%m-%d'))


# Function to calculate the new slope after applying the percentage drop
def calculate_slope_after_drop(previous_slope, threshold_pct):
    # Calculate the amount to subtract based on the threshold percentage
    slope_drop = previous_slope * (threshold_pct / 100)
    return previous_slope - slope_drop


yesterday = (today).strftime('%Y-%m-%d')

for i in range(1, len(df_results)):
    avg_slope = df_results['Average Slope'].iloc[i]
    previous_slope = df_results['Previous Slope'].iloc[i]  # The previous slope value
    window_3_threshold = df_results['Slope Change (%)'].rolling(window=3).mean().iloc[i] + \
                         df_results['Slope Change (%)'].rolling(window=3).std().iloc[i]
    window_5_threshold = df_results['Slope Change (%)'].rolling(window=5).mean().iloc[i] + \
                         df_results['Slope Change (%)'].rolling(window=5).std().iloc[i]

    # Calculate the slope after the percentage drop for both Window 3 and Window 5
    new_slope_window_3 = calculate_slope_after_drop(previous_slope, window_3_threshold)
    new_slope_window_5 = calculate_slope_after_drop(previous_slope, window_5_threshold)

    # Check if the current date is yesterday
    if df_results['Date'].iloc[i].strftime('%Y-%m-%d') == yesterday:
        # Print the results only for yesterday's date
        print(
            f"Date: {df_results['Date'].iloc[i]}, Current Average Slope: {avg_slope:.2f}, Previous Slope: {previous_slope:.2f}")
        print(f"Threshold (Window 3): {window_3_threshold:.2f}% -> New Slope after Window 3: {new_slope_window_3:.2f}")
        print(f"Threshold (Window 5): {window_5_threshold:.2f}% -> New Slope after Window 5: {new_slope_window_5:.2f}")

import numpy as np

# Convert big_dip_days from string to datetime for comparison with the 'Date' column
big_dip_days_dt = pd.to_datetime(big_dip_days)

# Plotting Average Slope, New Slope after Window 3, and New Slope after Window 5
dates = df_results['Date']
avg_slopes = df_results['Average Slope']

# Initialize lists for new slopes after Window 3 and Window 5
new_slopes_window_3 = [np.nan]  # Add NaN at the start to match the length of 'dates'
new_slopes_window_5 = [np.nan]  # Add NaN at the start to match the length of 'dates'

# Handle missing data by filling NaN if there are no values
for i in range(1, len(df_results)):
    avg_slope = df_results['Average Slope'].iloc[i]
    previous_slope = df_results['Previous Slope'].iloc[i]  # The previous slope value
    if pd.isna(previous_slope):
        new_slopes_window_3.append(np.nan)
        new_slopes_window_5.append(np.nan)
        continue

    window_3_threshold = df_results['Slope Change (%)'].rolling(window=3).mean().iloc[i] + \
                         df_results['Slope Change (%)'].rolling(window=3).std().iloc[i]
    window_5_threshold = df_results['Slope Change (%)'].rolling(window=5).mean().iloc[i] + \
                         df_results['Slope Change (%)'].rolling(window=5).std().iloc[i]

    # Calculate the slope after the percentage drop for both Window 3 and Window 5
    new_slope_window_3 = calculate_slope_after_drop(previous_slope, window_3_threshold)
    new_slope_window_5 = calculate_slope_after_drop(previous_slope, window_5_threshold)

    new_slopes_window_3.append(new_slope_window_3)
    new_slopes_window_5.append(new_slope_window_5)


# Function to calculate the new slope after applying the percentage change
def calculate_new_slope(previous_slope, threshold_pct):
    return previous_slope * (1 + threshold_pct / 100)


for i in range(1, len(df_results)):
    avg_slope = df_results['Average Slope'].iloc[i]
    previous_slope = df_results['Previous Slope'].iloc[i]  # The previous slope value
    window_3_threshold = df_results['Slope Change (%)'].rolling(window=3).mean().iloc[i] + \
                         df_results['Slope Change (%)'].rolling(window=3).std().iloc[i]
    window_5_threshold = df_results['Slope Change (%)'].rolling(window=5).mean().iloc[i] + \
                         df_results['Slope Change (%)'].rolling(window=5).std().iloc[i]

    # Calculate the slope after the percentage drop for both Window 3 and Window 5
    new_slope_window_3 = calculate_slope_after_drop(previous_slope, window_3_threshold)
    new_slope_window_5 = calculate_slope_after_drop(previous_slope, window_5_threshold)

    # Check if the current date is yesterday
    if df_results['Date'].iloc[i].strftime('%Y-%m-%d') == yesterday:
        # Print the results only for yesterday's date
        print(
            f"Date: {df_results['Date'].iloc[i]}, Current Average Slope: {avg_slope:.2f}, Previous Slope: {previous_slope:.2f}")
        print(f"Threshold (Window 3): {window_3_threshold:.2f}% -> New Slope after Window 3: {new_slope_window_3:.2f}")
        print(f"Threshold (Window 5): {window_5_threshold:.2f}% -> New Slope after Window 5: {new_slope_window_5:.2f}")

# Add Sync column to the summary table
summary_data = {
    "Date": df_results['Date'],
    "Current Average Slope (Blue Line)": df_results['Average Slope'],
    "New Slope after Window 3 (Green Line)": new_slopes_window_3,
    "New Slope after Window 5 (Red Line)": new_slopes_window_5}

# Create the summary DataFrame
summary_df = pd.DataFrame(summary_data)

# Add a column to check if the green line is 10 higher than the blue line and also greater than 10
summary_df['Green Line Above 20'] = (
        (summary_df['New Slope after Window 3 (Green Line)'] > summary_df['Current Average Slope (Blue Line)'] + 10) &
        (summary_df['New Slope after Window 3 (Green Line)'] > 10)
)

summary_df['Red Line Below -40'] = summary_df['New Slope after Window 5 (Red Line)'] < -40

# Plotting the data
plt.figure(figsize=(12, 8))

# Plot each line (no changes to your existing lines)
plt.plot(summary_df['Date'], summary_df['Current Average Slope (Blue Line)'], label="Blue Line (Current Average Slope)",
         marker='o', color="blue")
#plt.plot(summary_df['Date'], summary_df['New Slope after Window 3 (Green Line)'],
#         label="Green Line (New Slope after Window 3)", marker='x', color="green")
#plt.plot(summary_df['Date'], summary_df['New Slope after Window 5 (Red Line)'],
#         label="Red Line (New Slope after Window 5)", marker='s', color="red")

# Initialize variables
green_dot_days = []
red_dot_days = []
yellow_circle_days = []

# Track the last valid green slope and date
last_green_slope = None
last_green_date = None
pending_yellow_check = False  # Flag to ensure yellow checks continue past removed green spots

# Loop through the DataFrame
for i in range(len(summary_df)):
    current_date = summary_df['Date'].iloc[i]
    current_slope = summary_df['Current Average Slope (Blue Line)'].iloc[i]

    # Identify green dots and remove consecutive occurrences
    if summary_df['Green Line Above 20'].iloc[i]:
        # Only keep green dots that are not consecutive
        if last_green_date is None or (current_date - last_green_date).days > 1:
            green_dot_days.append(current_date.strftime('%Y-%m-%d'))
            last_green_slope = current_slope  # Update last valid green slope
            last_green_date = current_date  # Update last valid green date
            pending_yellow_check = True  # Ensure yellow checks start for this green dot

            # Plot green circle
            plt.scatter(
                current_date,
                current_slope,
                color='green',
                edgecolor='black',
                s=200,
                label='Green Line Above 20' if len(green_dot_days) == 1 else ""
            )
        else:
            # If the green dot is removed, ensure it can still be considered for yellow
            if pending_yellow_check and current_slope > last_green_slope:
                yellow_circle_days.append(current_date.strftime('%Y-%m-%d'))
                pending_yellow_check = False  # Reset the flag after plotting yellow

                # Plot yellow circle
                plt.scatter(
                    current_date,
                    current_slope,
                    color='yellow',
                    edgecolor='black',
                    s=200,
                    label='Yellow Circle' if len(yellow_circle_days) == 1 else ""
                )

    # Check for yellow circles
    elif pending_yellow_check and current_slope > last_green_slope:
        yellow_circle_days.append(current_date.strftime('%Y-%m-%d'))
        pending_yellow_check = False  # Reset the flag after plotting yellow

        # Plot yellow circle
        plt.scatter(
            current_date,
            current_slope,
            color='yellow',
            edgecolor='black',
            s=200,
            label='Yellow Circle' if len(yellow_circle_days) == 1 else ""
        )

# Plot red circles
for i, is_below in enumerate(summary_df['Red Line Below -40']):
    if is_below:
        red_dot_days.append(summary_df['Date'].iloc[i].strftime('%Y-%m-%d'))
        plt.scatter(
            summary_df['Date'].iloc[i],
            summary_df['Current Average Slope (Blue Line)'].iloc[i],
            color='red',
            edgecolor='black',
            s=200,
            label='Red Line Below -40' if len(red_dot_days) == 1 else ""
        )

# Print identified days
# Convert mygreendots to a list of dates for processing
date_list = [pd.to_datetime(ts).date() for ts in yellow_circle_days]

# Filter dates to ensure at least 4-day intervals
filtered_dates = []
previous_date = None

for date in date_list:
    if not previous_date or (date - previous_date >= timedelta(days=4)):
        filtered_dates.append(date)
    previous_date = date

# Prepare the final list of the last 4 filtered dates in 'YYYY-MM-DD' format
mylist = [date.strftime('%Y-%m-%d') for date in filtered_dates[-4:]]
print(mylist)


# Finalize plot
plt.xlabel("Date")
plt.ylabel("Slope")
plt.title("Green Buy, Red Sell, Yellow First Blue Above Green")
plt.legend()
plt.grid(True)

# Rotate x-axis labels for better visibility
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot as a JPG
file_path = os.path.join("plots", f"plot_{datetime.today().strftime('%Y-%m-%d')}.jpg")
plt.savefig(file_path, format='jpg', dpi=300)

# Show the plot
# Print identified days
print("Buy days (green):", green_dot_days)
print("Sell days (red):", red_dot_days)
print("Yellow circle days:", yellow_circle_days)
from datetime import datetime, timedelta


# Get yesterday's date as a string in the same format
yesterday = (today - timedelta(days=1)).strftime("%Y-%m-%d")

# Find index of yesterday's date if it exists
if yesterday in yellow_circle_days:
    yesterday_index = yellow_circle_days.index(yesterday)
    # Take 3 entries before yesterday's date
    selected_entries = yellow_circle_days[max(0, yesterday_index - 3):yesterday_index]
else:
    # If yesterday's date is not in the list, just take the last 3 entries
    selected_entries = yellow_circle_days[-3:]

# Write the selected entries to file
with open("yellowcircle.txt", "w") as file:
    file.write("\n".join(selected_entries))

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import smtplib

# Update the email body to include the green_dot_days
body = f"""\
Hello,

Please find attached the daily slope analysis plot.

Here are the identified yellow dot days:
{', '.join(mylist)}

Best regards,
Your Automated System
"""


# Function to send the email with the JPG attachment
def send_email_with_attachment(subject, body, recipient, attachment_path, sender_email, sender_password, smtp_server,
                               smtp_port):
    # Create the email
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient
    msg['Subject'] = subject

    # Attach the email body
    msg.attach(MIMEText(body, 'plain'))

    # Attach the JPG file
    with open(attachment_path, 'rb') as attachment:
        mime_base = MIMEBase('application', 'octet-stream')
        mime_base.set_payload(attachment.read())
        encoders.encode_base64(mime_base)
        mime_base.add_header(
            'Content-Disposition',
            f'attachment; filename="{os.path.basename(attachment_path)}"'
        )
        msg.attach(mime_base)

    # Send the email
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")
    finally:
        server.quit()



# Call the function to send the email
attachment_path = file_path
send_email_with_attachment(subject, body, recipient, attachment_path, sender_email, sender_password, smtp_server,smtp_port)
