import os
import re
import pandas as pd
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# Email sender config
recipient_email = "mina.moussa@hotmail.com"
sender_email = "minamoussa903@gmail.com"
sender_password = "qhvi syra bbad gylu"

# Get yesterday's dat
today_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
filename_pattern = f"{today_date}_oversold.txt"
print(filename_pattern)
directory = r"./oversold_analysis/"
file_path = os.path.join(directory, filename_pattern)

# Regex to parse data lines
line_regex = re.compile(
    r'\s*(\d{4}-\d{2}-\d{2})\s+'
    r'(\d{4}-\d{2}-\d{2})\s+'
    r'([\d\.]+)\s+'
    r'([\d\.]+)\s+'
    r'([\d\.]+)\s+'
    r'(\S+)'
)


def parse_df_for_date(file_path, target_date):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    rows = []
    for line in lines:
        if target_date in line:
            match = line_regex.match(line.strip())
            if match:
                row = match.groups()
                rows.append(row)

    df = pd.DataFrame(rows, columns=['Previous Date', 'Current Date', 'Previous RSI', 'Current RSI', 'Drop', 'Symbol'])
    df['Drop'] = pd.to_numeric(df['Drop'], errors='coerce')
    #print(df)
    return df



def send_email_with_analysis(body, df=None):
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = "Oversold Analysis Alert"
    msg.attach(MIMEText(body, 'plain'))

    # Save DataFrame to CSV if applicable
    temp_csv = None
    if df is not None and not df.empty:
        temp_csv = os.path.join(directory, "filtered_output.csv")
        df.to_csv(temp_csv, index=False)
        with open(temp_csv, 'rb') as f:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(temp_csv)}')
        msg.attach(part)

    # Send the email
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        print("Email sent.")
    except Exception as e:
        print("Error sending email:", e)


def main():
    if not os.path.exists(file_path):
        print("File not found:", file_path)
        return

    df = parse_df_for_date(file_path, today_date)

    #if df.empty:
    #    print("BUY (no data for date)")
    #    send_email_with_analysis("BUY NOW")
    if (df['Drop'].max() <= 17) and (len(df) < 10):
        print("BUY (all drops = 16 and < 10 rows)")
        send_email_with_analysis("BUY NOW", df)
    else:
        print("DONT BUY (drop too high or too many rows)")
        send_email_with_analysis("DONT BUY")


if __name__ == "__main__":
    main()
