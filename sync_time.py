import ntplib
import os
import time
from datetime import datetime

def sync_system_time():
    try:
        ntp_client = ntplib.NTPClient()
        response = ntp_client.request('pool.ntp.org')
        os.system(f'date -s "@{response.tx_time}"')
        return True
    except Exception as e:
        print(f"Failed to sync time: {e}")
        return False

if __name__ == "__main__":
    sync_system_time() 