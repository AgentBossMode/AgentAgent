stock_tools = r'''
import os
import threading
import signal
import time
import sys
from threading import Event
from intriniorealtime.equities_client import IntrinioRealtimeEquitiesClient
from suprsend import SuprSend
from composio import Composio
from composio_langchain import LangchainProvider
from langchain_core.tools import tool

composio = Composio(provider=LangchainProvider())
Investment_Database_CREATE = composio.tools.get(user_id=os.environ("USER_ID"), tools=["NOTION_INSERT_ROW_DATABASE"])
Investment_Database_READ = composio.tools.get(user_id=os.environ("USER_ID"), tools=["NOTION_QUERY_DATABASE"])

@tool
def Stock_Market_Data_API_READ():
    """To get real-time or near real-time stock prices and other market data."""
    trade_count = 0
    ask_count = 0
    bid_count = 0
    backlog_count = 0

    def on_quote(quote, backlog):
        nonlocal ask_count, bid_count, backlog_count
        backlog_count = backlog
        if hasattr(quote, 'type'):
            if quote.type == "ask":
                ask_count += 1
            else:
                bid_count += 1

    def on_trade(trade, backlog):
        nonlocal trade_count, backlog_count
        backlog_count = backlog
        trade_count += 1

    configuration = {
        'api_key': 'API_KEY_HERE',
        'provider': 'IEX',  # REALTIME (IEX), or IEX, or DELAYED_SIP, or NASDAQ_BASIC, or CBOE_ONE
        'on_quote': on_quote,
        'on_trade': on_trade
    }

    client = IntrinioRealtimeEquitiesClient(configuration)

    stop_event = Event()

    def on_kill_process(sig, frame):
        print("Stopping")
        stop_event.set()
        client.disconnect()
        sys.exit(0)

    signal.signal(signal.SIGINT, on_kill_process)

    client.join(['AAPL', 'GE', 'MSFT'])
    client.connect()

    time.sleep(120)

    print("Stopping")
    stop_event.set()
    client.disconnect()
    sys.exit(0)

@tool
def Notification_Service_CREATE():
    """To send notifications to users via various channels."""
    # Initialize SuprSend client
    client = SuprSend(workspace_key="YOUR_WORKSPACE_KEY", workspace_secret="YOUR_WORKSPACE_SECRET")

    # Send email notification
    response = client.send_email(
        to="recipient@example.com",
        subject="Test Email",
        body="Hello from SuprSend Python SDK!"
    )

    print(response)'''