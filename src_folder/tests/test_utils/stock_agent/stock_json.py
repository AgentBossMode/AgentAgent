stock_json = r'''
{
    "graphstate": {
      "type": "TypedDict",
      "fields": [
        {
          "name": "user_input",
          "type": "str",
          "description": "The user's initial request or input."
        },
        {
          "name": "request_type",
          "type": "str",
          "description": "The type of request identified from the user's input (e.g., 'add_investment', 'get_portfolio_value', 'set_alert', 'summarize_performance')."
        },
        {
          "name": "investment_details",
          "type": "dict",
          "description": "Details of the investment to be added, including stock symbol, shares, and price."
        },
        {
          "name": "portfolio_value",
          "type": "float",
          "description": "The current value of the user's investment portfolio."
        },
        {
          "name": "portfolio_performance",
          "type": "str",
          "description": "The performance of the user's investment portfolio over a specified period."
        },
        {
          "name": "alert_details",
          "type": "dict",
          "description": "Details of the alert to be set, including stock symbol, condition, and threshold."
        },
        {
          "name": "response",
          "type": "str",
          "description": "A natural language response to the user's request."
        }
      ]
    },
    "justification": "This workflow utilizes a standard stateful graph architecture. The 'route_request' node acts as a central dispatcher, analyzing the user's input and directing it to the appropriate specialized node. Each subsequent node (add_investment, get_portfolio_value, set_alert, summarize_performance) is responsible for handling a specific type of request, leveraging external tools as needed. This design is straightforward and efficient for the defined purposes and capabilities, as there's no explicit need for complex planning, hierarchical supervision, multi-agent collaboration, or iterative self-correction based on the given requirements and dry runs. The workflow directly addresses the user's intent and provides a clear path for execution.",
    "nodes": [
      {
        "id": "__START__",
        "llm_actions": [],
        "input_schema": [],
        "output_schema": [],
        "description": "Entry point of the workflow.",
        "function_name": "start_node"
      },
      {
        "id": "route_request",
        "llm_actions": [
          "analyze",
          "route"
        ],
        "input_schema": [
          {
            "name": "user_input",
            "type": "str",
            "description": "The user's initial request."
          }
        ],
        "output_schema": [
          {
            "name": "request_type",
            "type": "str",
            "description": "The identified type of request (e.g., 'add_investment', 'get_portfolio_value', 'set_alert', 'summarize_performance')."
          },
          {
            "name": "investment_details",
            "type": "dict",
            "description": "Extracted details for adding an investment, if applicable."
          },
          {
            "name": "alert_details",
            "type": "dict",
            "description": "Extracted details for setting an alert, if applicable."
          }
        ],
        "description": "Analyzes the user's input to determine the type of request and routes it to the appropriate handling node.",
        "function_name": "route_request_node"
      },
      {
        "id": "add_investment",
        "llm_actions": [
          "tool_call",
          "generate"
        ],
        "input_schema": [
          {
            "name": "investment_details",
            "type": "dict",
            "description": "Details of the investment to be added."
          }
        ],
        "output_schema": [
          {
            "name": "response",
            "type": "str",
            "description": "A confirmation message that the investment was successfully recorded."
          }
        ],
        "description": "Records a new investment in the user's portfolio using the Investment Database.",
        "function_name": "add_investment_node"
      },
      {
        "id": "get_portfolio_value",
        "llm_actions": [
          "tool_call",
          "aggregate",
          "analyze",
          "generate"
        ],
        "input_schema": [],
        "output_schema": [
          {
            "name": "portfolio_value",
            "type": "float",
            "description": "The calculated total value of the portfolio."
          },
          {
            "name": "response",
            "type": "str",
            "description": "A summary of the portfolio's current value and performance."
          }
        ],
        "description": "Retrieves investment records and current stock prices to calculate and report the total portfolio value.",
        "function_name": "get_portfolio_value_node"
      },
      {
        "id": "set_alert",
        "llm_actions": [
          "tool_call",
          "generate"
        ],
        "input_schema": [
          {
            "name": "alert_details",
            "type": "dict",
            "description": "Details of the alert to be set."
          }
        ],
        "output_schema": [
          {
            "name": "response",
            "type": "str",
            "description": "A confirmation message that the alert was successfully set."
          }
        ],
        "description": "Sets up a notification alert based on predefined conditions for stock prices.",
        "function_name": "set_alert_node"
      },
      {
        "id": "summarize_performance",
        "llm_actions": [
          "tool_call",
          "aggregate",
          "analyze",
          "generate"
        ],
        "input_schema": [],
        "output_schema": [
          {
            "name": "portfolio_performance",
            "type": "str",
            "description": "A summary of the portfolio's performance over the specified period."
          },
          {
            "name": "response",
            "type": "str",
            "description": "A natural language summary of the portfolio's performance."
          }
        ],
        "description": "Generates a summary of the portfolio's performance over a specified period.",
        "function_name": "summarize_performance_node"
      },
      {
        "id": "__END__",
        "llm_actions": [],
        "input_schema": [],
        "output_schema": [],
        "description": "End point of the workflow.",
        "function_name": "end_node"
      }
    ],
    "edges": [
      {
        "source": "__START__",
        "target": "route_request",
        "routing_conditions": "",
        "conditional": false
      },
      {
        "source": "route_request",
        "target": "add_investment",
        "routing_conditions": "request_type == 'add_investment'",
        "conditional": true
      },
      {
        "source": "route_request",
        "target": "get_portfolio_value",
        "routing_conditions": "request_type == 'get_portfolio_value'",
        "conditional": true
      },
      {
        "source": "route_request",
        "target": "set_alert",
        "routing_conditions": "request_type == 'set_alert'",
        "conditional": true
      },
      {
        "source": "route_request",
        "target": "summarize_performance",
        "routing_conditions": "request_type == 'summarize_performance'",
        "conditional": true
      },
      {
        "source": "add_investment",
        "target": "__END__",
        "routing_conditions": "",
        "conditional": false
      },
      {
        "source": "get_portfolio_value",
        "target": "__END__",
        "routing_conditions": "",
        "conditional": false
      },
      {
        "source": "set_alert",
        "target": "__END__",
        "routing_conditions": "",
        "conditional": false
      },
      {
        "source": "summarize_performance",
        "target": "__END__",
        "routing_conditions": "",
        "conditional": false
      }
    ],
    "tools": [
      {
        "name": "Investment_Database_CREATE",
        "description": "To store user's investment records, including stock purchases.",
        "is_composio_tool": true,
        "composio_toolkit_slug": "Notion",
        "composio_tool_slug": "NOTION_INSERT_ROW_DATABASE",
        "node_ids": [
          "add_investment"
        ]
      },
      {
        "name": "Investment_Database_READ",
        "description": "To retrieve user's investment records.",
        "is_composio_tool": true,
        "composio_toolkit_slug": "Notion",
        "composio_tool_slug": "NOTION_QUERY_DATABASE",
        "node_ids": [
          "get_portfolio_value",
          "summarize_performance"
        ]
      },
      {
        "name": "Stock_Market_Data_API_READ",
        "description": "To get real-time or near real-time stock prices and other market data.",
        "is_composio_tool": false,
        "py_code": "import threading\nimport signal\nimport time\nimport sys\nfrom threading import Event\nfrom intriniorealtime.equities_client import IntrinioRealtimeEquitiesClient\n\ntrade_count = 0\nask_count = 0\nbid_count = 0\nbacklog_count = 0\n\ndef on_quote(quote, backlog):\n    global ask_count, bid_count, backlog_count\n    backlog_count = backlog\n    if hasattr(quote, 'type'):\n        if quote.type == \"ask\":\n            ask_count += 1\n        else:\n            bid_count += 1\n\ndef on_trade(trade, backlog):\n    global trade_count, backlog_count\n    backlog_count = backlog\n    trade_count += 1\n\nconfiguration = {\n    'api_key': 'API_KEY_HERE',\n    'provider': 'IEX',  # REALTIME (IEX), or IEX, or DELAYED_SIP, or NASDAQ_BASIC, or CBOE_ONE\n    'on_quote': on_quote,\n    'on_trade': on_trade\n}\n\nclient = IntrinioRealtimeEquitiesClient(configuration)\n\nstop_event = Event()\n\ndef on_kill_process(sig, frame):\n    print(\"Stopping\")\n    stop_event.set()\n    client.disconnect()\n    sys.exit(0)\n\nsignal.signal(signal.SIGINT, on_kill_process)\n\nclient.join(['AAPL', 'GE', 'MSFT'])\nclient.connect()\n\ntime.sleep(120)\n\nprint(\"Stopping\")\nstop_event.set()\nclient.disconnect()\nsys.exit(0)",
        "node_ids": [
          "get_portfolio_value",
          "set_alert",
          "summarize_performance"
        ]
      },
      {
        "name": "Notification_Service_CREATE",
        "description": "To send notifications to users via various channels.",
        "is_composio_tool": false,
        "py_code": "from suprsend import SuprSend\n\n# Initialize SuprSend client\nclient = SuprSend(workspace_key=\"YOUR_WORKSPACE_KEY\", workspace_secret=\"YOUR_WORKSPACE_SECRET\")\n\n# Send email notification\nresponse = client.send_email(\n    to=\"recipient@example.com\",\n    subject=\"Test Email\",\n    body=\"Hello from SuprSend Python SDK!\"\n)\n\nprint(response)",
        "node_ids": [
          "set_alert"
        ]
      }
    ]
  }'''