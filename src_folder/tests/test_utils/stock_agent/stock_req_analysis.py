stock_req_analysis = r'''
{
    "purposes": [
      {
        "purpose": "Investment Monitoring",
        "emoji": "🎯",
        "description": "The primary goal is to help users keep an eye on their investments and understand how they are performing.",
        "confident": true,
        "selected": true
      },
      {
        "purpose": "Proactive Alerting",
        "emoji": "🚨",
        "description": "The agent aims to provide timely notifications to users based on predefined conditions, helping them make informed decisions.",
        "confident": true,
        "selected": true
      },
      {
        "purpose": "Performance Reporting",
        "emoji": "📝",
        "description": "The agent will simplify the process of understanding portfolio performance by generating clear and concise summaries.",
        "confident": true,
        "selected": true
      },
      {
        "purpose": "Portfolio Management",
        "emoji": "💼",
        "description": "The agent could help users manage their investment portfolio by providing tools for tracking and analysis.",
        "confident": false,
        "selected": true
      }
    ],
    "capabilities": [
      {
        "capability": "Track Investments",
        "emoji": "📈",
        "description": "The agent will monitor and record all investment activities, including purchases and sales of stocks.",
        "does_need_trigger": "no",
        "confident": true,
        "selected": true
      },
      {
        "capability": "Fetch Stock Prices",
        "emoji": "💲",
        "description": "The agent will retrieve real-time or near real-time stock prices for specified equities.",
        "does_need_trigger": "no",
        "confident": true,
        "selected": true
      },
      {
        "capability": "Generate Alerts",
        "emoji": "🔔",
        "description": "The agent will send notifications based on predefined thresholds or conditions for stock prices.",
        "does_need_trigger": "yes",
        "confident": true,
        "selected": true
      },
      {
        "capability": "Summarize Performance",
        "emoji": "📊",
        "description": "The agent will provide summaries of portfolio value and performance over specified periods.",
        "does_need_trigger": "no",
        "confident": true,
        "selected": true
      },
      {
        "capability": "Manage Portfolio",
        "emoji": "💼",
        "description": "The agent could allow users to add, modify, or remove investments from their portfolio.",
        "does_need_trigger": "no",
        "confident": false,
        "selected": true
      }
    ],
    "knowledge_sources": [],
    "targetted_users": [
      {
        "persona": "Individual Investors",
        "emoji": "🧑‍💻",
        "description": "These users need to stay informed about their investments without constant manual checking.",
        "confident": false,
        "selected": true
      }
    ],
    "toolings": [
      {
        "tool": "Investment Database",
        "emoji": "💾",
        "description": "To store and retrieve user's investment records, including stock purchases and sales.",
        "specific_tasks": [
          "CREATE",
          "READ",
          "UPDATE",
          "DELETE"
        ],
        "confident": true,
        "tooling_type": "invokable",
        "selected": true
      },
      {
        "tool": "Stock Market Data API",
        "emoji": "💰",
        "description": "To get real-time or near real-time stock prices and other market data.",
        "specific_tasks": [
          "READ"
        ],
        "confident": true,
        "tooling_type": "invokable",
        "selected": true
      },
      {
        "tool": "Notification Service",
        "emoji": "✉️",
        "description": "To send notifications to users via various channels like email, SMS, or push notifications.",
        "specific_tasks": [
          "CREATE"
        ],
        "confident": true,
        "tooling_type": "invokable",
        "selected": true
      }
    ],
    "additional_information": "Domain: Finance (Stock Portfolio)\nExamples:\n- Input: \"Add 10 shares of AAPL at $180.\" → Action: Store trade.\n- Input: \"What’s my portfolio value?\" → Output: \"$12,340, up 4%.\"\n- Input: \"Notify if TSLA < $200.\" → Action: Monitor and alert.",
    "user_selections": {
      "customPurposes": [],
      "customCapabilities": [],
      "customKnowledgeSources": [],
      "customUsers": [],
      "customToolings": []
    }
  }'''