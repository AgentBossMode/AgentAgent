stock_dry_runs = r'''
{
    "dry_runs": [
      {
        "input": "Add 10 shares of AAPL at $180.",
        "agent_actions": [
          {
            "toolings": "Investment Database (CREATE)",
            "action": "The agent receives the user's request to add an investment. It then uses the 'Investment Database' tool to record the purchase of 10 shares of AAPL at $180."
          }
        ],
        "output": "Investment of 10 shares of AAPL at $180 successfully recorded.",
        "selected": true,
        "confident": true
      },
      {
        "input": "What’s my portfolio value?",
        "agent_actions": [
          {
            "toolings": "Investment Database (READ), Stock Market Data API (READ)",
            "action": "The agent receives a request for the portfolio value. It uses the 'Investment Database' to retrieve all investment records. Then, it uses the 'Stock Market Data API' to fetch current prices for all stocks in the portfolio. Finally, it calculates the total portfolio value and performance."
          }
        ],
        "output": "$12,340, up 4%.",
        "selected": true,
        "confident": true
      },
      {
        "input": "Notify if TSLA < $200.",
        "agent_actions": [
          {
            "toolings": "Stock Market Data API (READ), Notification Service (CREATE)",
            "action": "The agent receives a request to set up an alert. It records the alert condition (TSLA < $200) in its system. Periodically, or upon a trigger, the agent will use the 'Stock Market Data API' to fetch the current price of TSLA. If the condition is met, it will use the 'Notification Service' to send an alert to the user."
          }
        ],
        "output": "Alert for TSLA < $200 successfully set.",
        "selected": true,
        "confident": true
      },
      {
        "input": "Generate a weekly performance summary.",
        "agent_actions": [
          {
            "toolings": "Investment Database (READ), Stock Market Data API (READ)",
            "action": "The agent is triggered by a scheduled event or a change in stock price. It uses the 'Investment Database' to retrieve all investment records. Then, it uses the 'Stock Market Data API' to fetch current prices for all stocks in the portfolio. Finally, it calculates the total portfolio value and performance over a specified period."
          }
        ],
        "output": "Weekly portfolio performance summary generated: Portfolio value $12,500, up 5% this week.",
        "selected": true,
        "confident": true
      }
    ],
    "user_selections": {
      "selectedDryRuns": [
        0,
        1,
        2,
        3
      ],
      "customDryRuns": []
    }
  }'''