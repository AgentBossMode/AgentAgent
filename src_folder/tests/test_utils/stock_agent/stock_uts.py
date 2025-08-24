stock_uts =r'''
{
    "trajectory_uts": [
      {
        "input": "Add 10 shares of AAPL at $180.",
        "expected_trajectory": [
          "route_input",
          "add_investment",
          "end"
        ]
      },
      {
        "input": "What’s my portfolio value?",
        "expected_trajectory": [
          "route_input",
          "get_portfolio_value",
          "end"
        ]
      },
      {
        "input": "Notify if TSLA < $200.",
        "expected_trajectory": [
          "route_input",
          "set_alert",
          "end"
        ]
      },
      {
        "input": "Scheduled alert check for all monitored stocks.",
        "expected_trajectory": [
          "route_input",
          "check_alerts",
          "end"
        ]
      }
    ]
  }'''