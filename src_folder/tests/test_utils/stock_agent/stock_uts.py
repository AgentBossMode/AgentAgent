stock_uts =r'''
{
    "trajectory_uts": [
      {
        "input_dict": "{\"messages\": [{\"role\": \"user\", \"content\": \"Add 10 shares of AAPL at $180.\"}]}",
        "expected_trajectory": [
          "route_input",
          "add_investment",
          "end"
        ]
      },
      {
        "input_dict": "{\"messages\": [{\"role\": \"user\", \"content\": \"What’s my portfolio value?\"}]}",
        "expected_trajectory": [
          "end"
        ]
      },
      {
        "input_dict":  "{\"messages\": [{\"role\": \"user\", \"content\": \"Notify if TSLA < $200.\"}]}",
        "expected_trajectory": [
          "route_input",
          "set_alert",
          "end"
        ]
      },
      {
        "input_dict": "{\"messages\": [{\"role\": \"user\", \"content\": \"Scheduled alert check for all monitored stocks.\"}]}",
        "expected_trajectory": [
          "route_input",
          "set_alert",
          "end"
        ]
      }
    ]
  }'''