nutrition_req_analysis = r'''
{
    "purposes": [
      {
        "purpose": "Calorie Tracking",
        "emoji": "📈",
        "description": "To help users track their daily calorie consumption and expenditure.",
        "confident": true,
        "selected": true
      },
      {
        "purpose": "Historical Analysis",
        "emoji": "🔍",
        "description": "To provide users with insights into their historical diet and exercise patterns.",
        "confident": true,
        "selected": true
      },
      {
        "purpose": "Net Calorie Management",
        "emoji": "⚖️",
        "description": "To enable users to understand their net calorie balance over time.",
        "confident": true,
        "selected": true
      },
      {
        "purpose": "Activity Automation",
        "emoji": "🤖",
        "description": "To automate the process of logging food and exercise, reducing manual effort.",
        "confident": false,
        "selected": true
      }
    ],
    "capabilities": [
      {
        "capability": "Calorie Tracking",
        "emoji": "📊",
        "description": "Calculate and store calories consumed from food and burned from exercise.",
        "does_need_trigger": "on-demand",
        "confident": true,
        "selected": true
      },
      {
        "capability": "Historical Data Retrieval",
        "emoji": "🕰️",
        "description": "Retrieve and present past food consumption and exercise data.",
        "does_need_trigger": "on-demand",
        "confident": true,
        "selected": true
      },
      {
        "capability": "Net Calorie Calculation",
        "emoji": "⚖️",
        "description": "Calculate and display the net calorie balance over a specified period.",
        "does_need_trigger": "on-demand",
        "confident": true,
        "selected": true
      },
      {
        "capability": "Data Logging",
        "emoji": "📝",
        "description": "Log and store user-reported food and exercise activities.",
        "does_need_trigger": "on-demand",
        "confident": false,
        "selected": true
      }
    ],
    "knowledge_sources": [
      {
        "knowledge_source": "Dietary Guidelines",
        "emoji": "📚",
        "description": "Provides guidelines and recommendations for daily calorie intake based on various factors like age, gender, and activity level.",
        "confident": false,
        "selected": true
      }
    ],
    "targetted_users": [],
    "toolings": [
      {
        "tool": "Calorie Database",
        "emoji": "💾",
        "description": "To store and retrieve all calorie-related data, including food intake and exercise expenditure.",
        "specific_tasks": [
          "Create",
          "Read",
          "Update",
          "Delete"
        ],
        "confident": true,
        "tooling_type": "invokable",
        "selected": true
      },
      {
        "tool": "Food Calorie API",
        "emoji": "🍔",
        "description": "To search for calorie information of various food items from a comprehensive external database.",
        "specific_tasks": [
          "Search"
        ],
        "confident": false,
        "tooling_type": "invokable",
        "selected": true
      },
      {
        "tool": "Exercise Calorie API",
        "emoji": "🏃",
        "description": "To search for calorie expenditure data for different types of exercises and durations.",
        "specific_tasks": [
          "Search"
        ],
        "confident": false,
        "tooling_type": "invokable",
        "selected": true
      }
    ],
    "additional_information": "The user wants to track calorie intake and expenditure, and get insights into their diet and exercise history. The agent should be able to calculate and store calorie data for both food and exercise, and retrieve historical data for Q&A.",
  }'''

nutrition_dry_runs = r'''
{
"dry_runs": [
      {
        "input_type": "on-demand",
        "input": "I ate a banana and ran for 30 minutes.",
        "agent_actions": [
          {
            "determinism": "llm",
            "knowledge": "Food Calorie Database",
            "toolings": "Calorie Log",
            "action": "Using the 'Food Calorie Database', agent identifies the calorie content of 'banana' and then uses the 'Calorie Log' tool to record the food intake."
          },
          {
            "determinism": "llm",
            "knowledge": "Exercise Calorie Database",
            "toolings": "Calorie Log",
            "action": "Using the 'Exercise Calorie Database', agent estimates calories burned for '30-minute run' and then uses the 'Calorie Log' tool to record the exercise."
          }
        ],
        "output": "Understood. A banana is approximately 105 calories. A 30-minute run typically burns around 300-450 calories, depending on intensity. This data has been recorded.",
        "selected": true
      },
      {
        "input_type": "on-demand",
        "input": "I ran for 20 minutes.",
        "agent_actions": [
          {
            "determinism": "llm",
            "knowledge": "Exercise Calorie Database",
            "toolings": "Calorie Log",
            "action": "Using the 'Exercise Calorie Database', agent estimates calories burned for '20-minute run' and then uses the 'Calorie Log' tool to record the exercise."
          }
        ],
        "output": "Okay, a 20-minute run has been logged. This typically burns around 200-300 calories. This data has been stored.",
        "selected": true
      },
      {
        "input_type": "on-demand",
        "input": "How many calories did I burn yesterday?",
        "agent_actions": [
          {
            "determinism": "deterministic",
            "knowledge": "Calorie Database",
            "toolings": "Calorie Data Retrieval",
            "action": "Using the 'Calorie Database', agent retrieves total calories burned for 'yesterday'."
          }
        ],
        "output": "Yesterday, you burned a total of [X] calories through your activities.",
        "selected": true
      },
      {
        "input_type": "on-demand",
        "input": "What did I eat on Monday?",
        "agent_actions": [
          {
            "determinism": "deterministic",
            "knowledge": "Calorie Database",
            "toolings": "Calorie Data Retrieval",
            "action": "Using the 'Calorie Database', agent retrieves food items and calorie counts for 'Monday'."
          }
        ],
        "output": "On Monday, you reported eating: [List of food items and their calorie counts].",
        "selected": true
      },
      {
        "input_type": "on-demand",
        "input": "What's my net calorie intake for the week?",
        "agent_actions": [
          {
            "determinism": "deterministic",
            "knowledge": "Calorie Database",
            "toolings": "Calorie Data Retrieval",
            "action": "Using the 'Calorie Database', agent retrieves total calorie intake and burned for 'this week' and calculates the net."
          }
        ],
        "output": "This week, your total calorie intake was [X] and your total calories burned were [Y], resulting in a net of [X-Y] calories.",
        "selected": true
      }
    ]
}'''