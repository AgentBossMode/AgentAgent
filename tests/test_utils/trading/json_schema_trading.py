json_schema_trading = """
{{
      "justification": "The \"Plan and Execute\" architecture is suitable for this problem. The agent needs to continuously analyze market data (plan) and then execute trades or adjust strategies based on its analysis (execute). The continuous monitoring and adjustment aspects align well with an iterative planning and execution cycle. The inclusion of a \"Human in the Loop\" node allows for intervention and review when necessary, particularly for critical decisions or when confidence is low, aligning with the need for human review in the examples. The primary change was to explicitly route from 'market_analysis' to 'prediction_and_decision' to ensure that predictions and decisions are always made based on the latest market analysis, and to add a conditional edge from 'prediction_and_decision' to 'human_review' for cases where the prediction confidence is low or a critical decision requires human oversight.",
      "nodes": [
        {{
          "id": "market_analysis",
          "schema_info": "GraphState:\n type: TypedDict\n fields:\n - name: market_data\n type: dict\n - name: analysis_result\n type: dict",
          "input_schema": "GraphState",
          "output_schema": "GraphState",
          "description": "Continuously analyzes market trends, economic indicators, geopolitical news, and social media sentiment across various asset classes in real-time. This node acts as the 'plan' step, gathering and processing all relevant market information.",
          "function_name": "analyze_market"
        }},
        {{
          "id": "prediction_and_decision",
          "schema_info": "GraphState:\n type: TypedDict\n fields:\n - name: analysis_result\n type: dict\n - name: trade_opportunity\n type: bool\n - name: risk_identified\n type: bool\n - name: decision\n type: str",
          "input_schema": "GraphState",
          "output_schema": "GraphState",
          "description": "Predicts future price movements and identifies fleeting micro-opportunities or emerging risks based on the market analysis. This node also determines if a trade should be executed or a strategy adjusted.",
          "function_name": "predict_and_decide"
        }},
        {{
          "id": "trade_execution",
          "schema_info": "GraphState:\n type: TypedDict\n fields:\n - name: decision\n type: str\n - name: trade_details\n type: dict\n - name: trade_status\n type: str",
          "input_schema": "GraphState",
          "output_schema": "GraphState",
          "description": "Executes buy/sell orders and manages positions without direct human validation for pre-approved strategies. This is a core 'execute' step.",
          "function_name": "execute_trade"
        }},
        {{
          "id": "position_monitoring",
          "schema_info": "GraphState:\n type: TypedDict\n fields:\n - name: trade_status\n type: str\n - name: position_data\n type: dict\n - name: monitoring_result\n type: dict",
          "input_schema": "GraphState",
          "output_schema": "GraphState",
          "description": "Monitors open positions, ready to sell if conditions change, target profit is met, or a stop-loss is triggered. This node continuously evaluates the performance of active trades.",
          "function_name": "monitor_position"
        }},
        {{
          "id": "strategy_adjustment",
          "schema_info": "GraphState:\n type: TypedDict\n fields:\n - name: monitoring_result\n type: dict\n - name: market_data\n type: dict\n - name: adjustment_details\n type: dict\n - name: human_review_needed\n type: bool",
          "input_schema": "GraphState",
          "output_schema": "GraphState",
          "description": "Dynamically adjusts trading strategies and risk parameters based on evolving market conditions, volatility, or pre-set thresholds. This node also determines if human review is needed.",
          "function_name": "adjust_strategy"
        }},
        {{
          "id": "human_review",
          "schema_info": "GraphState:\n type: TypedDict\n fields:\n - name: adjustment_details\n type: dict\n - name: review_outcome\n type: str",
          "input_schema": "GraphState",
          "output_schema": "GraphState",
          "description": "Flags events or decisions for human review and potential strategic re-evaluation, especially for critical adjustments or when confidence is low.",
          "function_name": "human_review_needed"
        }}
      ],
      "edges": [
        {{
          "source": "__START__",
          "target": "market_analysis",
          "routing_conditions": "Always proceeds to market_analysis.",
          "conditional": false
        }},
        {{
          "source": "market_analysis",
          "target": "prediction_and_decision",
          "routing_conditions": "If market analysis indicates a trading opportunity, proceed to prediction_and_decision. Otherwise, loop back to market_analysis for continuous monitoring.",
          "conditional": true
        }},
        {{
          "source": "prediction_and_decision",
          "target": "trade_execution",
          "routing_conditions": "If prediction and decision indicates a trade should be executed, proceed to trade_execution. If no trade is needed, loop back to market_analysis for continuous monitoring. If human review is needed, proceed to human_review.",
          "conditional": true
        }},
        {{
          "source": "trade_execution",
          "target": "position_monitoring",
          "routing_conditions": "If trade execution is successful and position needs monitoring, proceed to position_monitoring. If trade execution fails or no position is opened, loop back to market_analysis.",
          "conditional": true
        }},
        {{
          "source": "position_monitoring",
          "target": "strategy_adjustment",
          "routing_conditions": "If position monitoring indicates a need for adjustment or closure, proceed to strategy_adjustment. Otherwise, loop back to position_monitoring for continuous monitoring.",
          "conditional": true
        }},
        {{
          "source": "strategy_adjustment",
          "target": "trade_execution",
          "routing_conditions": "If strategy adjustment leads to a new trade, proceed to trade_execution. If adjustment is complete and no further action is needed, loop back to market_analysis for continuous monitoring. If human review is required, proceed to human_review.",
          "conditional": true
        }},
        {{
          "source": "human_review",
          "target": "market_analysis",
          "routing_conditions": "Always proceeds to market_analysis after human review.",
          "conditional": false
        }}
      ],
      "tools": [
        {{
          "name": "get_market_data",
          "description": "Retrieves real-time stock market data, high-frequency trading feeds, financial news headlines, and sentiment analysis from major financial forums.",
          "is_composio_tool": false,
          "node_ids": [
            "market_analysis"
          ]
        }},
        {{
          "name": "execute_buy_order",
          "description": "Executes a buy order for a specified asset and quantity.",
          "is_composio_tool": false,
          "node_ids": [
            "trade_execution"
          ]
        }},
        {{
          "name": "execute_sell_order",
          "description": "Executes a sell order for a specified asset and quantity.",
          "is_composio_tool": false,
          "node_ids": [
            "trade_execution"
          ]
        }},
        {{
          "name": "get_open_positions",
          "description": "Retrieves current open positions and their performance.",
          "is_composio_tool": false,
          "node_ids": [
            "position_monitoring"
          ]
        }},
        {{
          "name": "adjust_risk_parameters",
          "description": "Adjusts risk parameters for the trading strategy.",
          "is_composio_tool": false,
          "node_ids": [
            "strategy_adjustment"
          ]
        }},
        {{
          "name": "update_trading_strategy",
          "description": "Updates the trading strategy based on new market conditions.",
          "is_composio_tool": false,
          "node_ids": [
            "strategy_adjustment"
          ]
        }}
      ]
    }},
    "justification": "The original graph had a direct edge from `market_analysis` to `trade_execution` with a conditional routing based on whether market analysis indicated a trading opportunity. This bypasses the explicit `prediction_and_decision` node, which is crucial for identifying opportunities and risks based on the analysis. The updated graph ensures that after `market_analysis`, the flow always proceeds to `prediction_and_decision` to make an informed determination. Additionally, a conditional edge from `prediction_and_decision` to `human_review` was added. This addresses the need for human intervention when the prediction confidence is low or a critical decision requires human oversight, as implied by the examples and the overall objective of managing risk. The original graph only had `human_review` after `strategy_adjustment`, which might be too late for critical initial trade decisions. The rest of the graph structure and node functionalities remain appropriate for the stated objectives and use cases."
}}
"""