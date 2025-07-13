use_cases= """
<USE_CASES>
{
  "name": "Continuous Market Analysis",
  "description": "Continuously analyze market trends, economic indicators, geopolitical news, and social media sentiment across various asset classes in real-time.",
  "dry_run": "The 'market_data_ingestion' node is responsible for ingesting real-time financial market data, including stock prices, trading feeds, news, and social media sentiment. It uses tools like 'fetch_stock_data', 'fetch_financial_news', and 'analyze_social_sentiment'. This data then flows to 'market_data_analysis' for further processing. The loop back from 'risk_management_and_strategy_adjustment' to 'market_data_ingestion' ensures continuous monitoring and analysis. This satisfies the use case."
}
{
  "name": "Predictive Modeling and Opportunity/Risk Identification",
  "description": "Predict future price movements and identify fleeting micro-opportunities or emerging risks.",
  "dry_run": "The 'predictive_modeling' node is explicitly designed to apply predictive models to forecast price movements, identify trading opportunities, and assess potential risks. This directly addresses the use case. The output of this node ('trading_opportunity' and 'risk_assessment') feeds into subsequent decision-making."
}
{
  "name": "Autonomous Trade Execution",
  "description": "Execute buy/sell orders and manage positions without direct human validation for pre-approved strategies.",
  "dry_run": "The 'trade_execution' node's description explicitly states: 'Executes buy/sell orders and manages positions based on identified trading opportunities and pre-approved strategies. This node operates without direct human validation for pre-approved strategies.' This, along with the 'execute_buy_order' and 'execute_sell_order' tools, directly satisfies the use case. The 'get_portfolio_positions' tool also supports position management."
}
{
  "name": "Dynamic Strategy and Risk Adjustment",
  "description": "Dynamically adjust trading strategies and risk parameters based on evolving market conditions, volatility, or pre-set thresholds.",
  "dry_run": "The 'risk_management_and_strategy_adjustment' node is designed to 'Dynamically adjusts trading strategies and risk parameters based on evolving market conditions, volatility, or pre-set thresholds. Also flags significant events for human review.' It utilizes tools like 'update_strategy_parameters' and 'adjust_risk_exposure'. This directly addresses the use case. The conditional edge from 'predictive_modeling' to this node also allows for adjustments when no immediate trading opportunity is found or human review is needed."
}
</USE_CASES>
"""