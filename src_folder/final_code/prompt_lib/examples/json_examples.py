json_example_ecommerce = r'''
### Example 1: E-commerce Customer Support Agent with Multi-Agent Architecture
Input: "Create a customer support workflow that takes customer queries, analyzes them, routes to appropriate specialist agents (technical, billing, general), each agent researches using knowledge base and external APIs, generates responses, validates them, and if confidence is low, escalates to human agent"

Expected Schema Pattern:
{
  "graphstate": {
    "type": "TypedDict",
    "fields": [
      {
        "name": "customer_query",
        "type": "str",
        "description": "Original customer support query"
      },
      {
        "name": "query_category",
        "type": "str",
        "description": "Classified category (technical, billing, general)"
      },
      {
        "name": "confidence_score",
        "type": "float",
        "description": "Confidence score for classification (0-1)"
      },
      {
        "name": "knowledge_base_results",
        "type": "List[dict]",
        "description": "Relevant articles from knowledge base"
      },
      {
        "name": "external_data",
        "type": "dict",
        "description": "Data from external APIs (account info, product specs, etc.)"
      },
      {
        "name": "agent_response",
        "type": "str",
        "description": "Generated response from specialist agent"
      },
      {
        "name": "response_quality_score",
        "type": "float",
        "description": "Quality validation score for the response"
      },
      {
        "name": "escalation_needed",
        "type": "bool",
        "description": "Whether human escalation is required"
      },
      {
        "name": "final_response",
        "type": "str",
        "description": "Final response sent to customer"
      }
    ]
  },
  "tools": [
    {
      "name": "Knowledge_Base_Search",
      "description": "Searches internal knowledge base for relevant articles",
      "node_ids": [
        "technical_agent",
        "billing_agent",
        "general_agent"
      ]
    },
    {
      "name": "Customer_Account_API",
      "description": "Retrieves customer account information",
      "node_ids": [
        "billing_agent"
      ]
    },
    {
      "name": "Product_Specs_API",
      "description": "Gets detailed product specifications",
      "node_ids": [
        "technical_agent"
      ]
    },
    {
      "name": "Response_Quality_Validator",
      "description": "Validates response quality and accuracy",
      "node_ids": [
        "validate_response"
      ]
    }
  ],
  "nodes": [
    {
      "id": "__START__",
      "function_name": "start_node",
      "description": "Entry point of the workflow",
      "llm_actions": [],
      "input_schema": [],
      "output_schema": []
      
    },
    {
      "id": "analyze_and_route",
      "function_name": "analyze_and_route_query",
      "description": "Analyzes customer query and determines routing to appropriate specialist agent",
      "llm_actions": [
        "analyze",
        "route"
      ],
      "input_schema": [
        {
          "name": "customer_query",
          "type": "str",
          "description": "Raw customer query to be analyzed and categorized"
        }
      ],
      "output_schema": [
        {
          "name": "query_category",
          "type": "str",
          "description": "Determined category for routing (technical/billing/general)"
        },
        {
          "name": "confidence_score",
          "type": "float",
          "description": "Confidence in the categorization decision"
        }
      ],
      
    },
    {
      "id": "technical_agent",
      "function_name": "handle_technical_query",
      "description": "Handles technical queries by researching knowledge base and product specs, then generating technical response",
      "llm_actions": [
        "tool_call",
        "aggregate",
        "analyze",
        "generate"
      ],
      "input_schema": [
        {
          "name": "customer_query",
          "type": "str",
          "description": "Technical query to be resolved"
        }
      ],
      "output_schema": [
        {
          "name": "knowledge_base_results",
          "type": "List[dict]",
          "description": "Retrieved technical documentation and solutions"
        },
        {
          "name": "external_data",
          "type": "dict",
          "description": "Product specifications and technical details"
        },
        {
          "name": "agent_response",
          "type": "str",
          "description": "Generated technical response combining multiple sources"
        }
      ],
      "tools_used": [
        "Knowledge_Base_Search",
        "Product_Specs_API"
      ]
    },
    {
      "id": "billing_agent",
      "function_name": "handle_billing_query",
      "description": "Handles billing queries by accessing customer account and generating billing response",
      "llm_actions": [
        "tool_call",
        "aggregate",
        "analyze",
        "generate"
      ],
      "input_schema": [
        {
          "name": "customer_query",
          "type": "str",
          "description": "Billing-related query to be resolved"
        }
      ],
      "output_schema": [
        {
          "name": "knowledge_base_results",
          "type": "List[dict]",
          "description": "Retrieved billing policies and procedures"
        },
        {
          "name": "external_data",
          "type": "dict",
          "description": "Customer account information and billing history"
        },
        {
          "name": "agent_response",
          "type": "str",
          "description": "Generated billing response with account-specific information"
        }
      ],
      "tools_used": [
        "Knowledge_Base_Search",
        "Customer_Account_API"
      ]
    },
    {
      "id": "general_agent",
      "function_name": "handle_general_query",
      "description": "Handles general queries using knowledge base search and generates general response",
      "llm_actions": [
        "tool_call",
        "analyze",
        "generate"
      ],
      "input_schema": [
        {
          "name": "customer_query",
          "type": "str",
          "description": "General query to be resolved"
        }
      ],
      "output_schema": [
        {
          "name": "knowledge_base_results",
          "type": "List[dict]",
          "description": "Retrieved general information and FAQs"
        },
        {
          "name": "agent_response",
          "type": "str",
          "description": "Generated general response based on knowledge base"
        }
      ],
      "tools_used": [
        "Knowledge_Base_Search"
      ]
    },
    {
      "id": "validate_response",
      "function_name": "validate_agent_response",
      "description": "Validates the quality and accuracy of agent responses using AI validation tools",
      "llm_actions": [
        "tool_call",
        "validate",
        "analyze"
      ],
      "input_schema": [
        {
          "name": "agent_response",
          "type": "str",
          "description": "Response generated by specialist agent"
        },
        {
          "name": "customer_query",
          "type": "str",
          "description": "Original customer query for validation context"
        }
      ],
      "output_schema": [
        {
          "name": "response_quality_score",
          "type": "float",
          "description": "Quality score of the response (0-1)"
        },
        {
          "name": "escalation_needed",
          "type": "bool",
          "description": "Whether response quality requires human escalation"
        }
      ],
      "tools_used": [
        "Response_Quality_Validator"
      ]
    },
    {
      "id": "human_escalation",
      "function_name": "escalate_to_human",
      "description": "Handles human escalation for complex or low-confidence responses",
      "llm_actions": [
        "aggregate",
        "generate"
      ],
      "input_schema": [
        {
          "name": "customer_query",
          "type": "str",
          "description": "Original customer query needing human attention"
        },
        {
          "name": "agent_response",
          "type": "str",
          "description": "AI-generated response that needs human review"
        },
        {
          "name": "knowledge_base_results",
          "type": "List[dict]",
          "description": "Research results to provide context to human agent"
        }
      ],
      "output_schema": [
        {
          "name": "final_response",
          "type": "str",
          "description": "Human-reviewed and approved final response"
        }
      ],
      
    },
    {
      "id": "finalize_response",
      "function_name": "prepare_final_response",
      "description": "Finalizes the approved response for customer delivery",
      "llm_actions": [
        "transform",
        "generate"
      ],
      "input_schema": [
        {
          "name": "agent_response",
          "type": "str",
          "description": "Validated AI response ready for delivery"
        }
      ],
      "output_schema": [
        {
          "name": "final_response",
          "type": "str",
          "description": "Final formatted response for customer"
        }
      ],
      
    },
    {
      "id": "__END__",
      "function_name": "end_node",
      "description": "End point of the workflow",
      "llm_actions": [],
      "input_schema": [],
      "output_schema": []
      
    }
  ],
  "edges": [
    {
      "source": "__START__",
      "target": "analyze_and_route",
      "conditional": false,
      "routing_conditions": ""
    },
    {
      "source": "analyze_and_route",
      "target": "technical_agent",
      "conditional": true,
      "routing_conditions": "query_category == 'technical'"
    },
    {
      "source": "analyze_and_route",
      "target": "billing_agent",
      "conditional": true,
      "routing_conditions": "query_category == 'billing'"
    },
    {
      "source": "analyze_and_route",
      "target": "general_agent",
      "conditional": true,
      "routing_conditions": "query_category == 'general'"
    },
    {
      "source": "technical_agent",
      "target": "validate_response",
      "conditional": false,
      "routing_conditions": ""
    },
    {
      "source": "billing_agent",
      "target": "validate_response",
      "conditional": false,
      "routing_conditions": ""
    },
    {
      "source": "general_agent",
      "target": "validate_response",
      "conditional": false,
      "routing_conditions": ""
    },
    {
      "source": "validate_response",
      "target": "human_escalation",
      "conditional": true,
      "routing_conditions": "escalation_needed == True or response_quality_score < 0.7"
    },
    {
      "source": "validate_response",
      "target": "finalize_response",
      "conditional": true,
      "routing_conditions": "escalation_needed == False and response_quality_score >= 0.7"
    },
    {
      "source": "human_escalation",
      "target": "__END__",
      "conditional": false,
      "routing_conditions": ""
    },
    {
      "source": "finalize_response",
      "target": "__END__",
      "conditional": false,
      "routing_conditions": ""
    }
  ],
  "justification": "This workflow implements a Multi-Agent Supervisor architecture with Human-in-the-Loop capabilities. The supervisor (analyze_and_route) classifies queries and routes to specialized agents. Each specialist agent performs multiple LLM actions: tool_call to gather data, aggregate to combine sources, analyze to understand context, and generate responses. A validation layer ensures quality control, with automatic escalation to humans for complex cases."
}
'''
json_example_marketing = r'''
### Example 2: Content Marketing Pipeline with Plan-and-Execute Architecture
Input: "Create a content marketing workflow that takes a topic, researches competitors and trends, creates a content plan, generates multiple content pieces (blog post, social media, email), gets approval for each, optimizes based on feedback, and publishes across platforms"

Expected Schema Pattern:
{
  "graphstate": {
    "type": "TypedDict",
    "fields": [
      {
        "name": "topic",
        "type": "str",
        "description": "Main topic for content marketing campaign"
      },
      {
        "name": "competitor_analysis",
        "type": "dict",
        "description": "Analysis of competitor content strategies"
      },
      {
        "name": "trend_data",
        "type": "List[dict]",
        "description": "Current trends related to the topic"
      },
      {
        "name": "content_plan",
        "type": "dict",
        "description": "Strategic plan for content creation across platforms"
      },
      {
        "name": "blog_content",
        "type": "str",
        "description": "Generated blog post content"
      },
      {
        "name": "social_media_content",
        "type": "List[dict]",
        "description": "Generated social media posts for different platforms"
      },
      {
        "name": "email_content",
        "type": "str",
        "description": "Generated email newsletter content"
      },
      {
        "name": "approval_status_blog",
        "type": "bool",
        "description": "Approval status for blog content"
      },
      {
        "name": "approval_status_social",
        "type": "bool",
        "description": "Approval status for social media content"
      },
      {
        "name": "approval_status_email",
        "type": "bool",
        "description": "Approval status for email content"
      },
      {
        "name": "feedback_blog",
        "type": "str",
        "description": "Human feedback on blog content"
      },
      {
        "name": "feedback_social",
        "type": "str",
        "description": "Human feedback on social media content"
      },
      {
        "name": "feedback_email",
        "type": "str",
        "description": "Human feedback on email content"
      },
      {
        "name": "optimized_content",
        "type": "dict",
        "description": "Content optimized based on feedback"
      },
      {
        "name": "publishing_schedule",
        "type": "dict",
        "description": "Planned publishing schedule across platforms"
      },
      {
        "name": "publication_results",
        "type": "dict",
        "description": "Results from publishing across all platforms"
      }
    ]
  },
  "tools": [
    {
      "name": "Competitor_Research_API",
      "description": "Analyzes competitor content strategies",
      "node_ids": [
        "research_and_plan"
      ]
    },
    {
      "name": "Trend_Analysis_API",
      "description": "Identifies current trends in the topic area",
      "node_ids": [
        "research_and_plan"
      ]
    },
    {
      "name": "SEO_Optimization_Tool",
      "description": "Optimizes content for search engines",
      "node_ids": [
        "optimize_content"
      ]
    },
    {
      "name": "Social_Media_Publisher",
      "description": "Publishes content to social media platforms",
      "node_ids": [
        "publish_content"
      ]
    },
    {
      "name": "Blog_CMS_API",
      "description": "Publishes blog content to CMS",
      "node_ids": [
        "publish_content"
      ]
    },
    {
      "name": "Email_Marketing_API",
      "description": "Sends email newsletters",
      "node_ids": [
        "publish_content"
      ]
    }
  ],
  "nodes": [
    {
      "id": "__START__",
      "function_name": "start_node",
      "description": "Entry point of the workflow",
      "llm_actions": [],
      "input_schema": [],
      "output_schema": []
      
    },
    {
      "id": "research_and_plan",
      "function_name": "research_and_create_plan",
      "description": "Researches competitors and trends, then creates comprehensive content strategy plan",
      "llm_actions": [
        "tool_call",
        "analyze",
        "aggregate",
        "generate"
      ],
      "input_schema": [
        {
          "name": "topic",
          "type": "str",
          "description": "Topic to research and create content plan for"
        }
      ],
      "output_schema": [
        {
          "name": "competitor_analysis",
          "type": "dict",
          "description": "Comprehensive analysis of competitor content strategies"
        },
        {
          "name": "trend_data",
          "type": "List[dict]",
          "description": "Current trends and insights related to the topic"
        },
        {
          "name": "content_plan",
          "type": "dict",
          "description": "Strategic content plan across all platforms"
        }
      ],
      "tools_used": [
        "Competitor_Research_API",
        "Trend_Analysis_API"
      ]
    },
    {
      "id": "generate_blog_content",
      "function_name": "create_blog_post",
      "description": "Generates long-form blog content based on the content plan and research",
      "llm_actions": [
        "analyze",
        "generate"
      ],
      "input_schema": [
        {
          "name": "content_plan",
          "type": "dict",
          "description": "Content strategy and guidelines for blog creation"
        },
        {
          "name": "competitor_analysis",
          "type": "dict",
          "description": "Competitor insights to inform content differentiation"
        },
        {
          "name": "trend_data",
          "type": "List[dict]",
          "description": "Trend data to incorporate into content"
        }
      ],
      "output_schema": [
        {
          "name": "blog_content",
          "type": "str",
          "description": "Complete blog post with SEO optimization"
        }
      ],
      
    },
    {
      "id": "generate_social_content",
      "function_name": "create_social_media_posts",
      "description": "Generates social media content for multiple platforms based on content plan",
      "llm_actions": [
        "analyze",
        "transform",
        "generate"
      ],
      "input_schema": [
        {
          "name": "content_plan",
          "type": "dict",
          "description": "Content strategy for social media platforms"
        },
        {
          "name": "trend_data",
          "type": "List[dict]",
          "description": "Trending hashtags and topics to incorporate"
        }
      ],
      "output_schema": [
        {
          "name": "social_media_content",
          "type": "List[dict]",
          "description": "Platform-specific social media posts with optimal formatting"
        }
      ],
      
    },
    {
      "id": "generate_email_content",
      "function_name": "create_email_newsletter",
      "description": "Generates email newsletter content based on content plan and blog content",
      "llm_actions": [
        "analyze",
        "transform",
        "generate"
      ],
      "input_schema": [
        {
          "name": "content_plan",
          "type": "dict",
          "description": "Email marketing strategy and guidelines"
        },
        {
          "name": "blog_content",
          "type": "str",
          "description": "Blog content to reference and link to in email"
        }
      ],
      "output_schema": [
        {
          "name": "email_content",
          "type": "str",
          "description": "Complete email newsletter with engaging subject line"
        }
      ],
      
    },
    {
      "id": "get_approvals",
      "function_name": "request_human_approvals",
      "description": "Presents all generated content to humans for approval and feedback collection",
      "llm_actions": [
        "validate",
        "aggregate"
      ],
      "input_schema": [
        {
          "name": "blog_content",
          "type": "str",
          "description": "Blog content requiring approval"
        },
        {
          "name": "social_media_content",
          "type": "List[dict]",
          "description": "Social media content requiring approval"
        },
        {
          "name": "email_content",
          "type": "str",
          "description": "Email content requiring approval"
        }
      ],
      "output_schema": [
        {
          "name": "approval_status_blog",
          "type": "bool",
          "description": "Whether blog content is approved"
        },
        {
          "name": "approval_status_social",
          "type": "bool",
          "description": "Whether social media content is approved"
        },
        {
          "name": "approval_status_email",
          "type": "bool",
          "description": "Whether email content is approved"
        },
        {
          "name": "feedback_blog",
          "type": "str",
          "description": "Specific feedback on blog content"
        },
        {
          "name": "feedback_social",
          "type": "str",
          "description": "Specific feedback on social media content"
        },
        {
          "name": "feedback_email",
          "type": "str",
          "description": "Specific feedback on email content"
        }
      ],
      
    },
    {
      "id": "optimize_content",
      "function_name": "optimize_based_on_feedback",
      "description": "Optimizes content based on human feedback and SEO best practices",
      "llm_actions": [
        "tool_call",
        "analyze",
        "transform",
        "generate"
      ],
      "input_schema": [
        {
          "name": "blog_content",
          "type": "str",
          "description": "Original blog content to optimize"
        },
        {
          "name": "social_media_content",
          "type": "List[dict]",
          "description": "Original social media content to optimize"
        },
        {
          "name": "email_content",
          "type": "str",
          "description": "Original email content to optimize"
        },
        {
          "name": "feedback_blog",
          "type": "str",
          "description": "Human feedback to incorporate"
        },
        {
          "name": "feedback_social",
          "type": "str",
          "description": "Social media feedback to incorporate"
        },
        {
          "name": "feedback_email",
          "type": "str",
          "description": "Email feedback to incorporate"
        }
      ],
      "output_schema": [
        {
          "name": "optimized_content",
          "type": "dict",
          "description": "All content pieces optimized based on feedback"
        },
        {
          "name": "publishing_schedule",
          "type": "dict",
          "description": "Optimal timing schedule for content publication"
        }
      ],
      "tools_used": [
        "SEO_Optimization_Tool"
      ]
    },
    {
      "id": "publish_content",
      "function_name": "publish_across_platforms",
      "description": "Publishes optimized content across all designated platforms",
      "llm_actions": [
        "tool_call",
        "validate"
      ],
      "input_schema": [
        {
          "name": "optimized_content",
          "type": "dict",
          "description": "Final optimized content ready for publication"
        },
        {
          "name": "publishing_schedule",
          "type": "dict",
          "description": "Schedule for when to publish each piece"
        }
      ],
      "output_schema": [
        {
          "name": "publication_results",
          "type": "dict",
          "description": "Results and status of publication across all platforms"
        }
      ],
      "tools_used": [
        "Social_Media_Publisher",
        "Blog_CMS_API",
        "Email_Marketing_API"
      ]
    },
    {
      "id": "__END__",
      "function_name": "end_node",
      "description": "End point of the workflow",
      "llm_actions": [],
      "input_schema": [],
      "output_schema": []
      
    }
  ],
  "edges": [
    {
      "source": "__START__",
      "target": "research_and_plan",
      "conditional": false,
      "routing_conditions": ""
    },
    {
      "source": "research_and_plan",
      "target": "generate_blog_content",
      "conditional": false,
      "routing_conditions": ""
    },
    {
      "source": "research_and_plan",
      "target": "generate_social_content",
      "conditional": false,
      "routing_conditions": ""
    },
    {
      "source": "research_and_plan",
      "target": "generate_email_content",
      "conditional": false,
      "routing_conditions": ""
    },
    {
      "source": "generate_blog_content",
      "target": "get_approvals",
      "conditional": false,
      "routing_conditions": ""
    },
    {
      "source": "generate_social_content",
      "target": "get_approvals",
      "conditional": false,
      "routing_conditions": ""
    },
    {
      "source": "generate_email_content",
      "target": "get_approvals",
      "conditional": false,
      "routing_conditions": ""
    },
    {
      "source": "get_approvals",
      "target": "optimize_content",
      "conditional": true,
      "routing_conditions": "approval_status_blog == False or approval_status_social == False or approval_status_email == False"
    },
    {
      "source": "get_approvals",
      "target": "publish_content",
      "conditional": true,
      "routing_conditions": "approval_status_blog == True and approval_status_social == True and approval_status_email == True"
    },
    {
      "source": "optimize_content",
      "target": "get_approvals",
      "conditional": false,
      "routing_conditions": ""
    },
    {
      "source": "publish_content",
      "target": "__END__",
      "conditional": false,
      "routing_conditions": ""
    }
  ],
  "justification": "This workflow implements a Plan-and-Execute architecture with Human-in-the-Loop validation. The planning phase (research_and_plan) creates a comprehensive strategy, followed by parallel execution of content generation. Each node performs multiple LLM actions combining tool calls for data gathering, analysis for insights, and generation for content creation. The approval loop allows for iterative refinement based on human feedback."
}
'''
json_example_report_finance = r'''

### Example 3: Financial Report Analysis with Self-Correction Architecture
Input: "Build a workflow that takes financial documents, extracts key metrics, validates data accuracy, performs trend analysis, generates insights, cross-references with market data, identifies anomalies, creates visualizations, and produces executive summary with confidence scores"

{
  "graphstate": {
    "type": "TypedDict",
    "fields": [
      {
        "name": "financial_documents",
        "type": "List[str]",
        "description": "Paths to uploaded financial documents"
      },
      {
        "name": "extracted_metrics",
        "type": "dict",
        "description": "Key financial metrics extracted from documents"
      },
      {
        "name": "data_quality_score",
        "type": "float",
        "description": "Quality score of extracted data (0-1)"
      },
      {
        "name": "validation_errors",
        "type": "List[str]",
        "description": "List of data validation errors found"
      },
      {
        "name": "corrected_metrics",
        "type": "dict",
        "description": "Financial metrics after error correction"
      },
      {
        "name": "trend_analysis",
        "type": "dict",
        "description": "Analysis of financial trends over time"
      },
      {
        "name": "market_data",
        "type": "dict",
        "description": "External market data for comparison"
      },
      {
        "name": "comparative_analysis",
        "type": "dict",
        "description": "Analysis comparing company metrics to market data"
      },
      {
        "name": "anomalies_detected",
        "type": "List[dict]",
        "description": "Detected financial anomalies with severity scores"
      },
      {
        "name": "visualizations",
        "type": "List[dict]",
        "description": "Generated charts and graphs data"
      },
      {
        "name": "insights",
        "type": "List[dict]",
        "description": "AI-generated financial insights with confidence scores"
      },
      {
        "name": "executive_summary",
        "type": "str",
        "description": "Executive summary of financial analysis"
      },
      {
        "name": "confidence_score",
        "type": "float",
        "description": "Overall confidence in the analysis (0-1)"
      },
      {
        "name": "needs_review",
        "type": "bool",
        "description": "Whether analysis requires human expert review"
      }
    ]
  },
  "tools": [
    {
      "name": "Document_Parser",
      "description": "Extracts text and tables from financial documents",
      "node_ids": [
        "extract_financial_data"
      ]
    },
    {
      "name": "OCR_Service",
      "description": "Optical character recognition for scanned documents",
      "node_ids": [
        "extract_financial_data"
      ]
    },
    {
      "name": "Financial_Data_Validator",
      "description": "Validates financial data for accuracy and completeness",
      "node_ids": [
        "validate_and_correct"
      ]
    },
    {
      "name": "Market_Data_API",
      "description": "Retrieves current and historical market data",
      "node_ids": [
        "cross_reference_market_data"
      ]
    },
    {
      "name": "Anomaly_Detection_API",
      "description": "Advanced anomaly detection for financial patterns",
      "node_ids": [
        "detect_anomalies"
      ]
    },
    {
      "name": "Visualization_Engine",
      "description": "Creates charts and graphs from financial data",
      "node_ids": [
        "create_visualizations"
      ]
    }
  ],
  "nodes": [
    {
      "id": "__START__",
      "function_name": "start_node",
      "description": "Entry point of the workflow",
      "llm_actions": [],
      "input_schema": [],
      "output_schema": []
      
    },
    {
      "id": "extract_financial_data",
      "function_name": "extract_key_metrics",
      "description": "Extracts and structures key financial metrics from documents using OCR and parsing tools",
      "llm_actions": [
        "tool_call",
        "analyze",
        "transform"
      ],
      "input_schema": [
        {
          "name": "financial_documents",
          "type": "List[str]",
          "description": "Financial documents to process and extract data from"
        }
      ],
      "output_schema": [
        {
          "name": "extracted_metrics",
          "type": "dict",
          "description": "Structured financial metrics extracted from documents"
        }
      ],
      "tools_used": [
        "Document_Parser",
        "OCR_Service"
      ]
    },
    {
      "id": "validate_and_correct",
      "function_name": "validate_data_quality",
      "description": "Validates extracted data quality and attempts automatic correction of errors",
      "llm_actions": [
        "tool_call",
        "validate",
        "analyze",
        "transform"
      ],
      "input_schema": [
        {
          "name": "extracted_metrics",
          "type": "dict",
          "description": "Raw extracted financial metrics needing validation"
        }
      ],
      "output_schema": [
        {
          "name": "data_quality_score",
          "type": "float",
          "description": "Calculated quality score for the extracted data"
        },
        {
          "name": "validation_errors",
          "type": "List[str]",
          "description": "List of validation errors that were found"
        },
        {
          "name": "corrected_metrics",
          "type": "dict",
          "description": "Financial metrics after automatic error correction"
        }
      ],
      "tools_used": [
        "Financial_Data_Validator"
      ]
    },
    {
      "id": "perform_trend_analysis",
      "function_name": "analyze_financial_trends",
      "description": "Analyzes financial trends and patterns in the corrected data",
      "llm_actions": [
        "analyze",
        "aggregate"
      ],
      "input_schema": [
        {
          "name": "corrected_metrics",
          "type": "dict",
          "description": "Validated financial metrics for trend analysis"
        }
      ],
      "output_schema": [
        {
          "name": "trend_analysis",
          "type": "dict",
          "description": "Comprehensive trend analysis with patterns and projections"
        }
      ],
      
    },
    {
      "id": "cross_reference_market_data",
      "function_name": "compare_with_market",
      "description": "Retrieves market data and performs comparative analysis",
      "llm_actions": [
        "tool_call",
        "analyze",
        "aggregate"
      ],
      "input_schema": [
        {
          "name": "corrected_metrics",
          "type": "dict",
          "description": "Company financial metrics to compare with market"
        }
      ],
      "output_schema": [
        {
          "name": "market_data",
          "type": "dict",
          "description": "Retrieved market data for comparison"
        },
        {
          "name": "comparative_analysis",
          "type": "dict",
          "description": "Analysis comparing company performance to market benchmarks"
        }
      ],
      "tools_used": [
        "Market_Data_API"
      ]
    },
    {
      "id": "detect_anomalies",
      "function_name": "identify_financial_anomalies",
      "description": "Detects anomalies and unusual patterns in financial data using AI tools",
      "llm_actions": [
        "tool_call",
        "analyze",
        "validate"
      ],
      "input_schema": [
        {
          "name": "corrected_metrics",
          "type": "dict",
          "description": "Financial data to scan for anomalies"
        },
        {
          "name": "trend_analysis",
          "type": "dict",
          "description": "Trend context for anomaly detection"
        }
      ],
      "output_schema": [
        {
          "name": "anomalies_detected",
          "type": "List[dict]",
          "description": "Detected anomalies with severity and impact scores"
        }
      ],
      "tools_used": [
        "Anomaly_Detection_API"
      ]
    },
    {
      "id": "create_visualizations",
      "function_name": "generate_charts_and_graphs",
      "description": "Creates visual representations of financial data and analysis results",
      "llm_actions": [
        "tool_call",
        "transform"
      ],
      "input_schema": [
        {
          "name": "corrected_metrics",
          "type": "dict",
          "description": "Financial data to visualize"
        },
        {
          "name": "trend_analysis",
          "type": "dict",
          "description": "Trend context for visualization"
        },
        {
          "name": "comparative_analysis",
          "type": "dict",
          "description": "Comparative analysis to visualize"
        },
        {
          "name": "anomalies_detected",
          "type": "List[dict]",
          "description": "Anomalies to highlight in visualizations"
        }
      ],
      "output_schema": [
        {
          "name": "visualizations",
          "type": "List[dict]",
          "description": "Generated charts and graphs data"
        }
      ],
      "tools_used": [
        "Visualization_Engine"
      ]
    },
    {
      "id": "generate_summary_and_insights",
      "function_name": "create_executive_summary",
      "description": "Aggregates all analysis results and generates an executive summary with insights and confidence score",
      "llm_actions": [
        "aggregate",
        "generate",
        "validate"
      ],
      "input_schema": [
        {
          "name": "trend_analysis",
          "type": "dict",
          "description": "Trend analysis results to summarize"
        },
        {
          "name": "comparative_analysis",
          "type": "dict",
          "description": "Market comparison results to summarize"
        },
        {
          "name": "anomalies_detected",
          "type": "List[dict]",
          "description": "Detected anomalies to include in summary"
        },
        {
          "name": "visualizations",
          "type": "List[dict]",
          "description": "Visualizations to reference or describe in summary"
        }
      ],
      "output_schema": [
        {
          "name": "insights",
          "type": "List[dict]",
          "description": "Key financial insights generated from analysis"
        },
        {
          "name": "executive_summary",
          "type": "str",
          "description": "Final executive summary with all key findings"
        },
        {
          "name": "confidence_score",
          "type": "float",
          "description": "Confidence score of the generated insights and summary"
        }
      ],
      
    },
    {
      "id": "__END__",
      "function_name": "end_node",
      "description": "End point of the workflow",
      "llm_actions": [],
      "input_schema": [],
      "output_schema": []
      
    }
  ],
  "edges": [
    {
      "source": "__START__",
      "target": "extract_financial_data",
      "conditional": false,
      "routing_conditions": ""
    },
    {
      "source": "extract_financial_data",
      "target": "validate_and_correct",
      "conditional": false,
      "routing_conditions": ""
    },
    {
      "source": "validate_and_correct",
      "target": "perform_trend_analysis",
      "conditional": false,
      "routing_conditions": ""
    },
    {
      "source": "validate_and_correct",
      "target": "cross_reference_market_data",
      "conditional": false,
      "routing_conditions": ""
    },
    {
      "source": "validate_and_correct",
      "target": "detect_anomalies",
      "conditional": false,
      "routing_conditions": ""
    },
    {
      "source": "perform_trend_analysis",
      "target": "create_visualizations",
      "conditional": false,
      "routing_conditions": ""
    },
    {
      "source": "cross_reference_market_data",
      "target": "create_visualizations",
      "conditional": false,
      "routing_conditions": ""
    },
    {
      "source": "detect_anomalies",
      "target": "create_visualizations",
      "conditional": false,
      "routing_conditions": ""
    },
    {
      "source": "create_visualizations",
      "target": "generate_summary_and_insights",
      "conditional": false,
      "routing_conditions": ""
    },
    {
      "source": "generate_summary_and_insights",
      "target": "__END__",
      "conditional": false,
      "routing_conditions": ""
    }
  ],
  "justification": "This workflow employs a Self-Correction architecture by first extracting and validating data, and then performing analysis in parallel to identify trends, compare with market data, and detect anomalies. The process is self-correcting as the `validate_and_correct` node modifies the `corrected_metrics` field based on a quality score, which is then used by subsequent nodes. The final `generate_summary_and_insights` node aggregates all the parallel results into a coherent summary, effectively serving as the final aggregation and validation step.
}
'''