import os
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools

# Load environment variables
load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL")

print(f"Using API Key starts with: {API_KEY[:10]}")
print(f"Base URL: {BASE_URL}")

# Limit max_tokens to avoid 402 errors
MAX_TOKENS = 10000

# Web agent
web_agent = Agent(
    name="Web Agent",
    model=OpenAIChat(
        id="gpt-4o",
        api_key=API_KEY,
        base_url=BASE_URL,
        max_tokens=MAX_TOKENS
    ),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True
)

# Finance agent
finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=OpenAIChat(
        id="gpt-4o",
        api_key=API_KEY,
        base_url=BASE_URL,
        max_tokens=MAX_TOKENS
    ),
    tools=[YFinanceTools(
        stock_price=True,
        analyst_recommendations=True,
        company_info=True
    )],
    instructions=["Use tables to display data"],
    show_tool_calls=True,
    markdown=True
)

forecasting_agent = Agent(
    name="Forecasting Agent",
    role="Predict future stock or mutual fund prices",
    model=OpenAIChat(
        id="gpt-4o",
        api_key=API_KEY,
        base_url=BASE_URL,
        max_tokens=MAX_TOKENS
    ),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
    instructions=[
        "Provide future trend predictions with confidence levels",
        "Use tables to display forecast results"
    ],
    markdown=True
)

# Team of agents
agent_team = Agent(
    model=OpenAIChat(
        id="gpt-4o",
        api_key=API_KEY,
        base_url=BASE_URL,
        max_tokens=MAX_TOKENS
    ),
    team=[web_agent, finance_agent, forecasting_agent],
    instructions=["Always include sources", "Use tables to display data","Use future trend predictions with confidence levels"],
    show_tool_calls=True,
    markdown=True
)

# Run your query
agent_team.print_response(
    "Summarize analyst recommendations and share the latest news for ETH ",
    stream=True
)
