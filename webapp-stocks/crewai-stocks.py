import json 
import os
from datetime import datetime

import yfinance as yf

from crewai import Agent, Crew, Process, Task

from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults

import streamlit as st

# YAHOO FINANCE 
def fetch_stock_price(ticket):
    stock = yf.download(ticket, start="2024-01-01", end="2024-10-09")
    return stock

yahoo_finance_tool = Tool(
    name="Yahoo Finance Tool",
    description="Fetches this year's {ticket} stock prices for a specific company from the Yahoo Finance API",
    func=lambda ticket: fetch_stock_price(ticket),
)

# OPENAI LLM - GPT
os.environ["OPENAI_API_KEY"] = st.secrets["OPEN_API_KEY"]
llm = ChatOpenAI(model="gpt-3.5-turbo")

# AGENT 1 - Stock Price Analyses
stockPriceAnalyst = Agent(
    role= "Senior Stock Price Analyst",
    goal="Find the {ticket} stock price and analyses trends",
    backstory="""You have a lot of experience analyzing the price of a specific stock 
    and making predictions about its future price.""",
    verbose=True,
    llm=llm,
    max_iter=5,
    memory=True,
    tools=[yahoo_finance_tool],
)

getStockPrice = Task(
    description="Analyze {ticket} stock price history and create uṕ, down or sideways trend analysis",
    expected_output="""Specify the current trend stock price - up, down or sideways. 
    eg. stock= 'APPL, price UP'""",
    agent=stockPriceAnalyst,
)

# DUCKDUCKGO
search_tool = DuckDuckGoSearchResults(backend="news", num_results=10)

# AGENT 2 - Stock News Analyses
newsAnalyst = Agent(
    role= "Stock News Analyst",
    goal="""Create a brief summary of market news related to the stock {ticket} company. 
    Specify the current trend - up, down or sideways with the news context. 
    For each requested stock asset, specify a number between 0 and 100, 
    where 0 is extreme fear and 100 is extreme positivity.""",
    backstory="""You have a lot of experience in analyzing market trends and news and have been following assets for over 10 years.

    You are also a master-level analyst in traditional markets and have in-depth knowledge of human psychology.

    You understand the news, its headlines and information, but you look at it with a healthy dose of skepticism. 
    You also consider the source of the news articles.""",
    verbose=True,
    llm=llm,
    max_iter=10,
    memory=True,
    tools=[search_tool],
    allow_delegation=False,
)

get_news = Task(
    description=f"""Take the stock and always include BTC in it (if not requested).
    Use the search tool to search for each one individually. 
    The current date is {datetime.now()}.
    Compose the results into a useful report""",
    expected_output="""A summary of the overall market and a one-sentence summary for each asset requested. 
    Include a fear/positivity score for each asset based on the news. Use format:
    <STOCK ASSET>
    <NEWS-BASED SUMMARY>
    <TREND FORECAST>
    <FEAR/POSITIVITY SCORE>""",
    agent=newsAnalyst,
)

# AGENT 3 - Stock Analyst Writer
stockAnalystWriter = Agent(
    role="Senior Stock Analyst Writer",
    goal="""Analyze price trends and news and write an insightful, attractive and informative 3-paragraph 
    newsletter based on the stock report and price trend.""",
    backstory="""You are widely accepted as the best stock analyst in the market. You understand complex concepts 
    and create compelling stories and narratives that resonate with broader audiences. 
    You understand macro factors and combine multiple theories - e.g. cycle theory and fundamental analysis. 
    You are capable of having multiple opinions when analyzing anything.""",
    verbose=True,
    llm=llm,
    max_iter=5,
    memory=True,
    allow_delegation=True,
)

writeAnalyses = Task(
    description="""Use stock price trend and stock news report to create analysis and write company newsletter about the {ticket} company, 
    this is brief and highlights the most important points.
    Focus on the stock price trend, news, and fear/positivity score. What are the considerations for the near future?
    Includes previous stock trend analysis and news summary.""",
    expected_output="""An eloquent 3-paragraph newsletter formatted as markdown in an easy-to-read manner. It must contain:
    - 3 bullet executive summary 
    - Introduction - set the big picture and increase interest
    - Main part provides the gist of the analysis, including news summary and fead/positivity scores
    - Summary - important facts and concrete prediction of future trends - up, down or sideways.""",
    agent=stockAnalystWriter,
    context=[getStockPrice, get_news],
)

# CREWAI

crew = Crew(
    agents=[stockPriceAnalyst, newsAnalyst, stockAnalystWriter],
    tasks=[getStockPrice, get_news, writeAnalyses],
    verbose=2,
    process=Process.hierarchical,
    full_output=True,
    share_crew=False,
    manager_llm=llm,
    max_iter=15
)

# UI
with st.sidebar:
    st.header("Enter the Stock to Research")
    with st.form(key="research_form"):
        topic = st.text_input("Select the ticket")
        submit_button = st.form_submit_button(label = "Run Research")
if submit_button:
    if not topic:
        st.error("Please fill the ticket field")
    else:
        results = crew.kickoff(inputs={"ticket": topic})
        st.subheader("Results of research:")
        st.write(results["final_output"])