
import os
import gradio as gr
import numpy as np
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.globals import set_verbose
from sklearn.metrics.pairwise import cosine_similarity

from config import OPENAI_API_KEY
from data import df, tfidf, desc_matrix
from tools import optimize_price_with_lp

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
set_verbose(False)
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

custom_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Real-Time Pricing Optimization Agent. Use tools to decide the best price for a product to maximize revenue while considering demand elasticity, competitor pricing, margins, and stock constraints."),
    ("human", "{input}"),
    ("ai", "{agent_scratchpad}")
])

tools = [optimize_price_with_lp]
agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=custom_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

def chatbot_response(message, history):
    query = message.strip()

    if query in df["item_id"].values:
        product = df[df["item_id"] == query].iloc[0]
    else:
        query_vec = tfidf.transform([query])
        similarity = cosine_similarity(query_vec, desc_matrix).flatten()
        top_match_idx = np.argmax(similarity)
        product = df.iloc[top_match_idx]

    prompt_input = f"""
Use LP optimization to recommend the best price for this product:
{{
    "item_id": "{product.item_id}",
    "cost_price": {product.cost_price},
    "current_price": {product.current_price},
    "competitor_prices": {product.competitor_prices},
    "target_margin_percent": {product.target_margin_percent},
    "stock_level": {product.stock_level},
    "hourly_sales": {product.hourly_sales},
    "price_elasticity": {product.price_elasticity}
}}
Respond with Current Price , Recommended Price and Revenue Improvement.
"""
    try:
        response = agent_executor.invoke({
            "input": prompt_input,
            "agent_scratchpad": ""
        })
        return response["output"]
    except Exception as e:
        return f"Error running optimization: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("### 💰 Real-Time Price Recommendation Agent (Black Friday Edition)")
    chatbot = gr.ChatInterface(fn=chatbot_response)
    gr.Markdown("📌 Type a product `item_id` or description to get a recommendation.")

demo.launch()
