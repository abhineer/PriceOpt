**Pricing Optimization Agent**

A real-time pricing optimization agent built using LangChain, OpenAI GPT-4o, and Gradio.
This agent recommends the optimal price for a product to maximize revenue, considering factors like demand elasticity, competitor pricing, profit margins, and stock constraints.

🚀 Features

AI-powered pricing using GPT-4o via LangChain’s create_openai_functions_agent

Linear programming optimization through a custom tool (optimize_price_with_lp)

Interactive Gradio UI for easy experimentation

Cosine similarity based product matching using TF-IDF vectors

Modular structure with config.py, data.py, and tools.py

🗂️ Project Structure
pricing-agent/
│
├── app.py                 # Main entry point with LangChain agent + Gradio interface
├── config.py              # Stores OpenAI API key and environment variables
├── data.py                # Product dataset, TF-IDF model, and embeddings
├── tools.py               # Optimization tools (e.g., linear programming)
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation

⚙️ Setup Instructions
1️⃣ Clone the Repository
git clone https://github.com/<your-username>/pricing-agent.git
cd pricing-agent

2️⃣ Create and Activate Virtual Environment
python3 -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Add Your OpenAI API Key

Edit config.py:

OPENAI_API_KEY = "your-openai-api-key"

▶️ Run the App
python app.py


Then open the Gradio UI link printed in the console (usually http://127.0.0.1:7860).

🧩 Key Components

LangChain Agent — Handles reasoning and uses the optimization tool.

Optimization Tool — Applies linear programming to compute price suggestions.

Data Layer — Provides TF-IDF matrices for product similarity and context.

LLM (GPT-4o) — Interprets prompts, invokes tools, and generates recommendations.
