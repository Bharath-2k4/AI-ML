📊 AI SQL Data Analyst Agent

An AI-powered SQL Data Analyst built with Streamlit, Groq LLM, SQLite, and Python that allows users to analyze CSV datasets using natural language. Simply upload a CSV file, ask questions in plain English, and the application automatically generates SQL queries, executes them, visualizes the results, and provides AI-generated insights.

🚀 Features
📁 Upload any CSV dataset
🤖 Ask questions using natural language
🧠 AI automatically generates SQLite queries
⚡ Executes SQL queries on uploaded data
📊 Displays query results in an interactive table
📈 Automatically generates visualizations
💡 AI-generated insights for query results
📥 Download query results as CSV
🗑️ Clear chat history
📋 View generated SQL query
📌 Sidebar with dataset statistics and preview
💬 ChatGPT-inspired conversational interface
🖥️ Demo
Home Interface
Upload CSV
Ask questions naturally
AI generates SQL
View results
Download results

Example Questions:

Show the top 10 strongest Pokémon
Which Pokémon has the highest HP?
List all Legendary Pokémon
Average Attack by Type
Show all Fire type Pokémon
Top 5 fastest Pokémon
Count Pokémon in each Generation
🛠️ Tech Stack
Technology	Purpose
Python	Backend
Streamlit	Web Application
Groq API	Large Language Model
SQLite	Query Execution
Pandas	Data Processing
Matplotlib	Data Visualization
LangChain	LLM Integration
python-dotenv	Environment Variables
📂 Project Structure
AI_SQL_DATA_ANALYST/
│
├── app.py
├── data.db
├── .env
├── requirements.txt
├── README.md
│
└── assets/
    ├── screenshots/
    └── demo.gif
⚙️ Installation
1. Clone Repository
git clone https://github.com/yourusername/AI_SQL_DATA_ANALYST.git

cd AI_SQL_DATA_ANALYST
2. Create Virtual Environment
python -m venv myenv

Activate Environment

Windows

myenv\Scripts\activate

Mac/Linux

source myenv/bin/activate
3. Install Dependencies
pip install -r requirements.txt

or

pip install streamlit pandas matplotlib langchain langchain-groq python-dotenv
4. Create .env

Create a file named:

.env

Add your Groq API Key

GROQ_API_KEY=your_groq_api_key_here
5. Run Application
streamlit run app.py
💬 Example Queries
Show all Fire type Pokémon

Top 10 Pokémon by HP

Average Attack for each Type

Legendary Pokémon

Top 5 fastest Pokémon

Which Pokémon has the highest Attack?

Count Pokémon in each Generation

Show Pokémon with HP greater than 100
📈 Workflow
Upload CSV
      │
      ▼
Store Data in SQLite
      │
      ▼
Ask Question
      │
      ▼
Groq LLM Generates SQL
      │
      ▼
Execute SQL
      │
      ▼
Display Results
      │
      ▼
Generate Visualization
      │
      ▼
Generate AI Insight
📸 Screenshots

Add screenshots inside:

assets/screenshots/

Example:

Home Page

Chat Interface

Generated SQL

Results

Visualization

AI Insight
🔮 Future Improvements
📊 Smart chart selection (Bar, Pie, Scatter, Line)
📄 Export analysis as PDF
📈 Interactive Plotly visualizations
📂 Multiple dataset support
💾 Save chat history
🧮 Advanced SQL optimization
🌐 Cloud deployment
🔍 Dataset profiling
📉 Dashboard analytics
🤝 Contributing

Contributions, issues, and feature requests are welcome.

Feel free to fork the repository and submit a pull request.

📜 License

This project is licensed under the MIT License.

👨‍💻 Author

RB

🎓 B.E. Artificial Intelligence & Data Science
💻 Python | SQL | AI | Data Analytics | Machine Learning
🚀 Passionate about building AI-powered applications

⭐ If you found this project helpful, consider giving it a star on GitHub!
