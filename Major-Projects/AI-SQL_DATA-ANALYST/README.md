# 📊 AI SQL Data Analyst Agent

> An AI-powered SQL Data Analyst that lets users analyze CSV datasets using natural language. Upload a CSV file, ask questions in plain English, and the application automatically generates SQL queries, executes them on SQLite, visualizes the results, and provides AI-generated insights.

---

## ✨ Features

- 📁 Upload any CSV dataset
- 🤖 Ask questions in natural language
- 🧠 AI generates SQLite queries automatically
- ⚡ Executes SQL queries on uploaded data
- 📊 Displays query results in interactive tables
- 📈 Automatically generates charts
- 💡 AI-generated insights for every query
- 📥 Download query results as CSV
- 📋 View generated SQL query
- 🗑️ Clear chat history
- 📌 Dataset preview and statistics
- 💬 ChatGPT-inspired conversational interface
- ⚙️ Built using Groq LLM + LangChain

---

# 📸 Screenshots

Add your screenshots here.

```
assets/
└── screenshots/
    ├── home.png
    ├── upload_dataset.png
    ├── generated_sql.png
    ├── query_result.png
    ├── visualization.png
    └── ai_insight.png
```

---

# 🎥 Demo

You can also add a GIF demonstration.

```
assets/
└── demo.gif
```

---

# 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Backend |
| Streamlit | Web Application |
| SQLite | Database |
| Groq API | Large Language Model |
| LangChain | LLM Integration |
| Pandas | Data Processing |
| Matplotlib | Data Visualization |
| python-dotenv | Environment Variables |

---

# 📂 Project Structure

```text
AI_SQL_DATA_ANALYST/
│
├── app.py
├── data.db
├── requirements.txt
├── README.md
├── .env
│
└── assets/
    ├── screenshots/
    └── demo.gif
```

---

# 🚀 Installation Guide

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/AI_SQL_DATA_ANALYST.git

cd AI_SQL_DATA_ANALYST
```

---

## 2️⃣ Create Virtual Environment

### Windows

```bash
python -m venv myenv
```

Activate

```bash
myenv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv myenv
```

Activate

```bash
source myenv/bin/activate
```

---

## 3️⃣ Install Dependencies

Using requirements.txt

```bash
pip install -r requirements.txt
```

Or install manually

```bash
pip install streamlit
pip install pandas
pip install matplotlib
pip install langchain
pip install langchain-community
pip install langchain-groq
pip install python-dotenv
```

---

## 4️⃣ Get Groq API Key

1. Visit https://console.groq.com
2. Sign in
3. Create an API Key
4. Copy the API Key

---

## 5️⃣ Create `.env`

Create a file named

```
.env
```

Add

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## 6️⃣ Run the Project

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

# 💬 Example Questions

Try asking:

- Show all Fire-type Pokémon
- Top 10 strongest Pokémon
- Which Pokémon has the highest HP?
- Which Pokémon has the highest Speed?
- Show all Legendary Pokémon
- Count Pokémon in each Generation
- Average Attack by Type
- Top 5 fastest Pokémon
- Show Pokémon with HP greater than 100
- List Pokémon sorted by Defense
- Show Pokémon with the highest Special Attack
- Average HP of Legendary Pokémon
- Count Pokémon by Type
- Show Pokémon from Generation 3
- Which Type has the highest average Attack?

---

# 🔄 Workflow

```text
                Upload CSV
                     │
                     ▼
         Store Data in SQLite
                     │
                     ▼
        User asks a question
                     │
                     ▼
      Groq LLM generates SQL
                     │
                     ▼
        Execute SQL Query
                     │
                     ▼
          Display Results
                     │
                     ▼
      Generate Visualization
                     │
                     ▼
        Generate AI Insight
```

---

# 📊 Current Features

✅ CSV Upload

✅ SQLite Database Creation

✅ Natural Language to SQL

✅ AI SQL Generation

✅ SQL Execution

✅ Query Results

✅ Automatic Visualization

✅ AI Insights

✅ Download Result CSV

✅ Dataset Preview

✅ Dataset Statistics

✅ Chat Interface

✅ Clear Chat

---

# 📈 Future Improvements

- Smart chart selection (Bar, Pie, Scatter, Line)
- Plotly interactive charts
- Export analysis as PDF
- Multiple dataset support
- Query history
- Save chat history
- AI dashboard
- User authentication
- Cloud deployment
- Dark/Light mode
- Voice input
- Database connection support (MySQL/PostgreSQL)

---

# ⚙️ Requirements

```
Python 3.10+

Streamlit

Pandas

Matplotlib

SQLite3

LangChain

LangChain-Groq

python-dotenv
```

---

# 🤝 Contributing

Contributions are welcome.

Steps:

1. Fork the repository
2. Create a feature branch

```bash
git checkout -b feature-name
```

3. Commit changes

```bash
git commit -m "Added new feature"
```

4. Push changes

```bash
git push origin feature-name
```

5. Create a Pull Request

---

# 📝 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**RB**

**Artificial Intelligence & Data Science Engineer**

### Skills

- Python
- SQL
- Artificial Intelligence
- Machine Learning
- Data Analytics
- Streamlit
- LangChain
- Groq API

---

# 🌟 Support

If you found this project useful,

⭐ Star this repository

🍴 Fork the project

🛠️ Contribute to improve it

---

## Thank You ❤️

Thank you for checking out this project.

Happy Coding! 🚀
