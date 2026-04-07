# DataAnalystBot 🤖

**DataAnalystBot** is an interactive, AI-powered assistant designed to help users with all things data analysis. It leverages advanced retrieval-augmented generation (RAG) techniques, a custom vector database, and a conversational interface to provide expert guidance on data cleaning, visualization, statistics, machine learning, and popular tools like Python, SQL, Excel, and more.

---

## 🚀 Features

- **Conversational AI**: Chat with an LLM (Llama 3/4 via Groq) about any data analysis topic.
- **Multi-File Upload & Analysis**: Upload and analyze images (charts, screenshots), CSV/Excel files, and PDFs **simultaneously**. The bot uses all provided files as context for your question via the `/multi-upload` endpoint.
- **Data Cleaning & Analysis Endpoints**: Use `/analyze-data` for full AI-powered analysis (cleaning, stats, insights, visualizations) and `/clean-data` for fast, quota-free cleaning and summary.
- **SQL Query Generator**: Describe what you want in plain English, provide an optional schema or upload a CSV to auto-detect columns, and get a ready-to-run SQL query with a full explanation and optimization suggestions. Supports PostgreSQL, MySQL, SQLite, SQL Server, Oracle, BigQuery, and Snowflake.
- **Auto Data Augmentation**: Upload a CSV and let the bot automatically diagnose data quality issues — missing values, outliers, duplicates, skewed distributions, and class imbalance — then apply fixes in one click. Download the cleaned dataset or feed it directly into analysis.
- **Modern GUI**: Redesigned Streamlit interface with tabs for chat, data upload, SQL generation, and data augmentation, plus sidebar controls, recent chat management, and raw data preview.
- **Image Understanding**: Upload images and ask questions about them. The bot uses a multimodal LLM to analyze and respond, then grounds the answer using your chat history and knowledge base.
- **CSV Data Analysis**: Upload a CSV file and ask questions about its content. The bot uses the CSV content as context for the LLM, providing data-aware answers.
- **PDF Data Analysis**: Upload a PDF file and ask questions about its content. The bot extracts text from the PDF and uses it as context for the LLM, enabling document-aware responses.
- **File Caching**: Uploaded CSV, image, and PDF data are cached for each session, enabling fast, context-aware follow-up questions without re-uploading or re-processing files.
- **Image Upload Rate Limiting**: Each user can upload up to 3 images every 6 hours. If the limit is reached, only text, CSV, or PDF questions are allowed until the window resets.
- **Image Display in Chat**: Uploaded images are shown inline with your messages for easy reference.
- **Retrieval-Augmented Generation (RAG)**: Answers are grounded in a curated, chunked knowledge base from top data science sources.
- **Session Memory**: Each user session maintains its own chat history for context-aware conversations.
- **Recent Chats**: All conversations are saved and can be resumed from the sidebar.
- **Custom Vector Database**: Fast, semantic search over chunked documents using FAISS and HuggingFace embeddings.
- **Modern UI**: Built with Streamlit for a clean, interactive chat experience.
- **Extensible Scrapers**: Easily add new data sources with modular web scrapers.

---

## 📸 Screenshots

![Chat UI Example](https://github.com/user-attachments/assets/10652ec2-e53e-46a3-bb3c-13f2ccd7c34a)
*Chat with DataAnalystBot about Power BI for data analysis!*

---

## 🏗️ Architecture Overview
```mermaid
flowchart TD
    subgraph "👤 User Interface"
        A[👤 User] -->|📤 Uploads Files & Asks Questions| B[🖥️ Streamlit Web App]
    end

    subgraph "🔄 Processing Layer"
        B -->|📡 Sends Request| C[⚡ FastAPI Server]
        C -->|💾 Stores Uploads| J[📁 File Storage]
        C -->|🔍 Retrieves Context| E[🗄️ Vector Database]
        C -->|🧠 Generates Answer| D[🤖 AI Model - Groq]
        C -->|🛠️ Generates SQL| L[📝 SQL Generator]
        C -->|🔧 Cleans & Enriches| M[🧬 Data Augmentor]
    end

    subgraph "💾 Data Storage"
        E[🗄️ FAISS Vector Database]
        F[🔤 HuggingFace Embeddings]
        G[💭 Session Memory]
        I[⚡ Cache Storage]
        K[💬 Chat History]
        H[🕷️ Web Scrapers]
    end

    %% Data Flow
    E --> F
    H -->|📊 Adds Scraped Data| E
    C -->|💾 Saves Session| G
    C -->|⚡ Caches Results| I
    C -->|💬 Stores Chats| K
    L -->|✅ SQL + Explanation| C
    M -->|✅ Augmented CSV + Log| C

    %% Response Flow
    D -->|✅ AI Response| C
    C -->|📋 Final Answer| B
    B -->|📺 Shows Result| A

    class A,B userStyle
    class C,D,J,L,M processStyle
    class E,F,G,H,I,K storageStyle
```

---

## 🔧 Auto Data Augmentation

The augmentation pipeline runs in three stages — nothing is applied silently without the user reviewing and approving the plan first.

**Stage 1 — Diagnose**  
Scans the uploaded CSV and reports all detected issues with severity ratings:

| Issue | Detection Method |
|---|---|
| Missing values | Per-column null count + percentage |
| Duplicate rows | Exact row matching |
| Outliers | IQR (1.5× fence) per numeric column |
| Skewed distributions | Skewness > 1.0 on numeric columns with all-positive values |
| Class imbalance | Majority/minority ratio > 3:1 on categorical columns |
| Low row count | Dataset smaller than 100 rows |

**Stage 2 — Augment (user-controlled)**  
Each fix can be toggled on or off before applying:

| Option | Technique |
|---|---|
| Impute Missing Values | KNN imputation (≤10 numeric cols) or median; mode or "Unknown" for categorical |
| Treat Outliers | Winsorization — caps values at 1st/99th percentile |
| Remove Duplicates | Exact deduplication with reset index |
| Fix Skewed Distributions | log1p transform on skewed numeric columns |
| Generate Synthetic Rows | Gaussian noise (numeric) + frequency sampling (categorical) targeting 2× row count |

**Stage 3 — Review & Export**  
After augmentation the user sees before/after row counts, a full change log, a side-by-side data preview, a **Download Augmented CSV** button, and a one-click **Run Analysis on Augmented Data** option that feeds directly into the existing `/analyze-data` pipeline.

---

## 📚 Data Sources

- [GeeksforGeeks](https://geeksforgeeks.org)
- [TPointTech](https://tpointtech.com)
- [Towards Data Science](https://towardsdatascience.com)

All articles are scraped, chunked (500 chars), and stored in `data/data.jsonl` for efficient retrieval.

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend**: [FastAPI](https://fastapi.tiangolo.com/)
- **LLM**: [Groq Llama 3 & Multimodal Llama 4](https://groq.com/)
- **Vector DB**: [FAISS](https://github.com/facebookresearch/faiss)
- **Embeddings**: [HuggingFace Transformers](https://huggingface.co/)
- **Web Scraping**: [Selenium](https://selenium.dev/)
- **Data Augmentation**: [scikit-learn](https://scikit-learn.org/) + [SciPy](https://scipy.org/)
- **Session Memory**: In-memory per-session chat history
- **Caching**: DiskCache and Streamlit cache for fast file and context retrieval

---

## ⚡ Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/Lokesh-DataScience/Data-Analyst-Expert-Bot.git
cd DataAnalystBot
```

### 2. Install Dependencies
```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key
LANGSMITH_API_KEY=your_langsmith_api_key
```

### 4. Scrape and Prepare Data

Run the scrapers in the `scrapers/` folder to populate `data/data.jsonl` with chunked content:
```bash
python scrapers/gfg_scraper.py
python scrapers/pointtech_scraper.py
python scrapers/towardsdatascience_scrapper.py
```

### 5. Build the Vector Database
```bash
python vector_db/faiss_db.py
```

### 6. Start the Backend API
```bash
uvicorn api.main:app --reload
```

### 7. Launch the Streamlit Frontend
```bash
streamlit run streamlit_app/app.py
```

---

## 💬 Usage

- Open [http://localhost:8501](http://localhost:8501) in your browser.
- Ask questions about data analysis, tools, or techniques.
- **To analyze an image:** Upload a jpg, jpeg, or png file and enter your question. The bot will analyze the image and respond.
- **To analyze a CSV:** Upload a CSV file and ask a question about its content. The bot will use the CSV data as context for its answer.
- **To analyze a PDF:** Upload a PDF file and ask a question about its content. The bot will use the PDF text as context for its answer.
- **To generate a SQL query:** Go to the **🛠️ SQL Query Generator** tab. Paste your schema (or upload a CSV to auto-detect columns), select your database type and query type, describe what you want in plain English, and click **⚡ Generate SQL Query**. The bot returns a ready-to-run query, a plain English explanation, and optimization suggestions you can download as a `.sql` file.
- **To augment a dataset:** Go to the **🔧 Data Augmentation** tab. Upload a CSV, click **🔍 Diagnose Data** to see a full issues report and recommended fixes, toggle which steps to apply, then click **⚡ Apply Augmentation**. Review the change log and before/after preview, download the cleaned CSV, or click **📊 Run Analysis on Augmented Data** to analyze it immediately.
- **Note:** You can upload up to 3 images every 6 hours. If you reach the limit, you can still ask text questions.
- **Resume conversations:** Select any recent chat from the sidebar to continue where you left off.

---

## 🧩 Project Structure

```
DataAnalystBot/
│
├── api/                  # FastAPI backend
│   └── main.py
├── chains/               # RAG chain construction
│   └── rag_chain.py
├── data/                 # Chunked knowledge base (JSONL)
│   └── data.jsonl
├── loaders/              # Data loading utilities
│   ├── load_data.py
│   ├── load_csv.py
│   └── load_pdf.py
├── memory/               # Session memory management
│   └── session_memory.py
├── scrapers/             # Web scrapers for sources
│   ├── gfg_scraper.py
│   ├── pointtech_scraper.py
│   └── towardsdatascience_scrapper.py
├── streamlit_app/        # Streamlit UI
│   ├── components/
│   ├── config/
│   ├── styles/
│   ├── utils/
│   └── app.py
├── utils/                # Backend utilities
│   ├── data_analyzer.py
│   └── data_augmentor.py
├── vector_db/            # Vector DB creation/loading
│   └── faiss_db.py
├── requirements.txt
└── README.md
```

---

## 📝 Customization

- **Add new sources**: Write a new scraper in `scrapers/`, chunk the content, and append to `data/data.jsonl`.
- **Change chunk size**: Adjust the `textwrap.wrap(..., width=500)` in scrapers.
- **Swap LLM or embeddings**: Update model names in `chains/rag_chain.py` or `vector_db/faiss_db.py`.
- **Switch between full analysis and fast cleaning**: Use `/analyze-data` for AI-powered insights, or `/clean-data` for quick cleaning and stats.
- **Extend SQL generation**: The `/generate-sql` endpoint accepts any schema DDL and supports all major SQL dialects. Add dialect-specific prompt templates in `api/main.py` to further tailor output.
- **Extend augmentation**: Add new augmentation steps in `utils/data_augmentor.py` by adding a method and registering it in the `augment()` dispatcher. SMOTE-based oversampling can be enabled by installing `imbalanced-learn` and extending `_generate_synthetic_rows()`.

---

## 🛡️ Security & Privacy

- All chat history is stored in memory per session and is not persisted between server restarts.
- API keys are loaded from `.env` and never exposed to the frontend.
- Generated SQL queries are not executed server-side — the bot only returns query text, keeping your database safe.
- Data augmentation is performed entirely server-side in memory — uploaded CSVs are written to a temporary file, processed, and immediately deleted.

---

## 🤝 Contributing

Pull requests, issues, and feature suggestions are welcome!  
Please open an issue or submit a PR.

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain)
- [Groq](https://groq.com/)
- [HuggingFace](https://huggingface.co/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Streamlit](https://streamlit.io/)
- [Selenium](https://selenium.dev/)
- [scikit-learn](https://scikit-learn.org/)
- [SciPy](https://scipy.org/)

---

**Happy Analyzing!** 🚀