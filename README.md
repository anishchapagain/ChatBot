# Data Analysis Chatbot

A modular chatbot application that converts natural language queries into pandas operations and provides comprehensive data analysis tools through a dashboard interface.

## Features

- **Natural Language to Pandas Query Conversion**: Ask questions about your data in plain English and get pandas code executed
- **Data Analysis Dashboard**: Visualize and explore your data through interactive charts and statistics
- **Local LLM Integration**: Uses Ollama to run language models locally for query conversion
- **Error Handling**: Robust error management with detailed logging
- **Modular Design**: Well-structured codebase with separate components for data loading, analysis, and UI

## Project Structure

```
data_analyzer/
├── app.py                  # Main application entry point
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
├── config/
│   └── config.yaml         # Configuration parameters
├── data/
│   └── sample_data.csv     # Sample dataset
├── modules/
│   ├── __init__.py
│   ├── data_loader.py      # Data loading functionality
│   ├── data_analyzer.py    # Data analysis functionality
│   ├── llm_manager.py      # LLM selection and management
│   ├── query_converter.py  # Convert text to pandas queries
│   └── error_handler.py    # Error handling utilities
├── ui/
│   ├── __init__.py
│   ├── chatbot.py          # Chat interface
│   └── dashboard.py        # Data visualization dashboard
└── tests/
    ├── __init__.py
    ├── test_data_loader.py
    ├── test_query_converter.py
    └── test_llm_manager.py
```

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/data-analyzer-chatbot.git
cd data-analyzer-chatbot
```

2. Create and activate a virtual environment (optional but recommended):
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Install Ollama:
Follow the instructions at [Ollama's official website](https://ollama.ai/) to install Ollama on your system.

5. Pull the necessary models:
```
ollama pull llama2
```

## Usage

1. Start the Ollama server:
```
ollama serve
```

2. Run the application:
```
streamlit run app.py
```

3. Open your browser and navigate to http://localhost:8501

## Using the Application

### Chat Interface
1. Upload your data file (CSV, Excel, or JSON)
2. Type your query in natural language (e.g., "Show me the average sales by region")
3. View the results and the generated pandas code

### Dashboard
1. Navigate to the "Analysis Dashboard" tab
2. Choose from different analysis types:
   - Data Overview
   - Summary Statistics
   - Distribution Analysis
   - Correlation Analysis
   - Time Series Analysis
   - Custom Analysis
3. Interact with the visualizations and analysis tools

## Example Queries

Here are some example queries you can try:
- "What is the total sales for each product category?"
- "Show me the trend of profit over time"
- "Calculate the average cost by region and sort in descending order"
- "Find the correlation between sales and profit"
- "Show me the top 5 dates with highest sales"
- "Group by customer type and count the number of transactions"

## Configuration

You can modify the application's behavior by editing the `config/config.yaml` file:
- Data settings: default paths, allowed formats
- LLM settings: model, temperature, token limits
- UI settings: theme, chart heights, sidebar width
- Error handling: logging configuration

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
