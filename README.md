Candlelight

Candlelight is a transformer-based deep learning model designed for analyzing stock market candlestick patterns and predicting potential trends.

Features
Candlestick Tokenization: Converts OHLC data into tokenized formats for model consumption.
Transformer Architecture: Utilizes a multi-headed attention mechanism for sequential data analysis.
Custom Dataset Pipeline: Processes and loads candlestick data seamlessly.
GPU Acceleration: Optimized for training on NVIDIA GPUs.
Training and Testing Scripts: Includes pre-built scripts for training and evaluating the model.
Requirements
Python 3.8+
Libraries:
torch
pandas
numpy
alpha_vantage
yfinance
dotenv
Install the required packages using:


pip install -r requirements.txt
Directory Structure
plaintext
Copy code
candlelight/
├── data/                   # Raw and processed data
├── notebooks/              # Jupyter notebooks for experimentation
├── scripts/                # Scripts for running the project
│   ├── download_yfinance.py
│   ├── gpu_test.py
│   ├── test_embedding.py
│   ├── train_transformer.py
├── src/                    # Source files for the project
│   ├── data/               # Data loader
│   │   ├── data_loader.py
│   ├── models/             # Model components
│   │   ├── embedding.py
│   │   ├── transformer_model.py
├── tests/                  # Test cases
├── venv/                   # Virtual environment
├── .env                    # API keys and environment variables
├── .gitignore              # Ignored files and folders
├── README.md               # Project overview
└── requirements.txt        # List of dependencies
Usage
1. Download Data
Use the download_yfinance.py script to fetch historical stock data:

bash
Copy code
python scripts/download_yfinance.py
2. Train the Model
Train the transformer model with the train_transformer.py script:

bash
Copy code
python scripts/train_transformer.py
3. Test Components
Run the test scripts in the scripts/ directory to verify individual components:

python scripts/test_embedding.py
Environment Variables
The .env file should include:


ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
Add your API key before running the scripts.

To-Do
Enhance the model evaluation metrics.
Add visualization scripts for predictions.
Experiment with different transformer architectures.