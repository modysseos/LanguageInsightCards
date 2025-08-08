# 🃏 Language Insight Cards

A data-driven tool to analyze the rhetoric of public speeches, interviews, or any text without judgment. Instead of focusing on *what* is said, this engine surfaces *how* it's said, turning linguistic patterns into simple, interactive "insight cards."

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Features

- **Bilingual Analysis**: Seamlessly processes both **English** and **Greek** text with automatic language detection.
- **Dynamic UI**: The user interface and all AI-generated commentary can be displayed in either English or Greek.
- **Comprehensive Metrics**: Generates a rich set of metrics including coherence, confidence, lexical diversity, and more.
- **AI-Powered Commentary**: Utilizes **OpenAI's GPT-4o** to provide human-readable interpretations for each insight card.
- **Nuanced Lexicons**: Features expanded and nuanced lexicons for both languages to ensure high-accuracy analysis.

## Getting Started

Follow these instructions to set up and run the project locally.

### 1. Prerequisites

- Python 3.8+
- An OpenAI API Key

### 2. Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/modysseos/language-insight-cards.git
    cd language-insight-cards
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    # Windows
    python -m venv .venv
    .venv\Scripts\activate

    # macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### 3. API Key Configuration

This project requires an OpenAI API key to generate AI commentary. The key is loaded securely from a `.env` file.

1.  **Create a `.env` file** in the root of the project directory by copying the example file:
    ```bash
    # Windows
    copy .env.example .env

    # macOS/Linux
    cp .env.example .env
    ```

2.  **Open the `.env` file** and replace `"YOUR_API_KEY_HERE"` with your actual OpenAI API key.

### 4. Running the Application

Once the setup is complete, run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your web browser.

## How to Contribute

We welcome contributions from the community! Whether you're fixing a bug, improving the code, or expanding our language lexicons, your help is appreciated.

### Contributing Code

1.  **Fork the repository** and create a new branch for your feature or bug fix.
2.  Make your changes and ensure the code follows existing style conventions.
3.  Write a clear, concise commit message.
4.  **Submit a pull request** with a detailed description of your changes.

### Enriching the Lexicons

The accuracy of this tool depends heavily on the quality of its lexicons. If you are a native or fluent speaker of English or Greek, you can make a valuable contribution by improving the word lists in `lexicons.json`.

- **Add new terms**: Add words or phrases to the appropriate categories (`claims`, `hedges`, `specifics`, etc.).
- **Correct existing terms**: Fix any inaccuracies or add nuanced variations.
- **Expand to new languages**: We would love to add support for more languages! This would involve creating a new set of lexicon keys (e.g., `claims_fr`, `specifics_fr`) and adding the new language to the `UI_TEXT` dictionary in `app.py`.

When contributing to `lexicons.json`, please explain the reasoning for your changes in your pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## File Structure

-   `app.py`: The main Streamlit application script.
-   `requirements.txt`: A list of the required Python packages.
-   `lexicons.json`: A configurable file containing the bilingual word lists (English & Greek) used for the analysis. This can be expanded to support more languages.
-   `README.md`: This file.
