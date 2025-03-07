# Product Advertisement Bot

A customizable marketing assistant that answers queries about products in an engaging, advertisement-style format using RAG (Retrieval Augmented Generation) technology.

## 🌟 Features

- **Product-Specific Marketing**: Customizable for different brands and products
- **RAG Technology**: Retrieves relevant product information from documentation
- **Persuasive Copy**: Generates engaging, marketing-focused responses
- **Conversation Memory**: Maintains context throughout customer interactions
- **Brand Voice Control**: Adjustable tone, formality, and style settings
- **Call-to-Action**: Automatically includes website links and CTAs

## 📋 Requirements

- Python 3.8+
- OpenAI API key
- PDF product documentation
- Required libraries (see requirements.txt)

## 🔧 Installation

1. Clone or download this repository
2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## ⚙️ Setup

1. Create a `.env` file in the project directory with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

2. Add your product PDF documentation to the project directory
   
3. Generate embeddings for your product information:
   ```
   python generate_embeddings.py
   ```
   Note: Make sure to update the PDF file paths in `generate_embeddings.py` to match your documentation.

## 🚀 Usage

### Basic Usage

Run the main bot with:
