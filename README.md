# Nike Marketing Assistant - RAG-based Chat Application

## Project Overview

This application is an AI-powered marketing assistant built specifically for Nike products. It uses Retrieval Augmented Generation (RAG) to provide informative and persuasive responses about Nike's athletic footwear and apparel. The system leverages domain-specific knowledge stored in a vector database along with contextual conversation history to generate high-quality marketing-focused content.

## Architecture

### Components

1. **Vector Store (Chroma DB)**
   - Contains embedded product information, marketing materials, and knowledge about Nike
   - Uses sentence-transformers/all-MiniLM-L6-v2 for generating embeddings
   - Provides semantic search capabilities to retrieve relevant context for queries

2. **LLM Integration (OpenAI)**
   - Utilizes GPT-4o-mini model from OpenAI
   - Configured with temperature=0.7 for creative yet controlled responses
   - Limited to 300 tokens for concise marketing content

3. **RAG Pipeline**
   - Query processing → Context retrieval → LLM prompt construction → Response generation
   - Includes conversation history to maintain context across multiple interactions
   - Uses custom marketing-focused prompt templates

4. **FastAPI Server**
   - RESTful API endpoints for chat interactions
   - Session management for multiple concurrent users
   - Health check and monitoring endpoints

### Data Flow

```
User Query → API → Vector DB Lookup → Context Retrieval → 
Prompt Construction (with history) → LLM Processing → Response → User
```

## Features

### Marketing Focus

- **Brand-Specific Voice**: Maintains Nike's inspirational and motivational tone
- **Target Audience Awareness**: Tailored to athletes and fitness enthusiasts
- **Key Selling Points**: Highlights innovative technology, performance, design, athlete endorsements, and materials
- **Controlled Response**: Concise, under 100-word responses that emphasize benefits over promotion

### Conversation Management

- Maintains context for up to 4 previous exchanges
- Separate conversation histories for different users/sessions
- Ability to clear session history and start fresh

### Technical Capabilities

- **Vector Search**: Retrieves the 6 most relevant documents for each query
- **Stateless API Design**: All state maintained through session IDs
- **Scalable**: Can handle multiple simultaneous users with isolated conversation contexts

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Process a user query and return a marketing response |
| `/session` | POST | Create a new conversation session |
| `/session/{session_id}` | DELETE | Clear history for a specific session |
| `/health` | GET | Check service health status |

### Request/Response Examples

**Create Session:**
```
POST /session
Response: {"session_id": "550e8400-e29b-41d4-a716-446655440000", "message": "Session created successfully"}
```

**Chat:**
```
POST /chat
{
    "query": "Are Nike shoes good for running marathons?",
    "session_id": "550e8400-e29b-41d4-a716-446655440000"
}

Response:
{
    "response": "Nike's running shoes are engineered specifically for marathon performance with responsive cushioning and lightweight design. Models like Nike Air Zoom Alphafly NEXT% provide exceptional energy return for those long distances, helping you maintain pace when it matters most. Shop now at www.nike.com.",
    "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

## Setup and Installation

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd ad_thingy
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

4. Prepare your vector database:
   - Ensure the `chroma_db_nccn` directory exists with your embedded documents
   - Or create a new vector database with Nike product information

### Running the Application

#### Console Version
```
python rag.py
```

#### API Server
```
python api.py
```
Or with uvicorn directly:
```
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

## Customization Opportunities

The system is designed to be easily adaptable for different products by modifying:

1. **Product Configuration**: Edit the `selling_product_info` dictionary in `rag.py`
2. **Marketing Style**: Adjust the `marketing_style` dictionary to change tone and approach
3. **Vector Database**: Replace the content in the Chroma DB with information about different products
4. **Prompt Template**: Modify the template to emphasize different aspects of marketing

## Performance Considerations

- The system makes API calls to OpenAI for each request, so pricing will scale with usage
- Vector search is performed locally using the Chroma DB
- Consider implementing caching for frequent queries to reduce API costs

## Future Enhancements

- Image generation capabilities for visual marketing content
- A/B testing functionality to compare different marketing approaches
- Analytics dashboard to track user engagement and query patterns
- Integration with Nike's product catalog API for real-time product information
- Support for multimodal queries (text + image)

## Development and Testing

- API documentation is available at `/docs` when running the FastAPI server
- Use the interactive Swagger UI to test endpoints during development
- The console version (`rag.py`) is useful for debugging prompt construction and response quality
