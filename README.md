# Trip Planner LLM

A FastAPI-based trip planning application that uses a local LLM (via Ollama) to generate detailed trip itineraries based on user requests.

## Features

- Natural language trip planning interface
- Beautiful, responsive UI for viewing trip plans
- Integration with Ollama for local LLM inference
- RESTful API endpoints
- Health check endpoint
- Docker-ready setup

## Project Structure

```
trip-planner-llm/
├── main.py                 # FastAPI application
├── templates/
│   └── index.html          # Beautiful trip plan UI
├── README.md               # This file
├── .gitignore              # Git ignore rules
└── requirements.txt        # Python dependencies
```

## Setup

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com/) installed and running
- A compatible LLM model (default: `qwen3:4b`)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd trip-planner-llm
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure Ollama is running and the model is available:
   ```bash
   ollama serve
   # In another terminal:
   ollama pull qwen3:4b
   ```

### Configuration

The application uses environment variables for configuration:

- `OLLAMA_URL`: URL of the Ollama API (default: `http://localhost:11434`)
- `MODEL_NAME`: Name of the model to use (default: `qwen3:4b`)

You can set these in a `.env` file or export them in your shell.

## Usage

### Running the Application

```bash
uvicorn main:app --reload
```

The API will be available at:
- http://127.0.0.1:8000

### API Endpoints

- `POST /v1/plan-trip`: Plan a trip
  - Request body: `{ "text": "your trip request here" }`
  - Returns: JSON with context, query, and the generated trip plan

- `GET /health`: Health check endpoint
  - Returns status of the application and Ollama connectivity

- `GET /`: Serves the beautiful trip plan UI

### Example Request

```bash
curl -X POST http://127.0.0.1:8000/v1/plan-trip \
  -H "Content-Type: application/json" \
  -d '{"text":"Plan a 5-day trip to Goa with budget 50000 rupees for 2 people"}'
```

## UI Features

The web interface provides:

- Health status monitoring (auto-refreshes every 30 seconds)
- Form validation for trip requests
- Beautiful trip plan display with:
  - Trip overview cards
  - Day-by-day itinerary with time-blocked activities
  - Visual budget breakdown with icons
  - Travel tips section
  - Responsive design for all device sizes

## Development

### Debugging in IntelliJ

If you experience debugger issues with the pydevd SyntaxError, try:

1. Use the terminal to run the application: `.venv/bin/uvicorn main:app --reload`
2. Or use "Attach to Process" in IntelliJ after starting the server in terminal

## Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add some amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a pull request

## Acknowledgments

- Built with FastAPI
- Uses Ollama for local LLM inference
- UI designed for clarity and beauty