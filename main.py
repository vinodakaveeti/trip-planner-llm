import copy
import os
import json
import httpx
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

class TripPlannerRequest(BaseModel):
    text: str = Field(min_length=10, max_length=5000)

class TripPlannerResponse(BaseModel):
    context: str
    query: str
    message: dict


app = FastAPI(title="Trip Planner using ollama model")

# Configure templates
templates = Jinja2Templates(directory="templates")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_URL= os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3:4b")

logger.info(f"Ollama URL: {OLLAMA_URL}")
logger.info(f"Model name: {MODEL_NAME}")

SYSTEM_PROMPT = """You are a concise travel planner. Generate a practical itinerary using the user's destination, duration, budget, and number of travellers.

                    Rules:
                    - Generate exactly one entry per travel day.
                    - Include morning, afternoon, and evening activities for every day.
                    - Keep each activity to one short sentence.
                    - Include a compact budget breakdown and exactly 3 practical tips.
                    - Keep the total within the requested budget where possible.
                    - State reasonable assumptions for missing information.
                    - Return only valid JSON matching the provided schema.
                    - Do not include introductions, conclusions, or extra commentary."""

TRIP_SCHEMA = {
    "type": "object",
    "properties": {
        "destination": {"type": "string"},
        "duration_days": {"type": "integer"},
        "budget_inr": {"type": "number"},
        "travellers": {"type": "integer"},
        "itinerary": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "day": {"type": "integer"},
                    "title": {"type": "string"},
                    "morning": {"type": "string"},
                    "afternoon": {"type": "string"},
                    "evening": {"type": "string"},
                    "estimated_cost_inr": {"type": "number"}
                },
                "required": [
                    "day", "title", "morning",
                    "afternoon", "evening",
                    "estimated_cost_inr"
                ]
            }
        },
        "budget_breakdown": {
            "type": "object",
            "properties": {
                "travel": {"type": "number"},
                "accommodation": {"type": "number"},
                "food": {"type": "number"},
                "local_transport": {"type": "number"},
                "activities": {"type": "number"},
                "miscellaneous": {"type": "number"},
                "total": {"type": "number"}
            },
            "required": [
                "travel", "accommodation", "food",
                "local_transport", "activities",
                "miscellaneous", "total"
            ]
        },
        "tips": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": [
        "destination", "duration_days", "budget_inr",
        "travellers", "itinerary", "budget_breakdown", "tips"
    ]
}


@app.post("/v1/plan-trip", response_model=TripPlannerResponse)
async def plan_trip(request: TripPlannerRequest):
    logger.info(f"Received trip planning request: {request.text}")

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT

            },
            {
                "role": "user",
                "content": request.text
            }
        ],
        "format": TRIP_SCHEMA,
        "stream": False,
        "think": False,
        "keep_alive": "30m",
        "options": {
            "num_predict": 2000,
            "temperature": 0.1,
        }
    }

    try:
        logger.info(f"Sending request to Ollama at {OLLAMA_URL} and request payload is {payload}")
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
            response.raise_for_status()
            result = response.json()
            logger.info(f"Received response from Ollama: {result}...")

    except httpx.TimeoutException:
        logger.error("Ollama request timed out after 90 seconds")
        raise HTTPException(
            status_code=504,
            detail="The LLM request timed out. The model may be taking too long to respond or the service may be unavailable."
        )
    except httpx.ConnectError as e:
        logger.error(f"Failed to connect to Ollama service: {str(e)}")
        raise HTTPException(
            status_code=502,
            detail=f"Unable to connect to the LLM service at {OLLAMA_URL}. Please ensure Ollama is running and accessible."
        )
    except httpx.HTTPError as e:
        logger.error(f"HTTP error from Ollama service: {str(e)}")
        raise HTTPException(
            status_code=502,
            detail=f"Unable to communicate with the LLM service: {str(e)}"
        )
    except (ValueError, KeyError) as e:
        logger.error(f"Invalid response from Ollama: {str(e)}")
        raise HTTPException(
            status_code=502,
            detail="The LLM returned an invalid response."
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )

    # Extract the assistant's message from Ollama response
    # Ollama typically returns: {"model": "...", "created_at": "...", "message": {"role": "assistant", "content": "..."}}
    assistant_message = ""
    if isinstance(result, dict):
        # Try to extract the message content
        message_data = result.get("message", {})
        if not assistant_message and "content" in message_data:
            assistant_message = json.loads(message_data["content"])
    else:
        assistant_message = str(result)

    # calculate the total budget. LLM was not giving proper sum
    budget_breakdown_values = copy.deepcopy(assistant_message['budget_breakdown'])
    budget_breakdown_values.pop("total")
    assistant_message['budget_breakdown']["total"] = sum(list(budget_breakdown_values.values()))
    assistant_message['budget_inr'] = sum(list(budget_breakdown_values.values()))

    logger.info(f"Generated response of length: {len(assistant_message)}")

    return TripPlannerResponse(
        context=f"Processed request using {MODEL_NAME} model",
        query=request.text,
        message=assistant_message
    )


@app.get("/health")
async def health_check():
    """Health check endpoint to verify Ollama connectivity"""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{OLLAMA_URL}/api/tags")
            response.raise_for_status()
            ollama_status = "connected"
            models = response.json().get("models", [])
            model_names = [m.get("name", "unknown") for m in models]
            model_available = MODEL_NAME in model_names

            return {
                "status": "healthy",
                "ollama_status": ollama_status,
                "ollama_url": OLLAMA_URL,
                "configured_model": MODEL_NAME,
                "model_available": model_available,
                "available_models": model_names[:5]  # First 5 models
            }
    except httpx.ConnectError:
        return {
            "status": "unhealthy",
            "ollama_status": "disconnected",
            "error": f"Cannot connect to Ollama at {OLLAMA_URL}",
            "ollama_url": OLLAMA_URL,
            "configured_model": MODEL_NAME
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "ollama_status": "error",
            "error": str(e),
            "ollama_url": OLLAMA_URL,
            "configured_model": MODEL_NAME
        }


@app.get("/")
async def read_root(request: Request):
    """Serve the main UI page"""
    return templates.TemplateResponse(request,"index.html")


if __name__ == "__main__":
    # This is for development only - use uvicorn for production
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)