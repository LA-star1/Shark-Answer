"""Run the Shark Answer server."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "shark_answer.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
