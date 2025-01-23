from fastapi import FastAPI
from langserve import add_routes
from agent.graph import graph
from dotenv import load_dotenv
import os


load_dotenv()

app = FastAPI(
  title="Candidate search",
  version="1.0",
  description="",
)

add_routes(
    app,
    graph,
    path="/search",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=os.getenv("HOST"), port=int(os.getenv("PORT")), reload=True)
