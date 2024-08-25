from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import json
import tiktoken
import numpy as np
from typing import List, Dict
import logging

app = FastAPI()

@app.middleware("http")
async def increase_payload_size(request: Request, call_next):
    if request.method == "POST":
        payload_size = int(request.headers.get("content-length", 0))
        max_size = 100 * 1024 * 1024  # 100MB
        if payload_size > max_size:
            return JSONResponse(status_code=413, content={"detail": "Payload too large"})
    response = await call_next(request)
    return response

# CORSミドルウェアを追加
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","http://localhost:3001","https://ft-trainingfile-creator.vercel.app","https://ft-trainingfile-creator-git-main-rinteringotetos-projects.vercel.app"],
    # allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # すべてのHTTPメソッドを許可
    allow_headers=["*"],  # すべてのヘッダーを許可
)

encoding = tiktoken.get_encoding("cl100k_base")

class InputData(BaseModel):
    jsonl_data: str

def num_tokens_from_messages(messages: List[Dict], tokens_per_message: int = 3, tokens_per_name: int = 1) -> int:
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(str(value)))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens

def num_assistant_tokens_from_messages(messages: List[Dict]) -> int:
    return sum(len(encoding.encode(message["content"])) 
               for message in messages if message["role"] == "assistant")

def get_distribution(values: List[float], name: str) -> Dict:
    return {
        "name": name,
        "min": min(values),
        "max": max(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p5": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95))
    }

@app.post("/token-analysis")
async def token_analysis(input_data: InputData):
    try:
        # Split the input string into lines and parse each line as JSON
        dataset = [json.loads(line) for line in input_data.jsonl_data.strip().split('\n')]
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSONL data")

    num_tokens_per_conversation = []
    num_assistant_tokens_per_conversation = []
    
    for i, conv in enumerate(dataset):
        total_tokens = num_tokens_from_messages(conv["messages"])
        assistant_tokens = num_assistant_tokens_from_messages(conv["messages"])
        num_tokens_per_conversation.append(total_tokens)
        num_assistant_tokens_per_conversation.append(assistant_tokens)
        
        # 各メッセージのトークン数を詳細に出力
        for j, message in enumerate(conv["messages"]):
            message_tokens = len(encoding.encode(message["content"]))

    token_distribution = get_distribution(num_tokens_per_conversation, "number of tokens per conversation")
    assistant_token_distribution = get_distribution(num_assistant_tokens_per_conversation, "number of assistant tokens per conversation")

    threshold = 65536
    over_threshold_content = [
        message["content"]
        for conv, num_tokens in zip(dataset, num_tokens_per_conversation)
        if num_tokens > threshold
        for message in conv["messages"]
        if message["role"] == "user"
    ]

    response_data = {
        "token_distribution": {
            **token_distribution,
            "individual_counts": num_tokens_per_conversation,
            "total": sum(num_tokens_per_conversation)
        },
        "assistant_token_distribution": assistant_token_distribution,
        "over_threshold_content": over_threshold_content
    }
    return response_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)