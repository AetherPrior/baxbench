import os
import uuid
import time
import asyncio
from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor
from openai_harmony import load_harmony_encoding, HarmonyEncodingName, Conversation, Message, Role, SystemContent, DeveloperContent
from transformers import AutoTokenizer
# 1) Load encoding for gpt-oss
encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# vLLM imports
from vllm import LLM
from vllm.sampling_params import SamplingParams
import multiprocessing


# ── Reproducibility settings ──
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"   # disable vLLM multiprocessing for determinism
SEED = 42
TENSOR_PARALLEL_SIZE = 2  # or as your GPU setup

def create_app(llm: LLM) -> FastAPI:
    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-120b", cache_dir="/space1/asura/hf_home")
    app = FastAPI()


    class Message(BaseModel):
        role: str
        content: Union[str, List[Dict[str, Any]]]

    class CompletionRequest(BaseModel):
        model: str
        prompt: Optional[List[str]] = None
        max_tokens: int = Field(default=256, alias="max_tokens")
        temperature: float = 1.0
        top_p: float = 1.0
        n: int = 1
        stop: Optional[List[str]] = None
        extra_body: Optional[Dict[str, Any]] = None
        reasoning_effort: Optional[str] = None

    class Choice(BaseModel):
        text: str
        index: int
        finish_reason: Optional[str] = None

    class CompletionResponse(BaseModel):
        id: str
        object: str = "text_completion"
        created: int
        model: str
        choices: List[Choice]
        reasoning_effort: Optional[str] = None

    class ChatCompletionRequest(BaseModel):
        model: str
        messages: List[Message]
        max_tokens: int = Field(default=256, alias="max_tokens")
        temperature: float = 1.0
        top_p: float = 1.0
        n: int = 1
        stop: Optional[List[str]] = None
        extra_body: Optional[Dict[str, Any]] = None
        reasoning_effort: Optional[str] = None

    class ChatChoice(BaseModel):
        message: Message
        index: int
        finish_reason: Optional[str] = None

    class ChatCompletionResponse(BaseModel):
        id: str
        object: str = "chat.completion"
        created: int
        model: str
        choices: List[ChatChoice]
        reasoning_effort: Optional[str] = None

    class ResponseRequest(BaseModel):
        model: str
        input: List[Message]
        max_output_tokens: int = Field(default=256, alias="max_output_tokens")
        temperature: float = 1.0
        top_p: float = 1.0
        extra_body: Optional[Dict[str, Any]] = None
        reasoning_effort: Optional[str] = None

    class ResponseOutput(BaseModel):
        text: str

    class ResponseResponse(BaseModel):
        id: str
        object: str = "response"
        created: int
        model: str
        output: ResponseOutput
        reasoning_effort: Optional[str] = None

    class ModelsResponse(BaseModel):
        object: str = "list"
        data: List[Dict[str, Any]]

    ### Internal batching/queueing infrastructure

    _BatchEntry = Dict[str, Any]
    incoming_queue: asyncio.Queue[_BatchEntry] = asyncio.Queue()
    executor = ThreadPoolExecutor(max_workers=1)  # you may increase if your hardware allows

    async def batch_dispatcher(batch_size: int = 16, dispatch_interval: float = 1):
        while True:
            await asyncio.sleep(dispatch_interval)
            batch: List[_BatchEntry] = []
            try:
                entry = incoming_queue.get_nowait()
                batch.append(entry)
            except asyncio.QueueEmpty:
                continue
            for _ in range(batch_size - 1):
                try:
                    entry = incoming_queue.get_nowait()
                    batch.append(entry)
                except asyncio.QueueEmpty:
                    break

            # Prepare inputs lists
            prompts = []
            sampling_params_list = []
            modes = []
            for e in batch:
                if e["mode"] == "generate":
                    prompts.append(e["prompt"])
                if e["mode"] == "chat":
                    # convert to the format expected by vLLM
                    chat_prompt = []
                    for msg in e["conversation"]:
                        chat_prompt.append({"role": msg.role, "content": msg.content})
                    prompts.append(chat_prompt)
                sampling_params_list.append(e["sampling_params"])
                modes.append(e["mode"])

            print(f"Dispatching batch of size {len(batch)}")
            # Use run_in_executor to avoid blocking event loop
            def run_batch():
                outputs = llm.generate(prompts, sampling_params_list)
                return outputs
            
            def run_batch_chat():
                outputs = llm.chat(prompts, sampling_params_list)
                return outputs
            
            if all(mode == "generate" for mode in modes):
                outputs = await asyncio.get_event_loop().run_in_executor(executor, run_batch)
            elif all(mode == "chat" for mode in modes):
                outputs = await asyncio.get_event_loop().run_in_executor(executor, run_batch_chat)
            else:
                raise ValueError("Mixed modes in batch are not supported")
            # Distribute results to futures
            for e, out in zip(batch, outputs):
                if modes[0] == "generate":
                    # print text
                    text = out.outputs[0].text
                    print("GENERATE OUTPUT TEXT")
                    print(text)
                    token_ids = out.outputs[0].token_ids
                    # decode with tokenizer
                    text = tokenizer.decode(token_ids)
                else:
                    text = out.outputs[0].text
                e["future"].set_result(text)

    @app.on_event("startup")
    async def startup_event():
        asyncio.create_task(batch_dispatcher())

    ### API Endpoints

    @app.post("/v1/completions", response_model=CompletionResponse)
    async def completions_endpoint(req: CompletionRequest):
        request_id = str(uuid.uuid4())
        prompts = req.prompt or []
        if not prompts:
            raise HTTPException(status_code=400, detail="No prompt provided")
        if req.n != 1:
            raise HTTPException(status_code=400, detail="n>1 not supported yet")

        prompt = prompts[0]
        sp = SamplingParams(temperature=req.temperature, top_p=req.top_p, max_tokens=req.max_tokens)

        loop = asyncio.get_event_loop()
        future = loop.create_future()
        await incoming_queue.put({
            "request_id": request_id,
            "mode": "generate",
            "prompt": prompt,
            "sampling_params": sp,
            "future": future
        })

        try:
            text = await asyncio.wait_for(future, timeout=60.0)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Timeout generating")

        response = CompletionResponse(
            id=request_id,
            created=int(time.time()),
            model=req.model,
            choices=[Choice(text=text, index=0, finish_reason="stop")],
            reasoning_effort=req.reasoning_effort
        )
        return response

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions_endpoint(req: ChatCompletionRequest):
        request_id = str(uuid.uuid4())
        conversation = req.messages
        sp = SamplingParams(temperature=req.temperature, top_p=req.top_p, max_tokens=req.max_tokens)

        loop = asyncio.get_event_loop()
        future = loop.create_future()
        await incoming_queue.put({
            "request_id": request_id,
            "mode": "chat",
            "conversation": conversation,
            "sampling_params": sp,
            "future": future
        })

        try:
            text = await asyncio.wait_for(future, timeout=60.0)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Timeout generating")

        assistant_message = Message(role="assistant", content=text)
        response = ChatCompletionResponse(
            id=request_id,
            created=int(time.time()),
            model=req.model,
            choices=[ChatChoice(message=assistant_message, index=0, finish_reason="stop")],
            reasoning_effort=req.reasoning_effort
        )
        return response

    @app.post("/v1/responses", response_model=ResponseResponse)
    async def responses_endpoint(req: ResponseRequest):
        request_id = str(uuid.uuid4())
        conversation = req.input
        top_p = 1 if not hasattr(req, 'top_p') else req.top_p
        sp = SamplingParams(temperature=req.temperature, top_p=top_p, max_tokens=req.max_output_tokens, skip_special_tokens=False, spaces_between_special_tokens=True)

        loop = asyncio.get_event_loop()
        future = loop.create_future()
        await incoming_queue.put({
            "request_id": request_id,
            "mode": "chat",
            "conversation": conversation,
            "sampling_params": sp,
            "future": future
        })

        try:
            text = await asyncio.wait_for(future, timeout=60.0)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Timeout generating")

        response = ResponseResponse(
            id=request_id,
            created=int(time.time()),
            model=req.model,
            output=ResponseOutput(text=text),
            reasoning_effort=req.reasoning_effort
        )
        return response

    @app.get("/v1/models", response_model=ModelsResponse)
    async def models_endpoint():
        data = [{
            "id": llm.model,  # adjust to actual property if needed
            "object": "model",
            "owned_by": "self-hosted",
            "permission": []
        }]
        return ModelsResponse(object="list", data=data)

    return app

def build_engine():
    ### API schema definitions
    # Build the offline engine once
    # tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-120b", cache_dir="/space1/asura/hf_home")
    llm = LLM(
        model="openai/gpt-oss-120b",
        seed=SEED,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=1,
        enforce_eager=True,
        download_dir="/space1/asura/hf_home",
        tokenizer="openai/gpt-oss-120b",
        # you can add other reproducibility kwargs per docs
    )
    return llm

if __name__ == "__main__":
    multiprocessing.freeze_support()
    multiprocessing.set_start_method("spawn", force=True)


    llm = build_engine()
    app = create_app(llm)
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
