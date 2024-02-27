from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Optional
from pydantic import BaseModel
import logging
import kg_summarizer.config as CFG

from kg_summarizer.trapi import EdgeContainer
from kg_summarizer.ai import generate_response
from kg_summarizer.utils import LoggingUtil


logger = LoggingUtil.init_logging(
    __name__,
    level=logging.INFO,
    format_sel="medium",
)


class LLMParameters(BaseModel):
    gpt_model: str = "gpt-3.5-turbo"
    temperature: Optional[float] = 0.0
    system_prompt: Optional[str] = ""


class Parameters(BaseModel):
    llm: Optional[LLMParameters]
    # trapi: Optional[TrapiParameters]


class AbstractItem(BaseModel):
    abstract: str
    parameters: Parameters


class EdgeItem(BaseModel):
    edge: dict
    parameters: Parameters


KG_SUM_VERSION = "0.0.8"

# declare the application and populate some details
app = FastAPI(
    title="Knowledge Graph Summarizer - A FastAPI UI/web service",
    version=KG_SUM_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex="^https?:\/\/(((.+\.)*?renci\.org)|(localhost))(:\d{1,5})?(\/.*)?$",  # localhost and *.renci.org
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# if we're in development mode (PYTHON_ENV=dev) tell the browser we can access the server on this
# user's local machine. See https://developer.chrome.com/blog/private-network-access-preflight
if CFG.ENV.get("PYTHON_ENV") == "dev":

    class PrivateNetworkHeaderMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            response = await call_next(request)
            response.headers["Access-Control-Allow-Private-Network"] = "true"
            return response

    app.add_middleware(PrivateNetworkHeaderMiddleware)


@app.post("/summarize/abstract")
async def summarize_abstract_handler(item: AbstractItem):
    system_prompt = f"""
    You are a pharmacology researcher summarizing publication abstracts. Condense the follow abstract to a single sentence.
    """

    summary = generate_response(
        system_prompt,
        item.abstract,
        item.parameters.llm.gpt_model,
        item.parameters.llm.temperature,
    )
    return summary


@app.post("/summarize/edge")
async def summarize_edge_handler(item: EdgeItem):
    edge = EdgeContainer(item.edge)

    spo_sentence = edge.format_spo_sentence()

    if item.parameters.llm.system_prompt:
        system_prompt = item.parameters.llm.system_prompt
    else:
        system_prompt = f"""
        Summarize the following edge publication abstracts listed in the knowledge graph. Make sure the summary supports the statement '{spo_sentence}'. Only use information explicitly stated in the publication abstracts. I repeat, do not make up any information.
        """

    logger.info(f"GPT Prompt: {system_prompt}")
    logger.info(f"GPT Mode: {item.parameters.llm.gpt_model}")
    logger.info(f"GPT Temperature: {item.parameters.llm.temperature}")
    logger.info(f"GPT Input: {edge}")

    summary = generate_response(
        system_prompt,
        str(edge),
        item.parameters.llm.gpt_model,
        item.parameters.llm.temperature,
    )

    logger.info(f"GPT Summary: {summary}")
    return summary
