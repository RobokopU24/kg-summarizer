from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel

# from reasoner_pydantic import Response as PDResponse

from kg_summarizer.trapi import EdgeContainer
from kg_summarizer.ai import generate_response


class LLMParameters(BaseModel):
    gpt_model: str = "gpt-3.5-turbo"
    temperature: Optional[float] = 0.0
    system_prompt: Optional[str] = ""


# class TrapiParameters(BaseModel):
#     result_idx: Optional[int] = 0


class Parameters(BaseModel):
    llm: Optional[LLMParameters]
    # trapi: Optional[TrapiParameters]


class AbstractItem(BaseModel):
    abstract: str
    parameters: Parameters


# class ResponseItem(BaseModel):
#     response: PDResponse
#     parameters: Parameters


class EdgeItem(BaseModel):
    edge: dict
    parameters: Parameters


KG_SUM_VERSION = "0.0.6"

# declare the application and populate some details
app = FastAPI(
    title="Knowledge Graph Summarizer - A FastAPI UI/web service",
    version=KG_SUM_VERSION,
)


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

    summary = generate_response(
        system_prompt,
        str(edge),
        item.parameters.llm.gpt_model,
        item.parameters.llm.temperature,
    )
    return summary
