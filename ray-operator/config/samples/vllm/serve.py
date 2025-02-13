import os
from typing import Dict, Optional, List
import logging

# Comments in English
# Prometheus library imports
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from fastapi import FastAPI, Response
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse

from ray import serve

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_engine import LoRAModulePath
from vllm.utils import FlexibleArgumentParser

logger = logging.getLogger("ray.serve")

# Create a FastAPI app
app = FastAPI()

@app.get("/metrics")
def metrics():
    """
    Dieser Endpoint gibt alle Metriken aus der Default-Registry
    im Prometheus-Format zurück. Wenn vLLM seine Metriken korrekt
    registriert, erscheinen sie hier unter `vllm:...`.
    
    Wichtig:
    - Wenn du mehrere Ray-Replikas (oder mehrere Prozesse) hast,
      brauchst du evtl. die Multiprozess-Sammlung. Siehe unten.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@serve.deployment(name="VLLMDeployment")
@serve.ingress(app)
class VLLMDeployment:
    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        response_role: str,
        lora_modules: Optional[List[LoRAModulePath]] = None,
        chat_template: Optional[str] = None,
    ):
        logger.info(f"Initializing VLLMDeployment with {engine_args}")
        self.openai_serving_chat = None
        self.engine_args = engine_args
        self.response_role = response_role
        self.lora_modules = lora_modules
        self.chat_template = chat_template
        # Create the vLLM engine
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Request
    ):
        """
        OpenAI-compatible HTTP endpoint für Chat Completion.
        """
        if not self.openai_serving_chat:
            # Lazy-init der Chat-Logik
            model_config = await self.engine.get_model_config()
            if self.engine_args.served_model_name is not None:
                served_model_names = self.engine_args.served_model_name
            else:
                served_model_names = [self.engine_args.model]

            self.openai_serving_chat = OpenAIServingChat(
                self.engine,
                model_config,
                served_model_names=served_model_names,
                response_role=self.response_role,
                lora_modules=self.lora_modules,
                chat_template=self.chat_template,
                prompt_adapters=None,
                request_logger=None,
            )

        logger.info(f"Incoming request: {request}")
        generator = await self.openai_serving_chat.create_chat_completion(
            request, raw_request
        )

        if isinstance(generator, ErrorResponse):
            return JSONResponse(
                content=generator.model_dump(), status_code=generator.code
            )
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())


def parse_vllm_args(cli_args: Dict[str, str]):
    """
    vLLM CLI-Argumente parsen. Standardmäßig ist `disable_metrics=False`,
    sodass Metriken an Prometheus gesendet werden, wenn die Engine
    das unterstützt.
    """
    parser = FlexibleArgumentParser(description="vLLM CLI")
    parser = make_arg_parser(parser)
    arg_strings = []
    for key, value in cli_args.items():
        arg_strings.extend([f"--{key}", str(value)])
    logger.info(f"Argument-Strings für vLLM: {arg_strings}")
    parsed_args = parser.parse_args(args=arg_strings)
    return parsed_args


def build_app(cli_args: Dict[str, str]) -> serve.Application:
    """
    Baut das Ray-Serve Deployment, inkl. VLLMDeployment.
    """
    parsed_args = parse_vllm_args(cli_args)
    engine_args = AsyncEngineArgs.from_cli_args(parsed_args)
    # Wichtig, damit wir das Ray-Backend nutzen
    engine_args.worker_use_ray = True

    return VLLMDeployment.bind(
        engine_args,
        parsed_args.response_role,
        parsed_args.lora_modules,
        parsed_args.chat_template,
    )


# Erstelle eine Serve-App (bzw. -Deployment) mit den gewünschten Parametern
model = build_app(
    {
        "model": os.environ["MODEL_ID"],
        "gpu-memory-utilization": os.environ["GPU_MEMORY_UTILIZATION"],
        "download-dir": os.environ["DOWNLOAD_DIR"],
        "max-model-len": os.environ["MAX_MODEL_LEN"],
        "tensor-parallel-size": os.environ["TENSOR_PARALLELISM"],
        "pipeline-parallel-size": os.environ["PIPELINE_PARALLELISM"],
        
        # Falls du METRICS deaktivieren willst (nicht empfohlen), könntest du:
        # "disable-metrics": "True"
    }
)
