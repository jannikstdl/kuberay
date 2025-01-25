import os
import logging
from typing import Dict, Optional, List

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse, Response

from ray import serve

# --- VLLM imports ---
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
# Mit dem neuen Code kann man Abbrüche besser abfangen:
from vllm.entrypoints.utils import with_cancellation

# Für CLI-Argumente
from vllm.utils import FlexibleArgumentParser

logger = logging.getLogger("ray.serve")

app = FastAPI()


@serve.deployment(name="VLLMDeployment")
@serve.ingress(app)
class VLLMDeployment:
    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        response_role: str,
        lora_modules: Optional[List[LoRAModulePath]] = None,
        chat_template: Optional[str] = None,
        # Neue Felder, falls du sie brauchst:
        return_tokens_as_token_ids: bool = False,
        enable_auto_tools: bool = False,
        tool_parser: Optional[str] = None,
        enable_prompt_tokens_details: bool = False
    ):
        """
        Älteres Setup (nur /v1/chat/completions),
        aber mit neueren möglichen Parametern und einem Health-Check.
        """
        logger.info(f"Starting with engine args: {engine_args}")

        self.engine_args = engine_args
        self.response_role = response_role
        self.lora_modules = lora_modules
        self.chat_template = chat_template

        # Neuere Optionen, sofern deine vLLM-Version sie unterstützt:
        self.return_tokens_as_token_ids = return_tokens_as_token_ids
        self.enable_auto_tools = enable_auto_tools
        self.tool_parser = tool_parser
        self.enable_prompt_tokens_details = enable_prompt_tokens_details

        # In-Prozess-Engine starten
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        # Chat-Objekt wird lazy-initialisiert
        self.openai_serving_chat: Optional[OpenAIServingChat] = None

    @app.get("/health")
    async def health(self) -> Response:
        """
        Einfacher Health-Check, der die Engine überprüft.
        """
        await self.engine.check_health()
        return Response(status_code=200)

    @app.post("/v1/chat/completions")
    @with_cancellation
    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Request
    ):
        """
        OpenAI-kompatibler Endpoint für Chat Completions.
        Nur dieser Endpoint + /health, wie du es wünschst.
        """
        # Chat-Serving-Objekt bei Bedarf bauen
        if not self.openai_serving_chat:
            model_config = await self.engine.get_model_config()

            # Name(n) des Modells für die Chat-API
            if self.engine_args.served_model_name is not None:
                served_model_names = self.engine_args.served_model_name
            else:
                served_model_names = [self.engine_args.model]

            self.openai_serving_chat = OpenAIServingChat(
                engine_client=self.engine,
                model_config=model_config,
                served_model_names=served_model_names,
                response_role=self.response_role,
                lora_modules=self.lora_modules,
                chat_template=self.chat_template,
                prompt_adapters=None,
                request_logger=None,
                # Neue Parameter, falls vLLM sie unterstützt:
                chat_template_content_format="vllm",
                return_tokens_as_token_ids=self.return_tokens_as_token_ids,
                enable_auto_tools=self.enable_auto_tools,
                tool_parser=self.tool_parser,
                enable_prompt_tokens_details=self.enable_prompt_tokens_details,
            )

        logger.info(f"Request: {request}")

        # Chat Completion anfordern
        generator = await self.openai_serving_chat.create_chat_completion(
            request,
            raw_request
        )

        # Fehler?
        if isinstance(generator, ErrorResponse):
            return JSONResponse(
                content=generator.model_dump(),
                status_code=generator.code
            )

        # Streaming-Fall
        if request.stream:
            return StreamingResponse(
                content=generator,
                media_type="text/event-stream"
            )

        # Nicht-Streaming-Fall
        assert isinstance(generator, ChatCompletionResponse)
        return JSONResponse(content=generator.model_dump())


def parse_vllm_args(cli_args: Dict[str, str]):
    """
    Parst ein Dict von Key/Value-Argumenten in vLLM-Argumente via argparse.
    """
    parser = FlexibleArgumentParser(description="vLLM CLI")
    parser = make_arg_parser(parser)
    arg_strings = []
    for key, value in cli_args.items():
        arg_strings.extend([f"--{key}", str(value)])
    logger.info(f"vLLM CLI arg list: {arg_strings}")
    parsed_args = parser.parse_args(args=arg_strings)
    return parsed_args


def build_app(cli_args: Dict[str, str]) -> serve.Application:
    """
    Baut das Ray-Serve-App (nur Chat-Endpoint + Health-Check).
    """
    parsed_args = parse_vllm_args(cli_args)
    engine_args = AsyncEngineArgs.from_cli_args(parsed_args)
    engine_args.worker_use_ray = True

    # Hier kannst du neue Parameter reinpacken, falls du sie
    # über ENV oder CLI-Args steuern willst.
    return VLLMDeployment.bind(
        engine_args,
        parsed_args.response_role,
        parsed_args.lora_modules,
        parsed_args.chat_template,
        return_tokens_as_token_ids=False,
        enable_auto_tools=False,
        tool_parser=None,
        enable_prompt_tokens_details=False
    )


# Beispiel: ENV-Variablen (MODEL_ID etc.)
model = build_app({
    "model": os.environ["MODEL_ID"],
    "gpu-memory-utilization": os.environ["GPU_MEMORY_UTILIZATION"],
    "download-dir": os.environ["DOWNLOAD_DIR"],
    "max-model-len": os.environ["MAX_MODEL_LEN"],
    "tensor-parallel-size": os.environ["TENSOR_PARALLELISM"],
    "pipeline-parallel-size": os.environ["PIPELINE_PARALLELISM"],
})
