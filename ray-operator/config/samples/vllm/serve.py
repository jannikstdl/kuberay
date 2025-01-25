import os
import logging
from typing import Dict, Optional, List

import ray
from ray import serve

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse, Response

# vLLM imports for the language model engine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_engine import LoRAModulePath, BaseModelPath
from vllm.utils import FlexibleArgumentParser

logger = logging.getLogger("ray.serve")
app = FastAPI()


def parse_vllm_args(cli_args: Dict[str, str]):
    """
    Helper function to parse vLLM arguments from a dictionary.
    This converts dictionary key-value pairs into command line arguments.
    """
    parser = FlexibleArgumentParser(description="vLLM CLI")
    parser = make_arg_parser(parser)
    arg_strings = []
    for key, value in cli_args.items():
        if value is None:
            # Some vLLM flags might be flags without values (e.g. --enable-chunked-prefill)
            # If the environment var is present but empty, we just append the flag
            arg_strings.append(f"--{key}")
        else:
            arg_strings.extend([f"--{key}", str(value)])
    logger.info(f"vLLM CLI arg list: {arg_strings}")
    parsed_args = parser.parse_args(args=arg_strings)
    return parsed_args


@serve.deployment(name="VLLMDeployment")
@serve.ingress(app)
class VLLMDeployment:
    """
    Main class that handles the vLLM deployment with Ray Serve.
    This provides an OpenAI-compatible API for chat completions.
    """

    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        response_role: str,
        lora_modules: Optional[List[LoRAModulePath]] = None,
        chat_template: Optional[str] = None,
    ):
        """
        Initialize the deployment. This includes loading the model into an async engine.
        """
        # Store initialization arguments
        self.engine_args = engine_args
        self.response_role = response_role
        self.chat_template = chat_template

        # Initialize the async LLM engine (this loads the model)
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info(f"Initialized AsyncLLMEngine with args: {engine_args}")

        # We'll fetch the model configuration when needed
        self.model_config = None

        # OpenAIServingChat will be initialized on first request
        self.openai_serving_chat: Optional[OpenAIServingChat] = None

    @app.get("/health")
    async def health(self) -> Response:
        """
        Simple health check endpoint.
        Returns 200 if the service is running.
        """
        return Response(status_code=200)

    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Request
    ):
        """
        Main endpoint for chat completions.
        This is compatible with the OpenAI Chat API.
        """
        # 1) Load model configuration if not already loaded
        if self.model_config is None:
            self.model_config = await self.engine.get_model_config()

        # 2) Get the model name from arguments
        if self.engine_args.served_model_name is not None:
            model_name = self.engine_args.served_model_name
        else:
            model_name = self.engine_args.model

        # 3) Initialize OpenAIServingChat if not already done
        if not self.openai_serving_chat:
            # Create BaseModelPath with name and path
            base_model_paths = [
                BaseModelPath(
                    name=model_name,
                    model_path=self.engine_args.model
                )
            ]

            # Initialize the chat service with all required parameters
            self.openai_serving_chat = OpenAIServingChat(
                engine_client=self.engine,
                model_config=self.model_config,
                base_model_paths=base_model_paths,
                response_role=self.response_role,
                lora_modules=None,  # You can pass lora_modules here if needed
                prompt_adapters=None,
                request_logger=None,
                chat_template=self.chat_template,
                chat_template_content_format="auto",
                return_tokens_as_token_ids=False,
                enable_auto_tools=False,
                tool_parser=None,
                enable_prompt_tokens_details=False,
            )

        logger.info(f"Received request: {request}")

        # 4) Generate the chat completion
        result = await self.openai_serving_chat.create_chat_completion(
            request,
            raw_request
        )

        # 5) Handle error responses
        if isinstance(result, ErrorResponse):
            return JSONResponse(
                content=result.model_dump(),
                status_code=result.code
            )

        # 6) Handle streaming responses
        if request.stream:
            return StreamingResponse(
                content=result,
                media_type="text/event-stream"
            )

        # 7) Handle normal (non-streaming) responses
        assert isinstance(result, ChatCompletionResponse)
        return JSONResponse(content=result.model_dump())


def build_app(cli_args: Dict[str, str]) -> serve.Application:
    """
    Builds the Ray Serve application based on CLI arguments.

    1. Parse environment or user-provided arguments for vLLM.
    2. Create an AsyncEngineArgs object.
    3. Return a bound Ray Serve deployment that can be started.
    """
    parsed_args = parse_vllm_args(cli_args)
    engine_args = AsyncEngineArgs.from_cli_args(parsed_args)
    engine_args.worker_use_ray = True  # We're using Ray Serve
    logger.info(f"Building application with engine_args: {engine_args}")

    return VLLMDeployment.bind(
        engine_args=engine_args,
        response_role=parsed_args.response_role,
        lora_modules=parsed_args.lora_modules,
        chat_template=parsed_args.chat_template,
    )


# Beispielhafter Aufruf über Environment-Variablen:
env_args = {
    "model": os.environ.get("MODEL_ID", "some-default-model"),
    "download-dir": os.environ.get("DOWNLOAD_DIR", "./downloaded_models"),
    "tensor-parallel-size": os.environ.get("TENSOR_PARALLELISM", "1"),
    "pipeline-parallel-size": os.environ.get("PIPELINE_PARALLELISM", "1"),
    "gpu-memory-utilization": os.environ.get("GPU_MEMORY_UTILIZATION", "0.9"),
    "max-model-len": os.environ.get("MAX_MODEL_LEN"),
}

if os.environ.get("ENABLE_CHUNKED_PREFILL", "False").lower() == "true":
    env_args["enable-chunked-prefill"] = ""  # flag without value

if os.environ.get("ENABLE_PREFIX_CACHING", "False").lower() == "true":
    env_args["enable-prefix-caching"] = ""  # flag without value

model = build_app(env_args)
