import os
import logging
from typing import Dict, Optional, List

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import (
    StreamingResponse,
    JSONResponse,
    Response  # Needed for health-check
)

# The new code often uses uvloop, which is optional in your context:
# import uvloop

from ray import serve

# vLLM imports
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse
)
# Newer approach: use OpenAIServingModels for base model & LoRA management
from vllm.entrypoints.openai.serving_models import (
    BaseModelPath,
    OpenAIServingModels
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.utils import with_cancellation  # for handling canceled requests
from vllm.logger import init_logger

# If you want to log more details, you can initialize or configure the logger:
# NOTE: The new code calls init_logger("vllm.entrypoints.openai.api_server")
#       You can adapt that as needed:
logger = init_logger("ray.serve")

# This can also be used in your environment if you want usage stats, etc.:
# from vllm.usage.usage_lib import UsageContext

# LoRAModulePath is used if you have LoRA modules
from vllm.entrypoints.openai.serving_engine import LoRAModulePath

from vllm.utils import FlexibleArgumentParser

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
    ):
        """
        Dies ist der Konstruktor deines Deployment.

        Wir starten hier die Engine und legen die weiteren Hilfs-Objekte
        an, sobald sie tatsächlich benötigt werden (Lazy Initialization).
        """
        # If you'd like to remove any previously set CUDA devices:
        # if 'CUDA_VISIBLE_DEVICES' in os.environ:
        #     del os.environ['CUDA_VISIBLE_DEVICES']

        logger.info(f"Starting with engine args: {engine_args}")
        self.engine_args = engine_args
        self.response_role = response_role
        self.lora_modules = lora_modules
        self.chat_template = chat_template

        # Start the engine in-process (as in your old code)
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        # We'll keep references to these, but only build them on first request
        self.openai_serving_models: Optional[OpenAIServingModels] = None
        self.openai_serving_chat: Optional[OpenAIServingChat] = None

    @app.get("/health")
    async def health(self) -> Response:
        """
        Health check endpoint.
        In the newer code, there's a check to ensure the engine is responsive.
        """
        await self.engine.check_health()  # If it fails, an exception is raised
        return Response(status_code=200)

    @app.post("/v1/chat/completions")
    @with_cancellation
    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Request
    ):
        """
        Handles OpenAI-style ChatCompletion requests.
        (Newer vLLM uses OpenAIServingChat plus optionally a streaming approach.)
        """
        # Lazy-load the serving objects if not already created
        if self.openai_serving_chat is None:
            # 1) Get the model config from the engine
            model_config = await self.engine.get_model_config()

            # 2) Build the list of base models.
            #    Typically, we define "BaseModelPath" here. If you only have one model,
            #    that is enough.
            if self.engine_args.served_model_name is not None:
                # This can be a list of strings if you want multiple model names
                served_model_names = self.engine_args.served_model_name
            else:
                served_model_names = [self.engine_args.model]

            base_model_paths = [
                BaseModelPath(name=name, model_path=self.engine_args.model)
                for name in served_model_names
            ]

            # 3) Create OpenAIServingModels (handles main model path + optional LoRAs)
            self.openai_serving_models = OpenAIServingModels(
                engine_client=self.engine,
                model_config=model_config,
                base_model_paths=base_model_paths,
                # If you have a list of LoRA modules, pass them here
                lora_modules=self.lora_modules,
                # prompt_adapters can be assigned if needed
                prompt_adapters=None,
            )
            # Now load/initialize the static LoRAs if present
            await self.openai_serving_models.init_static_loras()

            # 4) Create the chat serving object
            self.openai_serving_chat = OpenAIServingChat(
                engine_client=self.engine,
                model_config=model_config,
                serving_models=self.openai_serving_models,
                response_role=self.response_role,
                # You can add a request_logger if you want
                request_logger=None,
                # If you have a custom template, pass it here:
                chat_template=self.chat_template,
                # Additional new parameters in the latest version:
                chat_template_content_format="vllm",   # default
                return_tokens_as_token_ids=False,
                enable_auto_tools=False,
                tool_parser=None,
                enable_prompt_tokens_details=False,
            )

        logger.info(f"Request: {request}")

        # 5) Create the actual response generator
        generator = await self.openai_serving_chat.create_chat_completion(
            request,
            raw_request
        )
        if isinstance(generator, ErrorResponse):
            return JSONResponse(
                content=generator.model_dump(),
                status_code=generator.code
            )
        if request.stream:
            return StreamingResponse(
                content=generator,
                media_type="text/event-stream"
            )
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())


def parse_vllm_args(cli_args: Dict[str, str]):
    """
    Parses vLLM args based on CLI inputs.

    The new code also uses FlexibleArgumentParser + make_arg_parser().
    We keep that approach.
    """
    parser = FlexibleArgumentParser(description="vLLM CLI")
    make_arg_parser(parser)
    arg_strings = []
    for key, value in cli_args.items():
        arg_strings.extend([f"--{key}", str(value)])
    logger.info(arg_strings)
    parsed_args = parser.parse_args(args=arg_strings)
    return parsed_args


def build_app(cli_args: Dict[str, str]) -> serve.Application:
    """
    Build the Serve app based on CLI arguments.
    In the new code, they'd do a lot more, e.g. CORSMiddleware, etc.
    Here, we keep it minimal for demonstration.
    """
    parsed_args = parse_vllm_args(cli_args)
    # Let the engine know we want to run in Ray
    engine_args = AsyncEngineArgs.from_cli_args(parsed_args)
    engine_args.worker_use_ray = True

    # Create the deployment
    return VLLMDeployment.bind(
        engine_args,
        parsed_args.response_role,
        parsed_args.lora_modules,
        parsed_args.chat_template,
    )


# Example usage: building the app with environment variable overrides
model = build_app({
    "model": os.environ["MODEL_ID"],
    "served-model-name":  os.environ["SERVED_MODEL_NAME"],
    "gpu-memory-utilization": os.environ["GPU_MEMORY_UTILIZATION"],
    "download-dir": os.environ["DOWNLOAD_DIR"],
    "max-model-len": os.environ["MAX_MODEL_LEN"],
    "tensor-parallel-size": os.environ["TENSOR_PARALLELISM"],
    "pipeline-parallel-size": os.environ["PIPELINE_PARALLELISM"],
})
