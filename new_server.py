from .server_utils import (
    init_models_and_workers,
    process_single_pdf,
    shutdown,
)
import base64
import secrets
import os
import signal
import shutil
import sys
import traceback
import asyncio
import time
import random
import redis
import threading

from pathlib import Path
from pydantic import BaseModel
from typing import Optional, Annotated, Any, Dict
from litestar import MediaType, Request, Litestar, Controller, Response, post, get
from litestar.datastructures import UploadFile
from litestar.enums import RequestEncodingType
from litestar.params import Body
from litestar.params import Parameter
from litestar.status_codes import HTTP_500_INTERNAL_SERVER_ERROR
import requests
import uvicorn


def rand_string() -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(8)).decode()


class BaseMarkerCliInput(BaseModel):
    in_folder: str
    out_folder: str
    chunk_idx: int = 0
    num_chunks: int = 1
    max_pdfs: Optional[int] = None
    min_length: Optional[int] = None
    metadata_file: Optional[str] = None


class PDFUploadFormData(BaseModel):
    file: bytes


class URLUpload(BaseModel):
    url: str


class PathUpload(BaseModel):
    path: str


TMP_DIR = Path("/tmp")
MARKER_TMP_DIR = TMP_DIR / Path("marker")


class RequestStatus(BaseModel):
    status: str
    success: str
    request_id: str
    request_check_url: str
    request_check_url_leaf: str
    markdown: Optional[str] = None
    error: Optional[str] = None


REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = 6379
REDIS_STATUS_KEY = "request_status"
REDIS_QUEUE_KEY = "request_queue"

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def update_status_in_redis(request_id: int, status: Dict[str, str]) -> None:
    test = redis_client.hmset(str(request_id), status)
    return test


def get_status_from_redis(request_id: int) -> dict:
    status = redis_client.hgetall(str(request_id))
    if status is None:
        return {"status": "not_found", "success": str(False)}
    return status


def push_to_queue(request_id: int):
    redis_client.rpush(REDIS_QUEUE_KEY, request_id)


def pop_from_queue() -> Optional[int]:
    request_id = redis_client.lpop(REDIS_QUEUE_KEY)
    return int(request_id) if request_id else None


async def run_background_process(request_id: int):
    push_to_queue(request_id)


def process_pdf_from_given_docdir(request_id: int) -> None:
    doc_dir = MARKER_TMP_DIR / Path(str(request_id))
    try:
        input_directory = doc_dir / Path("in")
        output_directory = doc_dir / Path("out")
        os.makedirs(input_directory, exist_ok=True)
        os.makedirs(output_directory, exist_ok=True)

        def get_pdf_files(pdf_path: Path) -> list[Path]:
            if not pdf_path.is_dir():
                raise ValueError("Path is not a directory")
            return [f for f in pdf_path.iterdir() if f.is_file() and f.suffix == ".pdf"]

        pdf_list = get_pdf_files(input_directory)
        if len(pdf_list) == 0:
            update_status_in_redis(
                request_id,
                {"status": "error", "success": False, "error": "No PDF files found"},
            )
            return

        first_pdf_filepath = pdf_list[0]
        process_single_pdf(first_pdf_filepath, output_directory)

        def pdf_to_md_path(pdf_path: Path) -> Path:
            return (pdf_path.parent).parent / Path(
                f"out/{pdf_path.stem}/{pdf_path.stem}.md"
            )

        output_filename = pdf_to_md_path(first_pdf_filepath)
        if not os.path.exists(output_filename):
            update_status_in_redis(
                request_id,
                {
                    "status": "error",
                    "success": str(False),
                    "error": f"Output markdown file not found at : {output_filename}",
                },
            )
            return

        with open(output_filename, "r") as f:
            markdown_content = f.read()

        update_status_in_redis(
            request_id,
            {"status": "complete", "success": str(True), "markdown": markdown_content},
        )
    except Exception as e:
        update_status_in_redis(
            request_id, {"status": "error", "success": str(False), "error": str(e)}
        )
    finally:
        shutil.rmtree(doc_dir)


class PDFProcessor(Controller):
    @post(path="/api/v1/marker")
    async def process_pdf_upload(
        self,
        data: Annotated[UploadFile, Body(media_type=RequestEncodingType.MULTI_PART)],
    ) -> dict:
        file = data.file
        request_id = random.randint(100000, 999999)
        doc_dir = MARKER_TMP_DIR / Path(str(request_id))
        os.makedirs(doc_dir / Path("in"), exist_ok=True)
        pdf_filename = doc_dir / Path("in") / Path(f"{request_id}.pdf")

        with open(pdf_filename, "wb") as f:
            f.write(file.read())

        update_status_in_redis(
            request_id,
            {
                "status": "processing",
                "success": str(True),
                "request_id": str(request_id),
                "request_check_url": f"https://marker.kessler.xyz/api/v1/marker/{request_id}",
                "request_check_url_leaf": f"/api/v1/marker/{request_id}",
            },
        )
        await run_background_process(request_id)

        return {
            "success": True,
            "error": "None",
            "request_id": str(request_id),
            "request_check_url": f"https://marker.kessler.xyz/api/v1/marker/{request_id}",
            "request_check_url_leaf": f"/api/v1/marker/{request_id}",
        }

    @get(path="/api/v1/marker/{request_id:int}")
    async def get_request_status(
        self,
        request_id: int = Parameter(
            title="Request ID", description="Request id to retieve"
        ),
    ) -> dict:
        return get_status_from_redis(request_id)


def plain_text_exception_handler(request: Request, exc: Exception) -> Response:
    tb = traceback.format_exc()
    status_code = getattr(exc, "status_code", HTTP_500_INTERNAL_SERVER_ERROR)
    return Response(media_type=MediaType.TEXT, content=tb, status_code=status_code)


def background_worker():
    while True:
        request_id = pop_from_queue()
        if request_id:
            process_pdf_from_given_docdir(request_id)
        else:
            time.sleep(1)


def start_server():
    init_models_and_workers(workers=5)
    port = os.environ.get("MARKER_PORT", 2718)
    app = Litestar(
        route_handlers=[PDFProcessor],
        exception_handlers={Exception: plain_text_exception_handler},
    )
    run_config = uvicorn.Config(app, port=port, host="0.0.0.0")
    server = uvicorn.Server(run_config)
    signal.signal(signal.SIGINT, lambda s, f: shutdown())

    # Start background worker in a separate thread
    worker_thread = threading.Thread(target=background_worker)
    worker_thread.start()

    server.run()
    shutdown()


if __name__ == "__main__":
    start_server()
