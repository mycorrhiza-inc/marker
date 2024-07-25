import os
import traceback
import random
import redis

from pathlib import Path
from pydantic import BaseModel
from typing import Optional, Annotated, Any, Dict
from litestar import MediaType, Request, Litestar, Controller, Response, post, get
from litestar.datastructures import UploadFile
from litestar.enums import RequestEncodingType
from litestar.params import Body
from litestar.params import Parameter
from litestar.status_codes import HTTP_500_INTERNAL_SERVER_ERROR
import uvicorn
import logging


REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = 6379
REDIS_STATUS_KEY = os.getenv("REDIS_STATUS_KEY", "request_status")
REDIS_BACKGROUND_QUEUE_KEY = os.getenv(
    "REDIS_BACKGROUND_QUEUE_KEY", "request_queue_background"
)
REDIS_PRIORITY_QUEUE_KEY = os.getenv(
    "REDIS_PRIORITY_QUEUE_KEY", "request_queue_priority"
)
SHARED_VOLUME_DIR = Path(os.getenv("SHARED_VOLUME_DIR", "/shared"))
MARKER_TMP_DIR = SHARED_VOLUME_DIR / Path("marker")


class PDFUploadFormData(BaseModel):
    file: bytes


class URLUpload(BaseModel):
    url: str


class PathUpload(BaseModel):
    path: str


class RequestStatus(BaseModel):
    status: str
    success: str
    request_id: str
    request_check_url: str
    request_check_url_leaf: str
    markdown: Optional[str] = None
    error: Optional[str] = None


logger = logging.getLogger(__name__)

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def get_status_from_redis(request_id: int) -> dict:
    status = redis_client.hgetall(str(request_id))
    if status is None:
        return {"status": "not_found", "success": str(False)}
    logger.info(type(status))
    return status


def push_to_queue(request_id: int, priority: bool):
    if priority:
        redis_client.rpush(REDIS_BACKGROUND_QUEUE_KEY, request_id)
    else:
        redis_client.rpush(REDIS_PRIORITY_QUEUE_KEY, request_id)


def update_status_in_redis(request_id: int, status: Dict[str, str]) -> None:
    test = redis_client.hmset(str(request_id), status)


class PDFProcessor(Controller):
    @post(path="/api/v1/marker")
    async def process_pdf_upload(
        self,
        data: Annotated[UploadFile, Body(media_type=RequestEncodingType.MULTI_PART)],
        priority: bool = True,
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
                "priority": str(priority),
            },
        )
        push_to_queue(request_id, priority)

        return {
            "success": True,
            "error": "None",
            "request_id": str(request_id),
            "request_check_url": f"https://marker.kessler.xyz/api/v1/marker/{request_id}",
            "request_check_url_leaf": f"/api/v1/marker/{request_id}",
            "priority": str(priority),
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


def start_server():
    port = os.environ.get("MARKER_PORT", 2718)
    port = int(port)
    app = Litestar(
        route_handlers=[PDFProcessor],
        exception_handlers={Exception: plain_text_exception_handler},
    )
    run_config = uvicorn.Config(app, port=port, host="0.0.0.0")
    server = uvicorn.Server(run_config)

    # Start background worker in a separate thread

    server.run()


if __name__ == "__main__":
    start_server()
