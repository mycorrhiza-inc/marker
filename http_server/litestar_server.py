# HTTP server
import os
import traceback
import random
import redis
import boto3

from pathlib import Path
from pydantic import BaseModel
from typing import Optional, Annotated, Any, Dict
from litestar import MediaType, Request, Litestar, Controller, Response, post, get
from litestar.datastructures import UploadFile
from litestar.enums import RequestEncodingType
from litestar.params import Body, Parameter
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
REDIS_S3_URLS_KEY = os.getenv("REDIS_S3_URLS_KEY", "request_s3_urls")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")

s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
)


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
        redis_client.rpush(REDIS_PRIORITY_QUEUE_KEY, request_id)
    else:
        redis_client.rpush(REDIS_BACKGROUND_QUEUE_KEY, request_id)


def update_status_in_redis(request_id: int, status: Dict[str, str]) -> None:
    redis_client.hmset(str(request_id), status)


def upload_file_to_s3(file, file_name):
    s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=file_name, Body=file)
    return f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{file_name}"


class PDFProcessor(Controller):
    @post(path="/api/v1/marker")
    async def process_pdf_upload(
        self,
        data: Annotated[UploadFile, Body(media_type=RequestEncodingType.MULTI_PART)],
        priority: bool = True,
    ) -> dict:
        file = data.file
        request_id = random.randint(100000, 999999)
        s3_file_name = f"{request_id}.pdf"

        # Upload file to S3
        s3_url = upload_file_to_s3(file.file.read(), s3_file_name)

        # Update Redis with status and S3 URL
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
        redis_client.hset(REDIS_S3_URLS_KEY, str(request_id), s3_url)
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
