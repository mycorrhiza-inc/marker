import traceback

import click
import os

from marker.models import create_model_dict
from pydantic import BaseModel, Field

from marker.config.parser import ConfigParser
from marker.output import text_from_rendered

import base64
from typing import Any, Optional, Annotated
import io

from marker.converters.pdf import PdfConverter

import logging
from pathlib import Path
from typing import Tuple, Dict
import boto3
import redis
from urllib.parse import urlparse
import asyncio
import sys
import random
import json


app_data = {}


UPLOAD_DIRECTORY = "./uploads"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)


class CommonParams(BaseModel):
    filepath: Annotated[
        Optional[str], Field(description="The path to the PDF file to convert.")
    ]
    page_range: Annotated[
        Optional[str],
        Field(
            description="Page range to convert, specify comma separated page numbers or ranges.  Example: 0,5-10,20",
            example=None,
        ),
    ] = None
    languages: Annotated[
        Optional[str],
        Field(
            description="Comma separated list of languages to use for OCR. Must be either the names or codes from from https://github.com/VikParuchuri/surya/blob/master/surya/languages.py.",
            example=None,
        ),
    ] = None
    force_ocr: Annotated[
        bool,
        Field(
            description="Force OCR on all pages of the PDF.  Defaults to False.  This can lead to worse results if you have good text in your PDFs (which is true in most cases)."
        ),
    ] = False
    paginate_output: Annotated[
        bool,
        Field(
            description="Whether to paginate the output.  Defaults to False.  If set to True, each page of the output will be separated by a horizontal rule that contains the page number (2 newlines, {PAGE_NUMBER}, 48 - characters, 2 newlines)."
        ),
    ] = False
    output_format: Annotated[
        str,
        Field(
            description="The format to output the text in.  Can be 'markdown', 'json', or 'html'.  Defaults to 'markdown'."
        ),
    ] = "markdown"


async def _convert_pdf(params: CommonParams):
    assert params.output_format in ["markdown", "json", "html"], "Invalid output format"
    assert params.filepath is not None, "Encountered an empty filepath"
    assert os.path.exists(params.filepath), "Filepath does not exist"
    try:
        options = params.model_dump()
        print(options)
        config_parser = ConfigParser(options)
        config_dict = config_parser.generate_config_dict()
        config_dict["pdftext_workers"] = 1
        converter = PdfConverter(
            config=config_dict,
            artifact_dict=app_data["models"],
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
        )
        print("Rendering filepath :", params.filepath)
        rendered = converter(params.filepath)
        text, _, images = text_from_rendered(rendered)
        metadata = rendered.metadata
    except Exception as e:
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
        }

    encoded = {}
    for k, v in images.items():
        byte_stream = io.BytesIO()
        v.save(byte_stream, format="PNG")
        encoded[k] = base64.b64encode(byte_stream.getvalue()).decode("utf-8")

    return {
        "format": params.output_format,
        "output": text,
        "images": encoded,
        "metadata": metadata,
        "success": True,
    }


TMP_DIR = Path("/tmp")
MARKER_TMP_DIR = TMP_DIR / Path("marker")

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

S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
S3_REGION = os.getenv("S3_REGION")
S3_ENDPOINT = os.getenv("S3_ENDPOINT")
TASKS_PER_CONTAINER = int(os.getenv("TASKS_PER_CONTAINER", "-1"))
assert TASKS_PER_CONTAINER > 0, "Tasks per container must be > 0 and defined in .env"

s3_client = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    region_name=S3_REGION,
)

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
logger = logging.getLogger(__name__)


class RequestStatus(BaseModel):
    status: str
    success: str
    request_id: str
    request_check_url: str
    request_check_url_leaf: str
    markdown: str = ""
    error: str = ""
    images: str = ""


def get_status_from_redis(request_id: int) -> RequestStatus:
    raw_dict = redis_client.hgetall(str(request_id))
    try:
        status = RequestStatus(**raw_dict)
        return status
    except Exception as e:
        raise e


def set_status_in_redis(request_id: int, status: RequestStatus) -> None:
    status_dict = status.model_dump()
    redis_client.hmset(str(request_id), status_dict)


def pop_from_queue() -> Optional[int]:
    # TODO : Clean up code logic
    request_id = redis_client.lpop(REDIS_PRIORITY_QUEUE_KEY)
    if request_id is None:
        request_id = redis_client.lpop(REDIS_BACKGROUND_QUEUE_KEY)
    if request_id is None:
        return None
    if isinstance(request_id, int):
        return request_id
    if isinstance(request_id, str):
        return int(request_id)
    logger.error(type(request_id))
    raise Exception(
        f"Request id is not string or none and is {type(request_id)} instead."
    )


def parse_s3_uri_to_bucket_and_key(s3_uri: str) -> Tuple[str, str]:
    """
    Parses an S3 URI and creates a boto3 request.

    Args:
        s3_uri (str): The S3 URI to parse.

    Returns:
        dict: A dictionary containing the bucket name and key.
    """
    parsed_url = urlparse(s3_uri)
    if parsed_url.hostname is None:
        raise Exception("Invalid S3 URI")

    # Extract the bucket name from the hostname
    bucket_name = parsed_url.hostname.split(".")[0]

    # Extract the key from the path
    key = parsed_url.path.lstrip("/")

    return (bucket_name, key)


def download_file_from_s3_url(s3_url: str, local_path: Path) -> None:
    s3_bucket, s3_key = parse_s3_uri_to_bucket_and_key(s3_url)
    # If you want to make this async use the async boto implementation
    # Doing it on a seperate thread as async might be causing some new
    # issues with unparsable pdfs
    s3_client.download_file(s3_bucket, s3_key, str(local_path))


async def process_pdf_from_s3(request_id: int) -> None:
    doc_dir = MARKER_TMP_DIR / Path(str(request_id))
    os.makedirs(doc_dir / Path("in"), exist_ok=True)
    input_directory = doc_dir / Path("in")
    output_directory = doc_dir / Path("out")

    # Get PDF URL from Redis
    status = get_status_from_redis(request_id)
    s3_url = str(redis_client.hget(REDIS_S3_URLS_KEY, str(request_id)))
    if s3_url is None:
        status.status = "error"
        status.success = str(False)
        status.error = "No S3 URL found"
        set_status_in_redis(
            request_id,
            status,
        )
        return None

    # Download PDF from S3
    pdf_filename = input_directory / f"{request_id}.pdf"
    try:
        download_file_from_s3_url(s3_url, pdf_filename)
    except Exception as e:
        logger.error(
            f"Encountered error while processing {request_id} in getting file from s3"
        )
        logger.error(e)
        status.status = "error"
        status.success = str(False)
        status.error = "Error in retreiving file from s3: " + str(e)
        set_status_in_redis(
            request_id,
            status,
        )
        return None

    # Now process as normal
    #
    #
    #
    params = CommonParams(
        filepath=str(pdf_filename),
        page_range=None,
        languages=None,
        force_ocr=False,
        paginate_output=True,
        output_format="markdown",
    )
    try:
        results = await _convert_pdf(params)
    except Exception as e:
        status.status = "error"
        status.success = str(False)
        status.error = "Error in processing pdf: " + str(e)
    else:
        if results.get("success") is not True:  # Also catches the none case
            status.success = str(False)
            status.error = str(results.get("error"))
            print("Encountered error while processing pdf")
            print(results.get("error"))
        else:
            status.markdown = results["output"]
            status.images = json.dumps(results["images"])
            status.status = "complete"
            status.success = str(True)
    finally:
        set_status_in_redis(
            request_id,
            status,
        )
        os.remove(pdf_filename)


def pdf_to_md_path(pdf_path: Path) -> Path:
    return (pdf_path.parent).parent / Path(f"out/{pdf_path.stem}/{pdf_path.stem}.md")


async def background_worker():
    rand_seconds = random.randint(0, 3)
    await asyncio.sleep(rand_seconds)
    print("Background worker started", file=sys.stderr)
    while True:
        request_id = pop_from_queue()
        if request_id is not None:
            print(
                f"Beginning to Process pdf with request: {request_id}", file=sys.stderr
            )
            await process_pdf_from_s3(request_id)
        else:
            # print("No request found", file=sys.stderr)
            await asyncio.sleep(2)


def initialize_background_workers(num_workers: Optional[int] = None):
    if num_workers is None:
        num_workers = TASKS_PER_CONTAINER
    for _ in range(num_workers):
        asyncio.create_task(background_worker())


def main():
    app_data["models"] = create_model_dict()
    initialize_background_workers()
    try:
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        if "models" in app_data:
            del app_data["models"]
        sys.exit(0)


if __name__ == "__main__":
    main()
