import traceback

import click
import os

import uvicorn
from pydantic import BaseModel, Field
from starlette.responses import HTMLResponse

from marker.config.parser import ConfigParser
from marker.output import text_from_rendered

import base64
from contextlib import asynccontextmanager
from typing import Optional, Annotated
import io

from fastapi import FastAPI, Form, File, UploadFile
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict

import logging
from pathlib import Path
from typing import Tuple, Dict
import boto3
import redis
from urllib.parse import urlparse
import asyncio
import sys
import random


app_data = {}


UPLOAD_DIRECTORY = "./uploads"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_data["models"] = create_model_dict()
    initialize_background_workers(1)

    yield

    if "models" in app_data:
        del app_data["models"]


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return HTMLResponse(
        """
<h1>Marker API</h1>
<ul>
    <li><a href="/docs">API Documentation</a></li>
    <li><a href="/marker">Run marker (post request only)</a></li>
</ul>
"""
    )


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


@app.post("/marker")
async def convert_pdf(params: CommonParams):
    return await _convert_pdf(params)


@app.post("/marker/upload")
async def convert_pdf_upload(
    page_range: Optional[str] = Form(default=None),
    languages: Optional[str] = Form(default=None),
    force_ocr: Optional[bool] = Form(default=False),
    paginate_output: Optional[bool] = Form(default=False),
    output_format: Optional[str] = Form(default="markdown"),
    file: UploadFile = File(
        ..., description="The PDF file to convert.", media_type="application/pdf"
    ),
):
    upload_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
    with open(upload_path, "wb") as upload_file:
        file_contents = await file.read()
        upload_file.write(file_contents)

    params = CommonParams(
        filepath=upload_path,
        page_range=page_range,
        languages=languages,
        force_ocr=force_ocr,
        paginate_output=paginate_output,
        output_format=output_format,
    )
    results = await _convert_pdf(params)
    os.remove(upload_path)
    return results


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
    markdown: Optional[str] = None
    error: Optional[str] = None


def update_status_in_redis(request_id: int, status: Dict[str, str]) -> None:
    redis_client.hmset(str(request_id), status)


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


async def download_file_from_s3_url(s3_url: str, local_path: Path) -> None:
    s3_bucket, s3_key = parse_s3_uri_to_bucket_and_key(s3_url)

    def run_download():
        s3_client.download_file(s3_bucket, s3_key, str(local_path))

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, run_download)


async def process_pdf_from_s3(request_id: int) -> None:
    doc_dir = MARKER_TMP_DIR / Path(str(request_id))
    os.makedirs(doc_dir / Path("in"), exist_ok=True)
    input_directory = doc_dir / Path("in")
    output_directory = doc_dir / Path("out")

    # Get PDF URL from Redis
    s3_url = str(redis_client.hget(REDIS_S3_URLS_KEY, str(request_id)))
    if s3_url is None:
        update_status_in_redis(
            request_id,
            {"status": "error", "success": str(False), "error": "No S3 URL found"},
        )
        return None

    # Download PDF from S3
    pdf_filename = input_directory / f"{request_id}.pdf"
    try:
        await download_file_from_s3_url(s3_url, pdf_filename)
    except Exception as e:
        logger.error(
            f"Encountered error while processing {request_id} in getting file from s3"
        )
        logger.error(e)
        update_status_in_redis(
            request_id,
            {
                "status": "error",
                "success": str(False),
                "error": "Error in retreiving file from s3: " + str(e),
            },
        )
        return None

    # Now process as normal
    #
    #
    #
    params = CommonParams(
        filepath=pdf_filename,
        page_range=None,
        languages=None,
        force_ocr=None,
        paginate_output=True,
        output_format="markdown",
    )
    try:
        results = await _convert_pdf(params)
    except Exception as e:
        update_status_in_redis(
            request_id,
            {
                "status": "error",
                "success": str(False),
                "error": "Error in processing pdf: " + str(e),
            },
        )
    else:
        text = results["output"]
        update_status_in_redis(
            request_id,
            {"status": "complete", "success": str(True), "markdown": text},
        )
    finally:
        os.remove(pdf_filename)


def pdf_to_md_path(pdf_path: Path) -> Path:
    return (pdf_path.parent).parent / Path(f"out/{pdf_path.stem}/{pdf_path.stem}.md")


async def background_worker():
    rand_seconds = random.randint(0, 10)
    await asyncio.sleep(rand_seconds)
    while True:
        request_id = pop_from_queue()
        if request_id is not None:
            print(
                f"Beginning to Process pdf with request: {request_id}", file=sys.stderr
            )
            await process_pdf_from_s3(request_id)
        else:
            print("No request found", file=sys.stderr)
            await asyncio.sleep(5)


def initialize_background_workers(num_workers: int = 1):
    for _ in range(num_workers):
        asyncio.create_task(background_worker())


@click.command()
@click.option("--port", type=int, default=8000, help="Port to run the server on")
@click.option("--host", type=str, default="127.0.0.1", help="Host to run the server on")
def main(port: int, host: str):
    # Run the server
    uvicorn.run(
        app,
        host=host,
        port=port,
    )


if __name__ == "__main__":
    main()
