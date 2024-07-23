import os
import pypdfium2  # Needs to be at the top to avoid warnings
import argparse
import torch.multiprocessing as mp
from tqdm import tqdm
import math

from marker.convert import convert_single_pdf
from marker.output import markdown_exists, save_markdown
from marker.pdf.utils import find_filetype
from marker.pdf.extract_text import get_length_of_text
from marker.models import load_all_models
from marker.settings import settings
from marker.logger import configure_logging


from typing import Optional
from pathlib import Path

try:
    mp.set_start_method("spawn")
except RuntimeError as e:
    if "context has already been set" in str(e):
        pass  # Ignore the error if the context has already been set
    else:
        raise
os.environ["IN_STREAMLIT"] = "true"  # Avoid multiprocessing inside surya
os.environ["PDFTEXT_CPU_WORKERS"] = "1"  # Avoid multiprocessing inside pdftext

configure_logging()

model_refs = None
pool = None


def worker_init(shared_model):
    global model_refs
    model_refs = shared_model


def worker_exit():
    global model_refs
    del model_refs


def init_models_and_workers(workers):
    global model_refs, pool
    model_lst = load_all_models()

    for model in model_lst:
        if model is None:
            continue

        if model.device.type == "mps":
            raise ValueError(
                "Cannot use MPS with torch multiprocessing share_memory. You have to use CUDA or CPU. Set the TORCH_DEVICE environment variable to change the device."
            )

        model.share_memory()

    model_refs = model_lst

    total_processes = int(workers)
    if settings.CUDA:
        tasks_per_gpu = (
            settings.INFERENCE_RAM // settings.VRAM_PER_TASK if settings.CUDA else 0
        )
        total_processes = int(min(tasks_per_gpu, total_processes))

    try:
        mp.set_start_method("spawn")
    except RuntimeError as e:
        if "context has already been set" in str(e):
            pass  # Ignore the error if the context has already been set
        else:
            raise
    pool = mp.Pool(
        processes=total_processes, initializer=worker_init, initargs=(model_lst,)
    )


def process_single_pdf(
    filepath: Path,
    out_folder: Path,
    metadata: Optional[dict] = None,
    min_length: Optional[int] = None,
) -> Optional[str]:
    string_filepath = str(filepath)
    fname = os.path.basename(filepath)
    if markdown_exists(out_folder, fname):
        return
    try:
        if min_length:
            filetype = find_filetype(string_filepath)
            if filetype == "other":
                return None

            length = get_length_of_text(string_filepath)
            if length < min_length:
                return
        global model_refs
        full_text, images, out_metadata = convert_single_pdf(
            string_filepath, model_refs, metadata=metadata
        )
        if len(full_text.strip()) > 0:
            save_markdown(out_folder, fname, full_text, images, out_metadata)
            return full_text
        else:
            print(f"Empty file: {filepath}. Could not convert.")
    except Exception as e:
        print(f"Error converting {filepath}: {e}")
        print(traceback.format_exc())


def process_pdfs_core_server(
    in_folder: Path,
    out_folder: Path,
    chunk_idx: int,
    num_chunks: int,
    max_pdfs: int,
    min_length: Optional[int],
    metadata_file: Optional[Path],
):
    print(f"called pdf processing core")

    files = [os.path.join(in_folder, f) for f in os.listdir(in_folder)]
    files = [f for f in files if os.path.isfile(f)]
    os.makedirs(out_folder, exist_ok=True)

    chunk_size = math.ceil(len(files) / num_chunks)
    start_idx = chunk_idx * chunk_size
    end_idx = start_idx + chunk_size
    files_to_convert = files[start_idx:end_idx]
    print(f"all variables initialized")

    if max_pdfs:
        files_to_convert = files_to_convert[:max_pdfs]

    metadata = {}
    if metadata_file:
        metadata_file_path = os.path.abspath(metadata_file)
        with open(metadata_file_path, "r") as f:
            metadata = json.load(f)

    task_args = [
        (f, out_folder, metadata.get(os.path.basename(f)), min_length)
        for f in files_to_convert
    ]
    print(f"calling actual function")

    def process_single_pdf_singlearg(args):
        filepath, out_folder, metadata, min_length = args
        return process_single_pdf(filepath, out_folder, metadata, min_length)

    try:
        list(
            tqdm(
                pool.imap(process_single_pdf_singlearg, task_args),
                total=len(task_args),
                desc="Processing PDFs",
                unit="pdf",
            )
        )
        return {
            "status": "success",
            "message": f"Processed {len(files_to_convert)} PDFs.",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


import base64
import secrets
import os
import signal
import shutil
import sys
import traceback

from pathlib import Path
from pydantic import BaseModel
from typing import Optional, Annotated
from litestar import MediaType, Request, Litestar, Controller, Response, post, get
from litestar.datastructures import UploadFile
from litestar.enums import RequestEncodingType
from litestar.params import Body
from litestar.status_codes import HTTP_500_INTERNAL_SERVER_ERROR
import requests
import uvicorn


import random

# Process the file in the background
# This is a simplified demonstration; ideally, use background task queue like Celery
import asyncio

background_tasks = set()


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
    success: bool
    request_id: int
    request_check_url: str
    markdown: Optional[str] = None
    meta: Optional[dict] = None
    error: Optional[str] = None
    page_count: Optional[int] = None


# In-memory store (simple example, use a persistent store in practice)
request_status = {}


class PDFProcessor(Controller):
    async def process_pdf_from_given_docdir(
        self, request_id: int, doc_dir: Path
    ) -> None:
        try:
            input_directory = doc_dir / Path("in")
            output_directory = doc_dir / Path("out")
            os.makedirs(input_directory, exist_ok=True)
            os.makedirs(output_directory, exist_ok=True)

            def get_pdf_files(pdf_path: Path) -> list[Path]:
                if not pdf_path.is_dir():
                    raise ValueError("Path is not a directory")
                return [
                    f for f in pdf_path.iterdir() if f.is_file() and f.suffix == ".pdf"
                ]

            pdf_list = get_pdf_files(input_directory)
            if len(pdf_list) == 0:
                request_status[request_id].update(
                    status="error", success=False, error="No PDF files found"
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
                request_status[request_id].update(
                    status="error",
                    success=False,
                    error=f"Output markdown file not found at : {output_filename}",
                )
                return

            with open(output_filename, "r") as f:
                markdown_content = f.read()

            request_status[request_id].update(
                status="complete", success=True, markdown=markdown_content, page_count=1
            )  # Simplified for example
        except Exception as e:
            request_status[request_id].update(
                status="error", success=False, error=str(e)
            )
        finally:
            shutil.rmtree(doc_dir)

    @post("/api/v1/marker", media_type=MediaType.JSON)
    async def process_pdf_upload(
        self,
        file: Annotated[UploadFile, Body(media_type=RequestEncodingType.MULTI_PART)],
        langs: Annotated[str, Body()] = "en",
        force_ocr: Annotated[bool, Body()] = False,
        paginate: Annotated[bool, Body()] = False,
    ) -> dict:
        request_id = random.randint(100000, 999999)
        doc_dir = MARKER_TMP_DIR / Path(str(request_id))
        os.makedirs(doc_dir / Path("in"), exist_ok=True)
        pdf_filename = doc_dir / Path("in") / Path(rand_string() + ".pdf")

        with open(pdf_filename, "wb") as f:
            f.write(file.read())

        request_status[request_id] = {
            "status": "processing",
            "success": True,
            "request_id": request_id,
            "request_check_url": f"/api/v1/marker/{str(request_id)}",
        }
        asyncio.create_task(self.process_pdf_from_given_docdir(request_id, doc_dir))
        # Uncessesary, state is managed in the memory queue.
        # task = asyncio.create_task(
        #     self.process_pdf_from_given_docdir(request_id, doc_dir)
        # )
        # background_tasks.add(task)
        # task.add_done_callback(background_tasks.discard)

        return {
            "success": True,
            "error": None,
            "request_id": request_id,
            "request_check_url": f"/api/v1/marker/{str(request_id)}",
        }

    @get("/api/v1/marker/{request_id}", media_type=MediaType.JSON)
    async def get_request_status(self, request_id: int) -> dict:
        return request_status.get(request_id, {"status": "not_found", "success": False})


def plain_text_exception_handler(request: Request, exc: Exception) -> Response:
    tb = traceback.format_exc()
    status_code = getattr(exc, "status_code", HTTP_500_INTERNAL_SERVER_ERROR)
    return Response(media_type=MediaType.TEXT, content=tb, status_code=status_code)


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
    server.run()
    shutdown()


def shutdown():
    global model_refs, pool
    if pool:
        pool.close()
        pool.join()
    del model_refs


if __name__ == "__main__":
    start_server()
