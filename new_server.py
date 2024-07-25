import os

from requests.api import request
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
        raise e


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


def shutdown():
    global model_refs, pool
    if pool:
        pool.close()
        pool.join()
    del model_refs


# ==================================================
# Seperate above code into its own file at some point
# ==================================================
# from .server_utils import (
#     init_models_and_workers,
#     process_single_pdf,
#     shutdown,
# )
import os
import shutil
import traceback
import time
import redis
import threading

from pathlib import Path
from pydantic import BaseModel
from typing import Optional, Annotated, Any, Dict
import logging


import pymupdf  # PyMuPDF
import numpy as np

import sys

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)


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
REDIS_STATUS_KEY = os.getenv("REDIS_STATUS_KEY", "request_status")
REDIS_BACKGROUND_QUEUE_KEY = os.getenv(
    "REDIS_BACKGROUND_QUEUE_KEY", "request_queue_background"
)
REDIS_PRIORITY_QUEUE_KEY = os.getenv(
    "REDIS_PRIORITY_QUEUE_KEY", "request_queue_priority"
)


redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def update_status_in_redis(request_id: int, status: Dict[str, str]) -> None:
    test = redis_client.hmset(str(request_id), status)


# Clean up definition stuff at some point
def pop_from_queue() -> Optional[int]:
    request_id = redis_client.lpop(REDIS_PRIORITY_QUEUE_KEY)
    if isinstance(request_id, int):
        return request_id
    if isinstance(request_id, str):
        return int(request_id)
    if request_id is None:
        request_id = redis_client.lpop(REDIS_BACKGROUND_QUEUE_KEY)
        if isinstance(request_id, int):
            return request_id
        if isinstance(request_id, str):
            return int(request_id)
        if request_id is None:
            return None
    logger.error(type(request_id))
    raise Exception(
        f"Request id is not string or none and is {type(request_id)} instead."
    )


def process_pdf_from_given_docdir(request_id: int) -> None:
    doc_dir = MARKER_TMP_DIR / Path(str(request_id))
    print(doc_dir, file=sys.stderr)
    try:
        input_directory = doc_dir / Path("in")
        output_directory = doc_dir / Path("out")
        os.makedirs(input_directory, exist_ok=True)
        os.makedirs(output_directory, exist_ok=True)
        print(input_directory, file=sys.stderr)
        print(output_directory, file=sys.stderr)

        def get_pdf_files(pdf_path: Path) -> list[Path]:
            if not pdf_path.is_dir():
                raise ValueError("Path is not a directory")
            return [f for f in pdf_path.iterdir() if f.is_file() and f.suffix == ".pdf"]

        pdf_list = get_pdf_files(input_directory)
        if len(pdf_list) == 0:
            logger.error(f"Encountered error while processing {request_id}")
            logger.error("No PDF File Found.")
            update_status_in_redis(
                request_id,
                {
                    "status": "error",
                    "success": str(False),
                    "error": "No PDF file found",
                },
            )
            return

        first_pdf_filepath = pdf_list[0]
        chunk_paths = split_large_pdf(first_pdf_filepath)

        full_markdown = ""
        for chunk_path in chunk_paths:
            process_single_pdf(chunk_path, output_directory)
            chunk_markdown_path = pdf_to_md_path(chunk_path)
            if not os.path.exists(chunk_markdown_path):
                update_status_in_redis(
                    request_id,
                    {
                        "status": "error",
                        "success": str(False),
                        "error": f"Output markdown file not found at : {chunk_markdown_path}",
                    },
                )
                return

            with open(chunk_markdown_path, "r") as f:
                full_markdown += f.read()

        update_status_in_redis(
            request_id,
            {"status": "complete", "success": str(True), "markdown": full_markdown},
        )
    except Exception as e:
        logger.error(f"Encountered error while processing {request_id}")
        logger.error(e)
        update_status_in_redis(
            request_id, {"status": "error", "success": str(False), "error": str(e)}
        )
    finally:
        shutil.rmtree(doc_dir)


def split_large_pdf(pdf_path: Path, max_pages: int = 300) -> list[Path]:
    logger.info("Splitting large pdf.")
    doc = pymupdf.open(pdf_path)
    num_pages = doc.page_count
    chunk_paths = []

    if num_pages <= max_pages:
        return [pdf_path]

    base_path = pdf_path.parent / Path("chunks")
    os.makedirs(base_path, exist_ok=True)

    split_ranges = np.array_split(range(num_pages), np.ceil(num_pages / max_pages))

    for idx, page_range in enumerate(split_ranges):
        chunk_path = base_path / f"{pdf_path.stem}_chunk{idx + 1}.pdf"
        chunk_doc = pymupdf.open()

        for page_num in page_range:
            chunk_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)

        chunk_doc.save(chunk_path)
        chunk_doc.close()
        chunk_paths.append(chunk_path)

    doc.close()
    return chunk_paths


def pdf_to_md_path(pdf_path: Path) -> Path:
    return (pdf_path.parent).parent / Path(f"out/{pdf_path.stem}/{pdf_path.stem}.md")


def background_worker():
    print("Starting Background Worker", file=sys.stderr)
    while True:
        request_id = pop_from_queue()
        if request_id is not None:
            print(
                f"Beginning to Process pdf with request: {request_id}", file=sys.stderr
            )
            process_pdf_from_given_docdir(request_id)
        else:
            print(
                "No new pdf's to process checking again in 1 second.", file=sys.stderr
            )
            time.sleep(1)


def start_server():
    print("Test output to stdout")
    print("Test output to stderr", file=sys.stderr)
    logger.info("Initializing models and workers.")
    init_models_and_workers(workers=5)
    background_worker()


if __name__ == "__main__":
    start_server()
