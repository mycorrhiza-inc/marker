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
    mp.set_start_method('spawn')
except RuntimeError as e:
    if 'context has already been set' in str(e):
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
            raise ValueError("Cannot use MPS with torch multiprocessing share_memory. You have to use CUDA or CPU. Set the TORCH_DEVICE environment variable to change the device.")

        model.share_memory()

    model_refs = model_lst

    total_processes = int(workers)
    if settings.CUDA:
        tasks_per_gpu = settings.INFERENCE_RAM // settings.VRAM_PER_TASK if settings.CUDA else 0
        total_processes = int(min(tasks_per_gpu, total_processes))

    try:
        mp.set_start_method('spawn')
    except RuntimeError as e:
        if 'context has already been set' in str(e):
            pass  # Ignore the error if the context has already been set
        else:
            raise
    pool = mp.Pool(processes=total_processes, initializer=worker_init, initargs=(model_lst,))


def process_single_pdf(filepath : Path, out_folder : Path, metadata : Optional[dict] = None, min_length : Optional[int] = None ) -> Optional[str]:
    string_filepath=str(filepath)
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
        full_text, images, out_metadata = convert_single_pdf(string_filepath, model_refs, metadata=metadata)
        if len(full_text.strip()) > 0:
            save_markdown(out_folder, fname, full_text, images, out_metadata)
            return full_text
        else:
            print(f"Empty file: {filepath}. Could not convert.")
    except Exception as e:
        print(f"Error converting {filepath}: {e}")
        print(traceback.format_exc())

def process_pdfs_core_server(in_folder : Path, out_folder : Path , chunk_idx : int , num_chunks : int, max_pdfs : int, min_length : Optional[int] , metadata_file : Optional[Path]):
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

    task_args = [(f, out_folder, metadata.get(os.path.basename(f)), min_length) for f in files_to_convert]
    print(f"calling actual function")
    def process_single_pdf_singlearg(args):
        filepath, out_folder, metadata, min_length = args
        return process_single_pdf(filepath,out_folder,metadata,min_length)


    try:
        list(tqdm(pool.imap(process_single_pdf_singlearg, task_args), total=len(task_args), desc="Processing PDFs", unit="pdf"))
        return {"status": "success", "message": f"Processed {len(files_to_convert)} PDFs."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# Server Stuff.
#
#
import base64
import secrets
def rand_string() -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(8)).decode()

from pathlib import Path

from pydantic import BaseModel

from typing import Optional
import signal  # Add this import to handle signal
from litestar import MediaType, Request, Litestar, Controller, Response, post, get  # Importing Litestar
import traceback
import json
import uvicorn

import os
import shutil
import requests
import sys

from litestar.status_codes import HTTP_500_INTERNAL_SERVER_ERROR

class BaseMarkerCliInput(BaseModel):
    in_folder: str
    out_folder: str 
    chunk_idx: int = 0
    num_chunks : int = 1
    max_pdfs : Optional[int] = None 
    min_length : Optional[int] = None 
    metadata_file : Optional[str] = None

class PDFUploadFormData(BaseModel):
    file: bytes

class URLUpload(BaseModel):
    url: str
class PathUpload(BaseModel):
    path: str

TMP_DIR = Path("/tmp")
MARKER_TMP_DIR = TMP_DIR / Path("marker")
class PDFProcessor(Controller):
    @get("/test_pdf_processing")
    async def test_pdf(self, request: Request ) -> str:
        print("This a test message")
        print("This is a test message sent to stderr",sys.stderr)
        doc_dir = MARKER_TMP_DIR / Path(rand_string())
        input_directory = doc_dir / Path("in")
        os.makedirs(input_directory, exist_ok=True)
        return await self.process_pdf_from_given_docdir(Path(data.path))
    @post("/process_pdf_from_file_path")
    async def process_pdf_from_file_path(self,data : URLUpload, request: Request ) -> str:
        print("This a test message")
        print("This is a test message sent to stderr",sys.stderr)
        doc_dir = MARKER_TMP_DIR / Path(rand_string())
        input_directory = doc_dir / Path("in")
        os.makedirs(input_directory, exist_ok=True)
        shutil.copy(data.path, input_directory)
        return await self.process_pdf_from_given_docdir(Path(data.path))

    async def process_pdf_from_given_docdir(self,doc_dir : Path) -> str:
        print(f"Called function on {doc_dir}")
        input_directory = doc_dir / Path("in")
        output_directory = doc_dir / Path("out")
        
        # Ensure the directories exist
        os.makedirs(input_directory, exist_ok=True)

        os.makedirs(output_directory, exist_ok=True)


        def get_pdf_files(pdf_path: Path) -> list[Path]:
            if not pdf_path.is_dir():
                raise ValueError("Path is not a directory")
            return [f for f in pdf_path.iterdir() if f.is_file() and f.suffix == '.pdf']
        print("trying to locate pdf file")
        pdf_list = get_pdf_files(input_directory)
        if len(pdf_list) == 0 :
            print(f"No PDF's found in input directory : {input_directory}")
            print(f"{input_directory.iterdir()}")
            return ""
        first_pdf_filepath = pdf_list[0]
        print(f"found pdf at: {first_pdf_filepath}")
        process_single_pdf(first_pdf_filepath,output_directory)
        
        print("Successfully processed pdf.")
        # Read the output markdown file
        # TODO : Fix at some point with tests
        def pdf_to_md_path(pdf_path: Path) -> Path:
            return pdf_path.with_suffix('.md').parent / Path(f'out/{pdf_path.stem}/{pdf_path.stem}.md')
        output_filename = pdf_to_md_path(first_pdf_filepath)
        if not os.path.exists(output_filename):
            return f"Output markdown file not found at : {output_filename}"
        with open(output_filename, "r") as f:
            markdown_content = f.read()
        # Cleanup directories
        shutil.rmtree(doc_dir)
        # Return the markdown content as a response
        return markdown_content 

    @post("/process_pdf_from_url")
    async def process_pdf_from_url(self,data : URLUpload ) -> str:
        print("This a test message")
        def download_file(url: str, savedir: Path,extension : Optional[str]=None) -> Path:
            if extension is None:
                extension = ""
            # TODO: Use a temporary directory for downloads or archive it in some other way.
            local_filename = savedir / Path(rand_string()+extension)
            # NOTE the stream=True parameter below
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(local_filename, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        # If you have chunk encoded response uncomment if
                        # and set chunk_size parameter to None.
                        # if chunk:
                        f.write(chunk)
            return local_filename
        doc_dir = MARKER_TMP_DIR / Path(rand_string())
        print(f"This a test message: {doc_dir}")

        input_directory = doc_dir / Path("in")
        os.makedirs(input_directory, exist_ok=True)
        print(f"This a test message: {input_directory}")
        download_file(data.url,input_directory,".pdf")
        return await self.process_pdf_from_given_docdir(doc_dir)
        
    @post("/process_pdf_upload", media_type="multipart/form-data")
    async def process_pdf_upload_endpoint(self,data : PDFUploadFormData ) -> str:
        doc_dir = MARKER_TMP_DIR / Path(rand_string())
        # Parse the uploaded file
        pdf_binary = data.file
        input_directory = doc_dir / Path("in")
        # Ensure the directories exist
        os.makedirs(input_directory, exist_ok=True)
        # Save the PDF to the output directory
        pdf_filename = input_directory / Path(rand_string() + ".pdf")
        with open(pdf_filename, "wb") as f:
            f.write(pdf_binary)
        return await self.process_pdf_from_given_docdir(doc_dir)


    @post("/process_pdfs_raw_cli")
    async def process_pdfs_endpoint_raw_cli(self,data: BaseMarkerCliInput) -> None:
        in_folder = data.in_folder
        out_folder = data.out_folder
        chunk_idx = data.chunk_idx
        num_chunks = data.num_chunks
        max_pdfs = data.max_pdfs
        min_length = data.min_length
        metadata_file = data.metadata_file

        result = process_pdfs_core(in_folder, out_folder, chunk_idx, num_chunks, max_pdfs, min_length, metadata_file)


def plain_text_exception_handler(request: Request, exc: Exception) -> Response:
    """Default handler for exceptions subclassed from HTTPException."""
    tb = traceback.format_exc()
    status_code = getattr(exc, "status_code", HTTP_500_INTERNAL_SERVER_ERROR)
    detail = getattr(exc, "detail", "")

    return Response(
        media_type=MediaType.TEXT,
        content=tb,
        status_code=status_code,
    )

def start_server():
    init_models_and_workers(workers=5)  # Initialize models and workers with a default worker count of 5
    port = os.environ.get("MARKER_PORT")
    if port is None:
        port = 2718
    app = Litestar(
        route_handlers = [PDFProcessor],
        exception_handlers={Exception: plain_text_exception_handler},
    )

    run_config = uvicorn.Config(app, port=port, host="0.0.0.0")
    server = uvicorn.Server(run_config)
    signal.signal(signal.SIGINT, lambda s, f: shutdown())  # Updated to catch Control-C and run shutdown
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



