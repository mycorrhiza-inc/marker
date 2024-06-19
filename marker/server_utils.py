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

