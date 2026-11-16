"""One fresh model consumer for local DEV-only wire verification; no targets."""
import os,sys
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.validation.protocol import official_runner
from src.streaming.inference import infer


def main():
    if len(sys.argv)!=3:raise ValueError('Endpoint and model directory required')
    endpoint=int(sys.argv[1]) if os.name=='nt' else sys.argv[1]
    directory=Path(sys.argv[2]).resolve()
    if not (directory/'model.json').is_file():raise ValueError('An existing model descriptor is required')
    official=official_runner()
    with official.os_create_client_socket() as sock:
        sock.settimeout(60);official.os_connect(sock,endpoint)
        official._run_with_double_protection(remote=official.Remote(sock),
            consumer_factory=lambda datasets:infer(datasets,directory),post_processor=official._post_process_infer_yield_result)


if __name__=='__main__':main()
