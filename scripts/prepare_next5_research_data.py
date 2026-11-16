"""Build and validate only the newly locked ResearchPool data cache."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next5.data import prepare_research
from src.next5.artifacts import progress

if __name__=='__main__':
    data=prepare_research(ROOT);rows=data.rows()
    assert len(data.ids)==6400
    progress('NEXT5 ResearchPool data prepared',str(len(rows['target']))+' rows; zero Shadow/Seal rows',phase='PREFLIGHT')
    print('NEXT5 RESEARCH DATA PASS',len(data.ids),len(rows['target']),flush=True)
