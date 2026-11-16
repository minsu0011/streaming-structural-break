"""Subprocess bridge for the pinned official runner's execute() API (synthetic only)."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.validation.protocol import official_runner

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--command',required=True);parser.add_argument('--data',required=True);parser.add_argument('--model',required=True);parser.add_argument('--parameters',default='{}');args=parser.parse_args()
    import main as user_module
    from src.streaming.inference import write_engineering_baseline
    class Module:
        def get_function(self,name):
            if name=='train':return lambda datasets,model_directory_path:write_engineering_baseline(model_directory_path)
            return getattr(user_module,name)
        def get_value(self,key,default=None):return getattr(user_module,key,default)
    context=SimpleNamespace(trip_data_fuse=lambda:None)
    commands=official_runner().execute(context,Module(),args.data,args.model)
    commands[args.command](**json.loads(args.parameters))

if __name__=='__main__':main()
