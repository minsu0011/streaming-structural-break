"""Run one registered task with stack diagnostics before the 30-minute hang gate."""
from pathlib import Path
import sys
import faulthandler
import runpy

ROOT=Path(__file__).resolve().parents[1]


def execute(arguments):
    if not arguments:raise ValueError('Explicit child arguments required')
    sys.path.insert(0,str(ROOT))
    faulthandler.enable()
    # Diagnostic output is finite and infrequent. The supervisor still checks
    # actual CPU/progress; a traceback does not prove a normally busy fit hung.
    faulthandler.dump_traceback_later(1500,repeat=True)
    try:
        if arguments[0]=='-m':
            if len(arguments)<2 or arguments[1]!='pytest':raise ValueError('Only the pytest module is accepted')
            sys.argv=['pytest']+arguments[2:]
            runpy.run_module('pytest',run_name='__main__',alter_sys=True)
        else:
            script=(ROOT/arguments[0]).resolve()
            if not script.is_relative_to((ROOT/'scripts').resolve()) or not script.is_file():
                raise ValueError('Registered project script required')
            sys.argv=[str(script)]+arguments[1:]
            runpy.run_path(str(script),run_name='__main__')
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__=='__main__':
    arguments=sys.argv[1:]
    if arguments and arguments[0]=='--':arguments=arguments[1:]
    execute(arguments)
