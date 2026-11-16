"""Check review documents and self-contained HTML navigation without a network."""
from pathlib import Path
import sys,re,argparse
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next6_extension.common import *

class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key in ('href','src') and value:self.links.append(value)

def check(paths,allowed_root):
    records=[]
    for p in paths:
        source=p.read_text(encoding='utf-8-sig');assert '\ufffd' not in source,'Unicode replacement character: '+str(p)
        if p.suffix=='.html':parser=Links();parser.feed(source);links=parser.links
        else:links=[m.group(1) for m in re.finditer(r'\[[^\]]+\]\(([^)]+)\)',source)]
        for link in links:
            parsed=urlsplit(link)
            if parsed.scheme in ('https','http','data','mailto') or not parsed.path:continue
            assert not parsed.scheme,'Unexpected link scheme: '+link
            target=(p.parent/unquote(parsed.path)).resolve();assert target.is_relative_to(allowed_root.resolve()),'Local navigation outside handoff: '+str(target)
            assert target.is_file(),'Missing document target: '+str(p)+' -> '+link
            records.append({'source':p.relative_to(allowed_root).as_posix(),'link':link,'target':target.relative_to(allowed_root).as_posix()})
    return records

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['preliminary','final'],required=True);a=parser.parse_args()
    paths=sorted(BASE.glob('*.md'))+[BASE/'DYNAMIC_STATE_VIEWER.html',BASE/'FIGURES/index.html'];records=check(paths,ROOT)
    save(f'audit/{a.stage.upper()}_DOCUMENT_LINKS.json',{'status':'PASS','created_utc':utc_now(),'documents':len(paths),'local_links_checked':len(records),'network_requests':0,'files_sha256':{p.relative_to(ROOT).as_posix():digest(p) for p in paths},'links':records,'source_sha256':digest(Path(__file__))})
    print('Document links PASS',len(paths),len(records))

if __name__=='__main__':main()
