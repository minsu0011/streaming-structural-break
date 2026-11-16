from pathlib import Path
import sys
import hashlib
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now


if __name__=='__main__':
    from src.next9.data import FeatureBank
    from src.next5.reference_features import implementation_hash
    with activity('Verify existing RP-only J102 C205 feature cache extension for early excluded reference fits','REFERENCE_GUARD','B6_BL80_REDUNDANCY'):
        bank=FeatureBank();files={};caches={}
        for kind in ['J102','C205']:
            paths=list((ROOT/'data/next5').glob('REF_'+kind+'_*/MANIFEST.json'))
            assert len(paths)==1
            p=paths[0];m=read_json(p);reg=ROOT/f'artifacts/next5/references/registrations/REF_{kind}.json';r=read_json(reg)
            assert m['status']=='COMPLETE' and m['identity']['scope']=='NEXT5_RESEARCH_POOL_ONLY'
            assert m['identity']['raw_manifest_sha256']==read_json(BASE/'PREFLIGHT_NEXT9.json')['cache_manifest_sha256']
            assert m['identity']['split_sha256']==bank.guard.split.digest
            assert m['identity']['row_ids_sha256']==hashlib.sha256(bank.row_ids.tobytes()).hexdigest()
            assert m['identity']['registration_sha256']==digest(reg)
            assert m['identity']['feature_source_sha256']==implementation_hash(r['feature_settings'])==r['feature_sha256']
            values=p.parent/'features.npy';assert digest(values)==m['features_sha256']
            for f in [p,values]:files[f.relative_to(ROOT).as_posix()]=digest(f)
            caches[kind]=dict(manifest=p.relative_to(ROOT).as_posix(),features=values.relative_to(ROOT).as_posix(),registration=reg.relative_to(ROOT).as_posix(),registration_sha256=digest(reg))
        atomic(BASE/'REFERENCE_CACHE_ACCESS_NEXT9.json',dict(status='PASS',timestamp=utc_now(),caches=caches,files_sha256=files,
            RP_series=6400,RP_row_ids_sha256=hashlib.sha256(bank.row_ids.tobytes()).hexdigest(),
            allowlist_only_these_four_existing_cache_files=True,break_labels_or_tau_decoded=False,Shadow_Seal_values_accessed=0,
            purpose='Exact early-only inner excluded BL80 reference, never a full-DEV baseline or an additional candidate'))
