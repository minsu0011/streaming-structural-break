"""Record and fix a display-only null shading error. No numerical result changes."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.next7.common import BASE, save, digest, utc_now


def main():
    path = ROOT / 'src/next7/viewer.html'
    before = digest(path)
    html = path.read_text(encoding='utf-8')
    old = 'if(at<n){s+='
    new = "if(at<n&&!tr.case.startsWith('N')){s+="
    assert html.count(old) == 1
    path.write_text(html.replace(old, new), encoding='utf-8')
    save('visuals/V1_DISPLAY_ERRATUM.json', {'created_utc': utc_now(),
        'status': 'DISPLAY_ONLY_REPAIR_FOR_V2',
        'issue': 'The null synthetic generator retains a scheduled RNG split time even without a true change. V1 HTML incorrectly shaded that interval as if an intervention occurred.',
        'fix': 'No intervention shading for N0/N1/N2/N3. Existing PNG builder already excluded N0.',
        'source_before_sha256': before, 'source_after_sha256': digest(path),
        'G1_G2_numeric_outputs_changed': False, 'V1_files_preserved': True,
        'review_action': 'Add browser assertion that both null examples have no shaded rectangles; use revised V2 deliverable.'})


if __name__ == '__main__':
    main()
