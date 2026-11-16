import zipfile
import pytest
from src.next8.desktop_delivery import verify_twenty


def test_twenty_archive_must_match_exact_files_and_content(tmp_path):
    directory = tmp_path / 'handoff'
    directory.mkdir()
    paths = []
    for i in range(20):
        path = directory / f'{i:02d}_result.txt'
        path.write_text('actual result ' + str(i), encoding='utf-8')
        paths.append(path)
    archive = tmp_path / 'twenty.zip'
    with zipfile.ZipFile(archive, 'x') as stream:
        for p in paths:
            stream.write(p, p.name)
    assert verify_twenty(directory, archive)['exact_file_count'] == 20
    paths[3].write_text('changed after archive', encoding='utf-8')
    with pytest.raises(AssertionError):
        verify_twenty(directory, archive)
    extra = directory / 'unrequested_extra.txt'
    extra.write_text('extra', encoding='utf-8')
    with pytest.raises(AssertionError):
        verify_twenty(directory, archive)
