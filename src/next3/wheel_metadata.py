"""Read a wheel's distribution metadata without mistaking vendored packages for it."""
from email.parser import BytesParser
import hashlib
import zipfile
from packaging.utils import canonicalize_name
from packaging.version import Version


def distribution_metadata(path, expected_name, expected_version):
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None: raise ValueError('Wheel ZIP CRC failed')
        # Wheels can legitimately contain vendored *.dist-info directories.
        # Only the top-level dist-info is the installed wheel's identity.
        candidates = [name for name in archive.namelist()
                      if name.endswith('.dist-info/METADATA') and name.count('/') == 1]
        if len(candidates) != 1: raise ValueError('Exactly one top-level wheel METADATA is required')
        relative = candidates[0]; folder = relative.split('/')[0][:-len('.dist-info')]
        folder_name, folder_version = folder.rsplit('-', 1)
        if canonicalize_name(folder_name) != canonicalize_name(expected_name) or Version(folder_version) != Version(expected_version):
            raise ValueError('Wheel metadata directory has the wrong distribution/version')
        content = archive.read(relative); metadata = BytesParser().parsebytes(content)
        names, versions = metadata.get_all('Name', []), metadata.get_all('Version', [])
        if len(names) != 1 or len(versions) != 1:
            raise ValueError('Unambiguous distribution Name and Version are required')
        if canonicalize_name(names[0]) != canonicalize_name(expected_name) or versions[0] != expected_version:
            raise ValueError('Wheel METADATA and exact pins disagree')
        return {'relative_path': relative, 'distribution': canonicalize_name(names[0]),
            'version': versions[0], 'sha256': hashlib.sha256(content).hexdigest(), 'zip_CRC_pass': True,
            'nested_vendored_metadata_files': sum(name.endswith('.dist-info/METADATA') and name.count('/') > 1
                                                 for name in archive.namelist())}
