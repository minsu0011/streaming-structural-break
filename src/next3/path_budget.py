"""Explicit legacy Win32 path budgets for unchanged archive member names."""
from pathlib import Path
from src.next3.archive_contract import member_name


def utf16_units(value):
    return len(str(value).encode('utf-16-le'))//2


def check_windows_copy_path_budget(destination, members):
    """Check before creating a copy; do not enable long paths or rename evidence."""
    root = Path(destination).absolute()
    maximum_file = 0; maximum_directory = utf16_units(root)
    longest_member = None; count = 0
    for name in members:
        relative = member_name(name)
        target = root.joinpath(*relative.parts)
        file_units = utf16_units(target); directory_units = utf16_units(target.parent)
        if file_units > maximum_file: maximum_file = file_units; longest_member = name
        maximum_directory = max(maximum_directory,directory_units); count += 1
        if file_units > 259 or directory_units > 247:
            raise ValueError(f'Windows copy path budget exceeded before copying: member={name}; '
                f'file_UTF16_units={file_units}, maximum=259; directory_UTF16_units={directory_units}, maximum=247. '
                'Use a shorter ordinary staging/extraction root; preserve member names and bytes.')
    if maximum_directory > 247:
        raise ValueError('Windows staging root exceeds the directory path budget before copying')
    return {'status':'PASS','destination':str(root),'members_checked':count,
        'maximum_file_UTF16_units':maximum_file,'maximum_directory_UTF16_units':maximum_directory,
        'longest_member':longest_member,'maximum_file_units_allowed':259,'maximum_directory_units_allowed':247,
        'scope':'Conservative legacy Win32 absolute path budget; no extended-path or arbitrary extraction-depth guarantee.'}
