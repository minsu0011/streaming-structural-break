"""Optional successful-job prerequisites, separate from artifact existence."""
from pathlib import Path
import json
import re

IDENTIFIER = re.compile(r'[A-Za-z0-9_-]+\Z')


def validate_dependencies(job, base):
    dependencies = job.get('depends_on_success', [])
    if not isinstance(dependencies, list) or any(not isinstance(value, str) for value in dependencies):
        raise ValueError('Successful-job prerequisites must be an explicit identifier list')
    if len(dependencies) != len(set(dependencies)):
        raise ValueError('Duplicate successful-job prerequisite')
    for name in dependencies:
        if not IDENTIFIER.fullmatch(name) or name == job['id']:
            raise ValueError('Unsafe or self-referential successful-job prerequisite')
        if not (Path(base)/'queue'/f'{name}.json').is_file():
            raise ValueError('Register each prerequisite before its dependent job: '+name)
    # Existing queue entries are immutable. Requiring earlier registrations
    # means the dependency graph cannot acquire a forward edge or a cycle.
    return tuple(dependencies)


def dependency_observations(job, base):
    observations = []
    for name in job.get('depends_on_success', []):
        if not isinstance(name, str) or not IDENTIFIER.fullmatch(name) or name == job.get('id'):
            raise ValueError('Invalid persisted successful-job prerequisite')
        path = Path(base)/'jobs'/name/'STATE.json'
        if not path.is_file():
            observations.append({'id': name, 'status': 'PENDING', 'successful': False,
                                 'reason': 'No completed worker attempt'})
            continue
        record = json.loads(path.read_text(encoding='utf-8-sig'))
        if record.get('id') != name:
            raise ValueError('Dependency state identity differs from its directory')
        successful = record.get('status') == 'COMPLETE' and type(record.get('exit_code')) is int and record['exit_code'] == 0
        observations.append({'id': name, 'status': record.get('status'), 'successful': successful,
                             'reason': 'Completed worker exited zero' if successful else 'Prerequisite has not completed successfully'})
    return observations


def prerequisites_succeeded(job, base):
    return all(item['successful'] for item in dependency_observations(job, base))
