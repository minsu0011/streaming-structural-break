"""Fail-closed proof joining before declaring the final distribution complete."""
import re
from src.next3.final_validation import all_true


def exact_count(value, expected):
    return type(value) is int and value == expected


def distribution_checks(pointer, actual, reviewer, top20):
    sha = pointer.get('archive_sha256')
    valid_sha = isinstance(sha, str) and re.fullmatch('[0-9a-f]{64}', sha) is not None
    files = top20.get('files')
    checks = {
        'all_four_distribution_receipts_pass': all(record.get('status') == 'PASS' for record in (pointer, actual, reviewer, top20)),
        'same_actual_archive_in_all_distribution_receipts': valid_sha and all(record.get('archive_sha256') == sha for record in (actual, reviewer, top20)),
        'all_archive_member_SHA_and_CRC_verified': pointer.get('all_member_sha256_verified') is True and pointer.get('crc_verified') is True,
        'actual_archive_replay_and_documented_command_pass': pointer.get('actual_archive_replay') == pointer.get('documented_reviewer_command') == 'PASS',
        'both_replays_have_exactly_98372_outputs': exact_count(actual.get('public_API_prediction_points'), 98372) and exact_count(reviewer.get('public_API_prediction_points'), 98372),
        'both_replays_have_zero_fits_and_seal_rows': all(exact_count(record.get(key), 0) for record in (actual, reviewer) for key in ('new_fits', 'seal_rows')),
        'archive_raw_credentials_excluded_and_seals_closed': pointer.get('raw_competition_data_included') is False and pointer.get('credentials_included') is False and pointer.get('all_seals_unopened') is True,
        'actual_replay_did_not_read_competition_observations': actual.get('raw_competition_observations_used') is False,
        'top20_exact_count_and_distinct_file_entries': exact_count(top20.get('file_count'), 20) and isinstance(files, dict) and len(files) == 20,
        'top20_root_reports_match_actual_archive': top20.get('19_root_reports_equal_actual_archive_members') is True,
        'top20_used_copies_and_preserved_existing_desktop_folders': top20.get('original_project_files_moved') is False and top20.get('old_desktop_folders_modified') is False,
    }
    all_true(checks)
    return checks
