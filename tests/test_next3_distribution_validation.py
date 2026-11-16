"""A partial or mismatched distribution cannot end the long-running session."""
import pytest
from src.next3.distribution_validation import distribution_checks


def proofs():
    sha = 'a'*64
    pointer = {'status':'PASS', 'archive_sha256':sha, 'all_member_sha256_verified':True,
        'crc_verified':True, 'actual_archive_replay':'PASS', 'documented_reviewer_command':'PASS',
        'raw_competition_data_included':False, 'credentials_included':False, 'all_seals_unopened':True}
    actual = {'status':'PASS', 'archive_sha256':sha, 'public_API_prediction_points':98372,
        'new_fits':0, 'seal_rows':0, 'raw_competition_observations_used':False}
    reviewer = {'status':'PASS', 'archive_sha256':sha, 'public_API_prediction_points':98372, 'new_fits':0, 'seal_rows':0}
    top20 = {'status':'PASS', 'archive_sha256':sha, 'file_count':20,
        'files': {str(i)+'.json':{} for i in range(20)}, '19_root_reports_equal_actual_archive_members':True,
        'original_project_files_moved':False, 'old_desktop_folders_modified':False}
    return [pointer, actual, reviewer, top20]


def test_complete_distribution_has_nonempty_literal_true_checks():
    checks = distribution_checks(*proofs())
    assert len(checks) == 11 and all(type(value) is bool and value for value in checks.values())


@pytest.mark.parametrize('index', range(4))
def test_each_receipt_must_pass(index):
    records = proofs(); records[index]['status'] = 'PENDING'
    with pytest.raises(RuntimeError): distribution_checks(*records)


@pytest.mark.parametrize('index', [1, 2, 3])
def test_every_receipt_must_refer_to_the_same_archive(index):
    records = proofs(); records[index]['archive_sha256'] = 'b'*64
    with pytest.raises(RuntimeError): distribution_checks(*records)


@pytest.mark.parametrize('counts', [(98371,98371), (98372,98371), (98371,98372), (True,98372), ('98372',98372)])
def test_equal_incomplete_counts_or_single_incomplete_replay_cannot_pass(counts):
    records = proofs(); records[1]['public_API_prediction_points'], records[2]['public_API_prediction_points'] = counts
    with pytest.raises(RuntimeError): distribution_checks(*records)


@pytest.mark.parametrize('key', ['all_member_sha256_verified', 'crc_verified', 'all_seals_unopened'])
def test_required_booleans_cannot_be_integer_truthiness(key):
    records = proofs(); records[0][key] = 1
    with pytest.raises(RuntimeError): distribution_checks(*records)


@pytest.mark.parametrize('key', ['raw_competition_data_included', 'credentials_included'])
def test_excluded_payload_claims_must_hold(key):
    records = proofs(); records[0][key] = True
    with pytest.raises(RuntimeError): distribution_checks(*records)


def test_top20_count_cannot_disagree_with_file_entries():
    records = proofs(); records[3]['files'].pop('0.json')
    with pytest.raises(RuntimeError): distribution_checks(*records)


@pytest.mark.parametrize('key', ['original_project_files_moved', 'old_desktop_folders_modified'])
def test_distribution_must_preserve_prior_files(key):
    records = proofs(); records[3][key] = True
    with pytest.raises(RuntimeError): distribution_checks(*records)


@pytest.mark.parametrize('index,key,value', [(1,'new_fits',1),(2,'seal_rows',1),(1,'new_fits',False), (2,'new_fits','0')])
def test_replay_zero_fit_seal_counts_are_literal_integers(index, key, value):
    records = proofs(); records[index][key] = value
    with pytest.raises(RuntimeError): distribution_checks(*records)
