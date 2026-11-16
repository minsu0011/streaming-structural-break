"""Report comparisons reject nonfinite values, loose relative tolerances and missing answers."""
import pytest
from src.next3.claim_checks import ClaimChecks, finite_number, numbered_answers, require_literal


@pytest.mark.parametrize('value', [True, False, float('nan'), float('inf'), '-inf', 'NaN'])
def test_non_numeric_or_nonfinite_claim_cannot_pass(value):
    with pytest.raises(ValueError): finite_number(value)


def test_absolute_tolerance_does_not_accept_large_relative_error():
    with pytest.raises(RuntimeError, match='differs'):
        ClaimChecks().numeric('memory', 1_000_000_001, 1_000_000_000, 'receipt.json', 'bytes', atol=0)


def test_exact_zero_contrast_is_preserved():
    checks = ClaimChecks(); checks.numeric('identity', '0.0', 0, 'paired.csv', 'selected/reference', atol=0)
    assert checks.claims[0]['absolute_difference'] == 0 and checks.claims[0]['stated'] == 0


def test_duplicate_claim_cannot_inflate_crosscheck_count():
    checks = ClaimChecks(); checks.numeric('same', 1, 1, 'receipt.json', 'a')
    with pytest.raises(ValueError, match='Duplicate'): checks.numeric('same', 1, 1, 'receipt.json', 'b')
    assert len(checks.claims) == 1


def test_failed_claim_is_not_added_to_successful_checks():
    checks = ClaimChecks()
    with pytest.raises(RuntimeError): checks.numeric('failure', 1, 2, 'receipt.json', 'a')
    assert checks.claims == []


def test_negative_tolerance_is_rejected():
    with pytest.raises(ValueError, match='Negative'):
        ClaimChecks().numeric('bad', 1, 1, 'receipt.json', 'a', atol=-1)


def test_numeric_comparison_requires_evidence_locator():
    with pytest.raises(ValueError, match='locator'): ClaimChecks().numeric('bad', 1, 1, '', 'a')


def test_numbered_answers_allow_multiline_explanation():
    text = '# Review\n\n1. **질문:** 첫 줄\n\n추가 근거\n\n2. **다음:** 답\n'
    result = numbered_answers(text, required=2)
    assert result[1]['question'] == '질문' and '추가 근거' in result[1]['answer']


@pytest.mark.parametrize('numbers', [[1, 3], [1, 1], [2, 1], [1], [1, 2, 3]])
def test_missing_duplicate_or_reordered_answers_fail(numbers):
    text = '\n'.join(f'{i}. **질문:** 답' for i in numbers)
    with pytest.raises(ValueError, match='numbered'): numbered_answers(text, required=2)


def test_empty_last_answer_is_rejected():
    with pytest.raises(ValueError, match='empty'): numbered_answers('1. **질문:** ', required=1)


@pytest.mark.parametrize('text', ['10.01', '0.011', '-0.01', '0.01e2'])
def test_decimal_substring_is_not_the_reported_number(text):
    with pytest.raises(RuntimeError, match='absent'): require_literal(text, '0.01')


def test_exact_number_inside_markdown_and_percent_is_found():
    require_literal('비율 **0.01**이며 26.35% 감소', '0.01')
    require_literal('비율 **0.01**이며 26.35% 감소', '26.35')
