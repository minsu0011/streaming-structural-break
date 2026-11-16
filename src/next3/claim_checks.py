"""Strict comparison and numbered-answer parsing for verifiable report claims."""
import math
import re


def finite_number(value):
    if isinstance(value, bool): raise ValueError('A boolean is not a numeric evidence value')
    result = float(value)
    if not math.isfinite(result): raise ValueError('Numeric evidence must be finite')
    return result


class ClaimChecks:
    def __init__(self): self.claims = []; self._names = set()

    def numeric(self, name, stated, observed, source, location, *, atol=1e-14):
        if name in self._names: raise ValueError('Duplicate numeric claim name: '+name)
        left = finite_number(stated); right = finite_number(observed); tolerance = finite_number(atol)
        if tolerance < 0: raise ValueError('Negative claim tolerance')
        if not source or not location: raise ValueError('A numeric claim needs a precise evidence locator')
        difference = abs(left-right)
        if difference > tolerance:
            raise RuntimeError(f'Numeric report claim differs: {name}: {left!r} != {right!r}; tolerance {tolerance!r}')
        self._names.add(name)
        self.claims.append({'claim': name, 'stated': left, 'observed': right,
            'absolute_difference': difference, 'absolute_tolerance': tolerance,
            'source': source, 'location': location})


def numbered_answers(text, *, required=44):
    matches = list(re.finditer(r'(?m)^(\d+)\. \*\*([^*\n]+)\*\* ', text))
    if [int(match.group(1)) for match in matches] != list(range(1, required+1)):
        raise ValueError('Required numbered Korean answers are missing, duplicated or reordered')
    result = {}
    for index, match in enumerate(matches):
        end = matches[index+1].start() if index+1 < len(matches) else len(text)
        body = text[match.end():end].strip()
        if not body: raise ValueError('An answer is empty')
        result[index+1] = {'question': match.group(2).rstrip(':'), 'answer': body}
    return result


def require_literal(answer, literal):
    token = str(literal)
    if not re.search(r'(?<![0-9.eE+-])'+re.escape(token)+r'(?![0-9.eE+-])', answer):
        raise RuntimeError('Expected literal is absent from its numbered answer: '+token)
