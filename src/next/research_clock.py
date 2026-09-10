"""Persistent wall-clock/monotonic research and additional-audit gates.

The original session UTC start remains authoritative. The monotonic anchor was
added during ongoing research; its retrospective origin is explicitly recorded.
"""
from datetime import datetime,timezone
from pathlib import Path
import json
import time
from src.utils.artifacts import write_json,sha256,utc_now


class ResearchTimeError(RuntimeError):
    pass


class ResearchClock:
    def __init__(self,root):
        self.root = Path(root)
        self.session_path = self.root/'artifacts/next/SESSION_NEXT.json'
        self.session = json.loads(self.session_path.read_text(encoding='utf-8'))
        self.directory = self.root/'artifacts/next/scheduler'
        self.directory.mkdir(parents=True,exist_ok=True)
        self.anchor_path = self.directory/'MONOTONIC_ANCHOR.json'
        start = datetime.fromisoformat(self.session['research_start_utc'])
        if self.anchor_path.exists():
            self.anchor = json.loads(self.anchor_path.read_text(encoding='utf-8'))
            if self.anchor['session_sha256']!=sha256(self.session_path):
                raise ResearchTimeError('Original research session changed')
        else:
            now = datetime.now(timezone.utc)
            prior_elapsed = (now-start).total_seconds()
            if prior_elapsed<0:
                raise ResearchTimeError('Research start is in the future')
            self.anchor = {'status':'LOCKED','created_utc':now.isoformat(),
                'session_sha256':sha256(self.session_path),'original_start_utc':start.isoformat(),
                'prior_elapsed_from_original_utc_seconds':prior_elapsed,
                'monotonic_origin':time.monotonic()-prior_elapsed,
                'provenance':'Added during uninterrupted ongoing research. Earlier elapsed time is reconstructed from the already frozen original UTC start, not claimed to be an originally captured monotonic sample.',
                'clock_drift_tolerance_seconds':120.}
            write_json(self.anchor_path,self.anchor)
        self.snapshot()

    def snapshot(self):
        elapsed = time.monotonic()-self.anchor['monotonic_origin']
        wall = (datetime.now(timezone.utc)-datetime.fromisoformat(self.session['research_start_utc'])).total_seconds()
        if elapsed<0 or abs(elapsed-wall)>self.anchor['clock_drift_tolerance_seconds']:
            raise ResearchTimeError('Monotonic/UTC disagreement or reboot requires an explicit clock audit; the minimum must not reset silently')
        result = {'created_utc':utc_now(),'research_elapsed_seconds':elapsed,
            'minimum_research_seconds':self.session['minimum_research_seconds'],
            'minimum_research_met':elapsed>=self.session['minimum_research_seconds'],
            'remaining_research_seconds':max(0.,self.session['minimum_research_seconds']-elapsed),
            'final_audit_started':False,'final_audit_elapsed_seconds':0.,'handoff_time_gate_pass':False}
        audit_path = self.directory/'FINAL_AUDIT_START.json'
        if audit_path.exists():
            audit = json.loads(audit_path.read_text(encoding='utf-8'))
            if audit['anchor_sha256']!=sha256(self.anchor_path) or audit['research_elapsed_at_start']<self.session['minimum_research_seconds']:
                raise ResearchTimeError('Invalid additional-audit start')
            audit_elapsed = time.monotonic()-audit['monotonic_start']
            audit_wall = (datetime.now(timezone.utc)-datetime.fromisoformat(audit['created_utc'])).total_seconds()
            if audit_elapsed<0 or abs(audit_elapsed-audit_wall)>120:
                raise ResearchTimeError('Additional-audit clock changed')
            result.update(final_audit_started=True,final_audit_elapsed_seconds=audit_elapsed,
                handoff_time_gate_pass=bool(result['minimum_research_met'] and audit_elapsed>=self.session['additional_final_audit_min_seconds']))
        return result

    def begin_final_audit(self):
        snapshot = self.snapshot()
        if not snapshot['minimum_research_met']:
            raise ResearchTimeError('Ten hours of research must elapse before the additional audit begins')
        path = self.directory/'FINAL_AUDIT_START.json'
        if not path.exists():
            write_json(path,{'created_utc':utc_now(),'monotonic_start':time.monotonic(),
                'research_elapsed_at_start':snapshot['research_elapsed_seconds'],'anchor_sha256':sha256(self.anchor_path),
                'minimum_additional_audit_seconds':self.session['additional_final_audit_min_seconds']})
        return self.snapshot()

    def assert_handoff_ready(self):
        snapshot = self.snapshot()
        if not snapshot['handoff_time_gate_pass']:
            raise ResearchTimeError('Final handoff requires ten research hours followed by at least 45 additional audit minutes')
        return snapshot
