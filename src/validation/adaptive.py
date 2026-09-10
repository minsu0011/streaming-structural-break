"""Finite hypotheses, evidence-dependent ordering, and a persisted wall-clock budget."""
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
from src.utils.artifacts import write_json, utc_now

MODELS=('M1_logistic','M2_ridge','M3_histgb','M4_lightgbm','M5_xgboost')
FAMILIES=('A','B','C','D','AB','AD','BD','ABD','ABCD','ABCDEFQ')

@dataclass(frozen=True)
class Candidate:
    model: str
    families: str = 'ABCD'
    sampling: str = 'S0'
    params: dict = field(default_factory=dict)
    feature_changes: dict = field(default_factory=dict)
    hypothesis: str = 'Establish the ordinary-row supervised reference.'
    phase: str = 'ordinary_ladder'
    representation: str = 'base'

    @property
    def identity(self):
        values=asdict(self);values.pop('hypothesis');values.pop('phase')
        if values['representation']=='base':values.pop('representation')
        return hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()[:12]

    @property
    def experiment_id(self):
        return f'{self.model}_{self.families}_{self.sampling}_{self.identity}'

class ResearchBudget:
    def __init__(self,path,hours=10,buffer_minutes=45,start=None):
        self.path=Path(path)
        if self.path.exists():
            self.state=json.loads(self.path.read_text(encoding='utf-8'))
        else:
            beginning=datetime.fromisoformat(start) if start else datetime.now(timezone.utc)
            deadline=beginning+timedelta(hours=hours)
            self.state={'start_utc':beginning.isoformat(),'deadline_utc':deadline.isoformat(),
                'new_experiment_deadline_utc':(deadline-timedelta(minutes=buffer_minutes)).isoformat(),
                'budget_hours':hours,'final_buffer_minutes':buffer_minutes,'status':'ACTIVE'}
            write_json(self.path,self.state)

    def remaining_experiment_seconds(self,now=None):
        now=now or datetime.now(timezone.utc)
        return (datetime.fromisoformat(self.state['new_experiment_deadline_utc'])-now).total_seconds()

    def can_start(self,estimated_seconds=0,now=None):
        return not self.state.get('model_research_closed',False) and self.remaining_experiment_seconds(now)>max(estimated_seconds,0)

    def checkpoint(self,**values):
        self.state.update(values);self.state['updated_utc']=utc_now();write_json(self.path,self.state)

def select_next(results,attempted,primary_id=None):
    """Regenerate a rational queue from current CV evidence after every candidate.

    No stochastic sweep and no repetition of an identical configuration. The
    caller continues to robustness work when this bounded hypothesis set ends.
    """
    queue=[Candidate(model) for model in MODELS]
    supervised=[r for r in results if r.get('candidate') and r.get('status')=='COMPLETE']
    if supervised:
        champion=next((r for r in supervised if r['experiment_id']==primary_id),None) if primary_id else None
        if champion is None:champion=max(supervised,key=lambda r:(r['median'],-r['std']))
        if champion['candidate'].get('representation')=='cross_order_ar4_ar8_equal':
            # The equal blend has exactly one frozen formula. Continue bounded
            # component hypotheses without inventing blend parameter tuning.
            champion=next(r for r in supervised if r['experiment_id']==champion['components'][0])
        spec=Candidate(**champion['candidate'])
        tree_results=[r for r in supervised if r['model'] in MODELS[2:]]
        best_tree=max(tree_results,key=lambda r:r['median'])['model'] if tree_results else 'M4_lightgbm'
        for model in dict.fromkeys(('M1_logistic',best_tree)):
            for sampling in ('S1','S2','S3','S3_sqrt'):
                queue.append(Candidate(model,sampling=sampling,phase='sampling_weighting',
                    hypothesis={'S1':'Cap series contribution independently of outcomes.',
                        'S2':'Balance training coverage across fixed online-time strata.',
                        'S3':'Align training time/class weight with within-time positive-negative pair contribution.',
                        'S3_sqrt':'Temper pair contribution weights using the predeclared square-root family.'}[sampling]))
        for model in dict.fromkeys(('M1_logistic',best_tree)):
            for family in FAMILIES:
                queue.append(Candidate(model,families=family,sampling='S1',phase='family_ablation',
                    hypothesis=f'Estimate standalone or marginal {family} contribution under the same capped sampling.'))
        base={k:v for k,v in asdict(spec).items() if k not in ('hypothesis','phase')}
        # A single prospectively recorded follow-up: the mean/SD AR4 candidate
        # improved median but missed the worst-fold gate. Keep its representation
        # and change only the small tree learner. The score gates do not change.
        followup_path=Path(__file__).resolve().parents[2]/'configs/ar_mean_std_histgb_followup.json'
        if followup_path.exists():
            followup=json.loads(followup_path.read_text(encoding='utf-8'))
            if any(r.get('experiment_id')==followup['anchor'] for r in supervised):
                fixed=Candidate(**followup['candidate'])
                if fixed.identity!=followup['identity'] or followup['maximum_candidates']!=1:raise RuntimeError('Prospective model follow-up policy changed')
                queue.append(fixed)
        for targeted_path in sorted((Path(__file__).resolve().parents[2]/'configs').glob('targeted_followups*.json')):
            targeted=json.loads(targeted_path.read_text(encoding='utf-8'))
            if len(targeted['entries'])!=targeted['maximum_new_candidates']:raise RuntimeError('Targeted follow-up family changed')
            for entry in targeted['entries']:
                if any(r.get('experiment_id')==entry['anchor'] for r in supervised):
                    fixed=Candidate(**entry['candidate'])
                    if fixed.identity!=entry['identity']:raise RuntimeError('Targeted candidate identity changed')
                    queue.append(fixed)
        if spec.representation!='base':
            for model in MODELS:
                queue.append(Candidate(**{**base,'model':model,'params':{}},phase='calibrated_model_refinement',
                    hypothesis=f'Change only model family to {model} after historical calibration; retain representation, sampling and feature families.'))
            calibrated_families=('A','B','C','D','AB','AD','BD','ABD','BCD','ABCDEFQ') if spec.representation.startswith(('ar_residual_input_', 'ar_order2_', 'ar_order8_')) else ('D','BD','BCD','ABCDEFQ')
            for family in calibrated_families:
                queue.append(Candidate(**{**base,'families':family},phase='calibrated_family_refinement',
                    hypothesis=f'Test {family} with the current historical-input representation; removing predictable dependence can change standalone and marginal family contributions.'))
            for sampling in ('S0','S1','S2','S3','S3_sqrt'):
                queue.append(Candidate(**{**base,'sampling':sampling},phase='calibrated_sampling_refinement',
                    hypothesis=f'Change only {sampling} after improving historical feature comparability.'))
            from src.models.representations import ar_prefix,calibration_policy
            input_prefix=ar_prefix(spec.representation)
            current_calibration=calibration_policy(spec.representation);method=current_calibration.method
            other=input_prefix+('historical_median_mad' if method=='historical_mean_std' else 'historical_mean_std')+('_clip24' if current_calibration.clip==24.0 else '')
            queue.append(Candidate(**{**base,'representation':other},phase='robust_historical_calibration',
                hypothesis='Test the predeclared median/MAD versus mean/SD historical null normalization under heavy tails.'))
            if spec.representation in ('historical_mean_std','historical_median_mad'):
                queue.append(Candidate(**{**base,'representation':spec.representation+'_startup'},phase='startup_refinement',
                    hypothesis='Early online-time discrimination remains weak. Change only finite-prefix EWMA variance normalization on the current promoted representation.'))
        # At most one configuration field changes per refinement; scales form one mechanism.
        time_profile=champion.get('time_profile',{})
        weakest_time=min(time_profile,key=lambda k:time_profile[k]) if time_profile else 'not_yet_summarized'
        refinements=[('scales',(3,12,48)),('scales',(8,32,128)),('scales',(5,20,160)),
            ('normalization','median_mad'),('clip_z',10.0),('clip_z',40.0),
            ('cusum_drift',0.1),('cusum_drift',0.5),('lags',(1,3,6,12)),('lags',(1,4,16,32)),('lags',(1,8,32,64)),
            ('tail_thresholds',(1.5,2.5)),('memory_decay',0.97),('memory_decay',0.995)]
        for key,value in refinements:
            affected={'cusum_drift':'A','lags':'DEF','tail_thresholds':'C','memory_decay':'F'}.get(key,'ABCDEF')
            if not set(spec.families)&set(affected):continue
            changes=dict(spec.feature_changes);changes[key]=value
            queue.append(Candidate(**{**base,'feature_changes':changes},phase='feature_refinement',
                hypothesis=f'Current median {champion["median"]:.6f}, worst fold {champion["worst_fold"]:.6f}, weakest time bucket {weakest_time}: test only {key}={value}.'))
        # Decompose an observed ALL-family gain before attributing it to either
        # cross-scale or memory features. Keep sampling fixed for the comparison.
        all_results=[r for r in supervised if r.get('families')=='ABCDEFQ']
        for all_result in all_results:
            reference=all_result['candidate']
            reference_base={k:v for k,v in reference.items() if k not in ('hypothesis','phase')}
            for family in ('ABCDE','ABCDF','E','F','EF'):
                queue.append(Candidate(**{**reference_base,'families':family},phase='memory_crossscale_ablation',
                    hypothesis=f'Decompose cross-scale versus memory contribution after ALL-family CV; retain all other settings from {all_result["experiment_id"]}.'))
        if spec.model in ('M3_histgb','M4_lightgbm','M5_xgboost'):
            settings={'max_depth':[3,5], 'learning_rate':[.03,.08]}
            if spec.model=='M3_histgb':settings.update(max_iter=[120,200],max_leaf_nodes=[7,31],l2_regularization=[30,100],min_samples_leaf=[100,300])
            elif spec.model=='M4_lightgbm':settings.update(n_estimators=[50,150,250],num_leaves=[7,31],reg_lambda=[30,100],min_child_samples=[100,300])
            else:settings.update(n_estimators=[50,150,250],reg_lambda=[30,100],min_child_weight=[100,300])
        else:settings={'C':[.01,1,10]} if spec.model=='M1_logistic' else {'alpha':[10,1000]}
        for key,values in settings.items():
            for value in values:
                params=dict(spec.params);params[key]=value
                queue.append(Candidate(**{**base,'params':params},phase='shallow_refinement',
                    hypothesis=f'One-parameter bias/variance or latency refinement: {key}={value}; weakest fold assessed before promotion.'))
    for candidate in queue:
        if candidate.identity not in attempted:return candidate
    return None

def promotion_decision(candidate,champion):
    if champion is None:return {'decision':'PROMOTE','reason':'First fully validated development candidate'}
    median_gain=candidate['median']-champion['median']
    worst_loss=champion['worst_fold']-candidate['worst_fold']
    ratio=candidate['latency_us']/max(champion['latency_us'],1e-9)
    accepted=median_gain>=.001 and worst_loss<=.002 and (ratio<=5 or median_gain>=.01)
    return {'decision':'PROMOTE' if accepted else 'RETAIN_AS_EVIDENCE',
        'median_gain':median_gain,'worst_fold_loss':worst_loss,'latency_multiplier':ratio,
        'reason':'Frozen gates: median +0.001, worst loss <=0.002; >5x latency requires +0.01 median'}
