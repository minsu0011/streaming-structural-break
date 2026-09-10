"""Unchanged locked recurrence, crossed background diagnostic in distinct RNG partition 3."""
import copy
from src.next2.counterexamples import generate

FACTORS=(1.5,2.,.7,.5)
BACKGROUNDS=('N0','N1','N2','N3')


def crossed_lock(original,factor):
    if factor not in FACTORS:
        raise ValueError('Variance multiplier outside fixed cross-background menu')
    lock=copy.deepcopy(original)
    specs={s['id']:s for s in original['scenarios']}
    lock['scenarios']=[]
    for background in BACKGROUNDS:
        spec=copy.deepcopy(specs[background])
        spec.update(id='CB_'+background,variance_factor=factor,
            phi_after=spec['phi_before'],arch_after=spec['arch_before'],t_df_after=spec['t_df_before'],mean_shift_stationary_sd=0.)
        lock['scenarios'].append(spec)
    return lock


def series(original,background,factor,index,*,control=False,horizon=None):
    if background not in BACKGROUNDS:
        raise ValueError('Background outside fixed menu')
    return generate('CB_'+background,index,partition=3,control=control,horizon=horizon,
                    lock=crossed_lock(original,factor))
