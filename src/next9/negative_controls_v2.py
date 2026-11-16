"""Fixed paired synthetic observation interventions, isolated from RP fitting."""
import hashlib
import numpy as np

BACKGROUNDS=('GAUSSIAN','AR_0.8','ARCH','STUDENT_T5')
INTERVENTIONS=('NONE','MEAN_SHIFT','SCALE_UP','SCALE_DOWN','DEPENDENCE_SHIFT','SINGLE_SHOCK','REPEATED_SHOCK')
SEEDS=(9101604,9101605,9101606)
H_LENGTHS=(512,2048)


def generate(background,intervention,seed,h_length):
    if background not in BACKGROUNDS or intervention not in INTERVENTIONS or seed not in SEEDS or h_length not in H_LENGTHS:
        raise ValueError('Outside fixed synthetic menu')
    key=hashlib.sha256(f'NEXT9_CONTROL|{background}|{seed}|{h_length}'.encode()).hexdigest()
    rng=np.random.default_rng(int(key[:16],16))
    burn=512;count=burn+h_length+512
    noise=rng.standard_t(5,count)*np.sqrt(3/5) if background=='STUDENT_T5' else rng.normal(size=count)
    onset=64+48*((seed+h_length)%6)
    latent=np.empty(count);previous=0.;observed=np.empty(count)
    for k in range(count):
        t=k-burn-h_length
        phi=.8 if background=='AR_0.8' else 0.
        if intervention=='DEPENDENCE_SHIFT' and t>=onset:phi=-.5
        if background=='ARCH':
            innovation=noise[k]*np.sqrt(arch_innovation_variance(phi,previous))
        else:innovation=noise[k]*np.sqrt(1-phi**2)
        value=phi*previous+innovation;latent[k]=value;previous=value
        # Observation shocks are deliberately not inserted into the latent
        # recursion: delayed feature responses can expose filter memory.
        if t>=onset:
            if intervention=='MEAN_SHIFT':value+=1.
            elif intervention=='SCALE_UP':value*=2.
            elif intervention=='SCALE_DOWN':value*=.5
            elif intervention=='SINGLE_SHOCK' and t==onset:value+=8.
            elif intervention=='REPEATED_SHOCK' and (t-onset)%32==0:value+=8.*(-1 if ((t-onset)//32)%2 else 1)
        observed[k]=value
    values=observed[burn:].astype(np.float32)
    return values[:h_length],values[h_length:],dict(background=background,intervention=intervention,seed=seed,H_length=h_length,
        known_synthetic_onset=onset,online_length=512,paired_noise_key_sha256=key,
        structural_label_is_generator_metadata_only=True,observation_shocks_do_not_change_latent_recursion=True)


def arch_innovation_variance(phi,previous):
    return (1-phi**2)*(.1+.85*previous**2)
