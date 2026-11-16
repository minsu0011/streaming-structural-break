"""Show every completed fixed classifier scenario without merging alarm endpoints."""
from pathlib import Path
import csv
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.next7.common import BASE,read_json,save,digest,utc_now,activity,note

METHODS=('M0','M3','M_EVENT','G2_EVENT','I4_EVENT_SCALE_TREE','G2_FULL','G3_PAIRWISE_FULL','AMPLITUDE_0_9_M0')
LABELS=('BL80','NEXT6 M3','G1 event','G2 event','I4','G2 full','G3','0.9 × BL80')
NEGATIVE=('N0','N1','N2','N3','O1','OB16','REPEATED_SHOCK')
CASE_LABELS={
    'N0':'N0 · Gaussian 정상','N1':'N1 · AR 정상','N2':'N2 · ARCH 정상','N3':'N3 · heavy-tail 정상',
    'O1':'O1 · 한 번의 충격','OB16':'OB16 · 16개 연속 충격','REPEATED_SHOCK':'반복 충격 · 128개 간격 네 번',
    'M1':'M1 · 평균 변화','M2':'M2 · 평균 변화','V1':'V1 · scale 확대','V2':'V2 · scale 확대',
    'V3':'V3 · scale 축소','V4':'V4 · scale 축소','A1':'A1 · ARCH 증가','A2':'A2 · ARCH 감소',
    'D1':'D1 · AR 의존성 증가','D2':'D2 · AR 부호 전환','T1':'T1 · Gaussian → t3','T2':'T2 · t3 → Gaussian',
    'TEMP_SCALE_PULSE':'32개 관측의 분산 확대 후 회복','RETURN_NORMAL':'256개 관측의 분산 확대 후 회복',
    'LATE_PERSISTENT':'늦은 지속 변화 · 시작 1,536','SCALE_PLUS_SHOCK':'분산 확대 + 변화 후 충격'}


def main():
    protocol_path=BASE/'CLASSIFIER_STRESS_PROTOCOL_LOCK.json'
    protocol=read_json(protocol_path)
    assert protocol['status']=='FROZEN'
    cases_all=tuple(protocol['evaluation']['cases'])
    assert len(cases_all)==23 and set(cases_all)==set(CASE_LABELS)
    receipt=read_json(BASE/'COMPLETE_DIAGNOSTICS_CLASSIFIERS_NEXT7.json')
    assert receipt['status']=='COMPLETE' and not receipt['partial_cohorts_included']
    source=BASE/'CLASSIFIER_STRESS_RESULTS_NEXT7.csv'
    assert digest(source)==receipt['outputs_sha256'][source.name]
    with source.open(newline='',encoding='utf-8-sig') as handle: rows=list(csv.DictReader(handle))
    assert set(r['case'] for r in rows)==set(cases_all)
    keyed={(r['case'],r['method'],int(r['horizon']),r['metric']):r for r in rows}
    assert len(keyed)==len(rows)
    directory=BASE/'visuals/classifier_stress/full_suite_001'
    assert not directory.exists(),'Keep every completed figure snapshot'
    positive=tuple(case for case in cases_all if case not in NEGATIVE)
    assert len(positive)==16 and len(NEGATIVE)==7 and len(METHODS)==8
    selected=[]
    matrices={}
    for horizon in (1024,4096):
        matrices[horizon]=[]
        for cases,metric in ((NEGATIVE,'any_alarm'),(positive,'postbreak_first_detection')):
            matrix=np.full((len(cases),len(METHODS)),np.nan)
            for i,case in enumerate(cases):
                for j,method in enumerate(METHODS):
                    row=keyed[case,method,horizon,metric]
                    value=row['mean_across_fixed_folds']
                    missing=value in ('',None)
                    assert missing==(case=='LATE_PERSISTENT' and horizon==1024)
                    if not missing:
                        matrix[i,j]=float(value)
                        assert 0<=matrix[i,j]<=1
                    selected.append({'case':case,'method':method,'horizon':horizon,'metric':metric,
                        'value':None if missing else float(value),'source_CI_low':row['CI_low'],'source_CI_high':row['CI_high']})
            np.testing.assert_array_equal(matrix[:,0],matrix[:,-1])
            matrices[horizon].append(matrix)
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':11,
        'figure.facecolor':'#f7f9fa','savefig.facecolor':'#f7f9fa','svg.fonttype':'none'})
    directory.mkdir(parents=True)
    with activity('Plot all23 completed classifier scenarios with separate endpoints','SCIENTIFIC_VISUALIZATION','F5_CAUSAL_NESTED_INTEGRATION'):
        for horizon in (1024,4096):
            fig,axes=plt.subplots(2,1,figsize=(16,17),gridspec_kw={'height_ratios':[7,16]})
            fig.suptitle(f'분류기의 분별력 · 기존 23개 시나리오 전체 · 관측 {horizon:,}개',
                x=.04,ha='left',fontsize=21,weight='bold',color='#193d49')
            fig.text(.04,.945,'거짓 경보는 낮을수록, 실제 변화의 첫 검출은 높을수록 좋습니다. 두 지표를 하나의 점수로 합치지 않았습니다.',
                fontsize=11,color='#4d6672')
            settings=((NEGATIVE,'오경보율 · 한 번이라도 경보한 경로 · 낮을수록 좋음','Reds'),
                      (positive,'실제 변화의 첫 검출률 · 변화 전 경보 없이 이후 첫 경보 · 높을수록 좋음','GnBu'))
            for ax,matrix,(cases,title,cmap_name) in zip(axes,matrices[horizon],settings):
                cmap=plt.get_cmap(cmap_name).copy();cmap.set_bad('#dde3e6')
                plot=ax.imshow(matrix,cmap=cmap,vmin=0,vmax=1,aspect='auto')
                ax.set_xticks(range(len(METHODS)),LABELS)
                ax.set_yticks(range(len(cases)),[CASE_LABELS[case] for case in cases])
                ax.set_title(title,loc='left',fontsize=14,pad=14,color='#193d49')
                ax.set_xticks(np.arange(-.5,len(METHODS),1),minor=True)
                ax.set_yticks(np.arange(-.5,len(cases),1),minor=True)
                ax.grid(which='minor',color='white',linewidth=1.2)
                ax.tick_params(which='minor',bottom=False,left=False)
                for (i,j),value in np.ndenumerate(matrix):
                    ax.text(j,i,'평가 불가' if not np.isfinite(value) else f'{value:.1%}',ha='center',va='center',
                        color='white' if np.isfinite(value) and value>.65 else '#263f4c',fontsize=10)
                colorbar=fig.colorbar(plot,ax=ax,fraction=.025,pad=.02)
                colorbar.set_ticks([0,.25,.5,.75,1],labels=['0%','25%','50%','75%','100%'])
            fig.text(.04,.024,
                '원래 512개 시드 · 고정된 5개 fold 모델의 평균. 모델별 원래 H-only 경보 기준을 그대로 적용했습니다.\n'
                '정상·충격은 전체 경보율이며 사건 기억률이나 대조에만 없는 경보율이 아닙니다. 일시 변화도 처음 변한 뒤 정답은 계속 1입니다.\n'
                '1,024시점에는 늦은 변화가 아직 시작되지 않아 첫 검출을 평가하지 않습니다. 회색은 0%가 아닙니다.\n'
                '원래 2,000회 대응 구간과 전체 방법·지표는 CLASSIFIER_STRESS_RESULTS_NEXT7.csv / CLASSIFIER_STRESS_PAIRED_NEXT7.csv에 있습니다.\n'
                '색과 점추정만으로 유의성·인과효과를 판단하지 않습니다. 새 RP 성능 개선이나 모델 승격은 확인되지 않았습니다.',
                fontsize=10,color='#4d6672',linespacing=1.6)
            fig.subplots_adjust(left=.245,right=.965,top=.90,bottom=.145,hspace=.29)
            for extension in ('png','svg'):
                fig.savefig(directory/f'CLASSIFIER_ALL_CASES_HORIZON_{horizon}.{extension}',dpi=150)
            plt.close(fig)
        save(str((directory/'MANIFEST.json').relative_to(BASE)),{'status':'COMPLETE','created_utc':utc_now(),
            'source_sha256':digest(__file__),'completed_cases':list(cases_all),'methods':list(METHODS),
            'horizons':[1024,4096],'displayed_cells':len(selected),'missing_cells':sum(r['value'] is None for r in selected),
            'amplitude_control_all_displayed_values_exact':True,'selected_original_rows':selected,
            'input_sha256':{source.relative_to(BASE).as_posix():digest(source),
                protocol_path.relative_to(BASE).as_posix():digest(protocol_path),
                'COMPLETE_DIAGNOSTICS_CLASSIFIERS_NEXT7.json':digest(BASE/'COMPLETE_DIAGNOSTICS_CLASSIFIERS_NEXT7.json')},
            'files_sha256':{p.name:digest(p) for p in directory.iterdir() if p.is_file()},
            'no_new_draws_or_thresholds_or_fits':True,'no_combined_metric_or_selected_cases':True})
    note('All23 classifier scenario figures complete','Two original horizons; false alarms and correct first detection separated','F5_CAUSAL_NESTED_INTEGRATION')


if __name__=='__main__':
    main()
