"""Compare the four original variance changes using frozen classifier cutoffs."""
from pathlib import Path
import argparse
import csv
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.next7.common import BASE,read_json,save,digest,utc_now,activity
from scripts.next7.plot_classifier_nulls import METHODS,LABELS

CASES=('V1','V2','V3','V4')
CASE_LABELS=('V1 · 분산1.5배','V2 · 분산2배','V3 · 분산0.7배','V4 · 분산0.5배')


def rows(path):
    with path.open(newline='',encoding='utf-8-sig') as handle:return list(csv.DictReader(handle))


def main(snapshot):
    assert snapshot and all(c.isalnum() or c in '_-' for c in snapshot)
    relative='visuals/classifier_scale/'+snapshot
    destination=BASE/relative
    assert not destination.exists()
    individual,paired,inputs=[],[],{}
    for case in CASES:
        directory=BASE/'classifier_statistics'/case
        receipt=read_json(directory/'DONE.json')
        assert receipt['status']=='COMPLETE' and receipt['fixed_folds']==5 and receipt['draws']==2000
        for name in ('SUMMARY.csv','PAIRED_CONTRASTS.csv'):
            assert digest(directory/name)==receipt['files_sha256'][name]
            inputs[f'classifier_statistics/{case}/{name}']=digest(directory/name)
        individual += [r for r in rows(directory/'SUMMARY.csv') if r['metric']=='postbreak_first_detection']
        paired += [r for r in rows(directory/'PAIRED_CONTRASTS.csv') if r['metric']=='postbreak_first_detection' and r['reference']=='M0']
    assert len(individual)==64 and len(paired)==56
    lookup={(r['case'],int(r['horizon']),r['method']):r for r in individual}
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':11,
        'figure.facecolor':'#f5f7fb','axes.facecolor':'white','savefig.facecolor':'#f5f7fb'})
    with activity('Visualize completed fixed classifier variance-change power','VISUALIZATION_ENGINEERING','F3_SYMMETRIC_SCALE'):
        destination.mkdir(parents=True)
        fig,axes=plt.subplots(1,2,figsize=(15.8,7.2))
        for ax,horizon in zip(axes,(1024,4096)):
            matrix=np.array([[float(lookup[case,horizon,method]['mean_across_fixed_folds']) for case in CASES] for method in METHODS])
            assert np.isfinite(matrix).all()
            ax.imshow(matrix,cmap='YlGnBu',vmin=0,vmax=1,aspect='auto')
            ax.set_yticks(range(8),LABELS if horizon==1024 else ['']*8)
            ax.set_xticks(range(4),CASE_LABELS,fontsize=10)
            ax.tick_params(length=0,pad=9)
            ax.set_title(f'{horizon:,}개 관측까지 올바른 최초 경보',fontsize=14,pad=16)
            for i in range(8):
                for j in range(4):ax.text(j,i,f'{100*matrix[i,j]:.1f}%',ha='center',va='center',color='white' if matrix[i,j]>.58 else '#172a38')
            for spine in ax.spines.values():spine.set_visible(False)
        fig.suptitle('분산 변화의 검출: 사건 기억과 최종 경보를 구분합니다',x=.03,ha='left',fontsize=19,y=.98)
        fig.text(.03,.09,'각 모델의 별도 고정 경보 기준 · 원래 512개 시드 · 고정된 5개 fold의 평균 · 변화 전 경보는 올바른 최초 검출에서 제외',fontsize=10)
        fig.text(.03,.045,'여기서 배율은 분산입니다. V4 분산 0.5배의 표준편차는 약 0.707배입니다. 이 그림의 비율은 사건 기억률이나 TS-AUC가 아닙니다.',fontsize=10,color='#475569')
        fig.subplots_adjust(left=.145,right=.99,bottom=.18,top=.85,wspace=.08)
        fig.savefig(destination/'CLASSIFIER_SCALE_FIRST_ALARM.png',dpi=160)
        fig.savefig(destination/'CLASSIFIER_SCALE_FIRST_ALARM.svg')
        plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(15,9.5))
        for ax,case,label in zip(axes.flat,CASES,CASE_LABELS):
            chosen={r['candidate']:r for r in paired if r['case']==case and r['horizon']=='1024'}
            for i,method in enumerate(METHODS[1:]):
                row=chosen[method]
                mean,low,high=(100*float(row[key]) for key in ('mean_delta','CI_low','CI_high'))
                color='#047857' if low>0 else '#b45309' if high<0 else '#475569'
                ax.plot([low,high],[i,i],color=color,lw=2)
                ax.scatter([mean],[i],color=color,s=34,zorder=3)
                ax.text(.985,i,f'{mean:+.2f}',va='center',ha='right',transform=ax.get_yaxis_transform(),color=color,fontsize=9)
            ax.axvline(0,color='#71848f',lw=1,ls='--')
            ax.set_yticks(range(7),LABELS[1:]);ax.invert_yaxis()
            ax.set_xlim(-10,25);ax.grid(axis='x',alpha=.18)
            ax.set_title(label+' · BL80 '+f"{100*float(lookup[case,1024,'M0']['mean_across_fixed_folds']):.1f}%",loc='left')
            ax.set_xlabel('BL80 대비 올바른 최초 경보율 차이 (%p)')
            ax.spines[['top','right']].set_visible(False)
        fig.suptitle('1,024개까지 최초 경보의 대응 차이와 불확실성',fontsize=20,y=.985)
        fig.text(.025,.06,'점: 대응 차이 · 선: 기존 2,000회 시계열 단위 재표본의 95% 구간 · 모든 비교를 동시에 보장하는 구간은 아닙니다.',fontsize=10)
        fig.text(.025,.027,'국소 검출 이득은 긴 정상 경로의 오경보 및 전체 RP 순위 성능과 함께 판단합니다. 새 문턱이나 후보 선택에 사용하지 않았습니다.',fontsize=10,color='#475569')
        fig.subplots_adjust(left=.145,right=.98,top=.9,bottom=.145,wspace=.42,hspace=.4)
        fig.savefig(destination/'CLASSIFIER_SCALE_PAIRED_GAIN.png',dpi=160)
        fig.savefig(destination/'CLASSIFIER_SCALE_PAIRED_GAIN.svg')
        plt.close(fig)
        save(relative+'/DONE.json',{'status':'COMPLETE','created_utc':utc_now(),'source_sha256':digest(__file__),
            'inputs_sha256':inputs,'outputs_sha256':{p.name:digest(p) for p in destination.iterdir() if p.is_file()},
            'original_cases':4,'frozen_classifiers':8,'new_statistics_or_calibration':False,'visual_review_required':True})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',required=True)
    main(parser.parse_args().snapshot)
