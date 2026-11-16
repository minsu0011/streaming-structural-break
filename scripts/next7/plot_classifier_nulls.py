"""Plot all four completed fixed null backgrounds with their original intervals."""
from pathlib import Path
import argparse
import csv
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.next7.common import BASE, read_json, save, csv_write, digest, utc_now, activity

METHODS = ('M0','M3','AMPLITUDE_0_9_M0','M_EVENT','G2_EVENT','I4_EVENT_SCALE_TREE','G2_FULL','G3_PAIRWISE_FULL')
LABELS = ('BL80 (M0)','NEXT6 M3','0.9 × M0','원시 event 혼합','H 보정 event 혼합','I4 잔차 트리','G2 전체 보정','G3 쌍별 보정')


def rows(path):
    with path.open(newline='',encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def main(snapshot):
    assert snapshot and all(c.isalnum() or c in '_-' for c in snapshot)
    relative = 'visuals/classifier_nulls/' + snapshot
    directory = BASE/relative
    assert not directory.exists()
    inputs, individual, paired = {}, [], []
    for case in ('N0','N1','N2','N3'):
        source = BASE/'classifier_statistics'/case
        done = read_json(source/'DONE.json')
        assert done['status']=='COMPLETE' and done['fixed_folds']==5 and done['draws']==2000
        for name in ('SUMMARY.csv','PAIRED_CONTRASTS.csv'):
            assert digest(source/name)==done['files_sha256'][name]
            inputs[f'classifier_statistics/{case}/{name}']=digest(source/name)
        individual += [r for r in rows(source/'SUMMARY.csv') if r['metric']=='any_alarm']
        paired += [r for r in rows(source/'PAIRED_CONTRASTS.csv') if r['metric']=='any_alarm' and r['reference']=='M0']
    assert len(individual)==4*3*8 and len(paired)==4*3*7
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':11,
        'figure.facecolor':'#f5f7fb','axes.facecolor':'white','savefig.facecolor':'#f5f7fb'})
    with activity('Visualize completed four-background classifier null risk', 'VISUALIZATION_ENGINEERING','F4_NULL_GROWTH_CORRECTION'):
        directory.mkdir(parents=True)
        lookup={(r['case'],int(r['horizon']),r['method']):r for r in individual}
        fig,axes=plt.subplots(1,3,figsize=(16,6.7),gridspec_kw={'wspace':.10})
        for j,horizon in enumerate((1024,4096,16384)):
            matrix=np.array([[float(lookup[(case,horizon,method)]['mean_across_fixed_folds']) for case in ('N0','N1','N2','N3')] for method in METHODS])
            im=axes[j].imshow(matrix,cmap='YlOrRd',vmin=0,vmax=1,aspect='auto')
            axes[j].set_xticks(range(4),('N0','N1','N2','N3'))
            axes[j].set_yticks(range(8),LABELS if j==0 else ['']*8)
            axes[j].tick_params(length=0,pad=9)
            axes[j].set_title(f'{horizon:,}개 관측까지 한 번 이상 경보',pad=15,fontsize=12)
            for row in range(8):
                for col in range(4):
                    axes[j].text(col,row,f'{100*matrix[row,col]:.1f}%',ha='center',va='center',
                        color='white' if matrix[row,col]>.64 else '#162237',fontsize=11)
            for spine in axes[j].spines.values():spine.set_visible(False)
        fig.suptitle('같은 모델도 배경에 따라 장기 오경보가 달라집니다',fontsize=19,x=.52,y=.98)
        fig.text(.03,.075,'각 모델의 고정된 자체 경보 기준 사용 · 512개 동일 seed · 고정된 5개 fold의 평균',fontsize=11)
        fig.text(.03,.035,'0.9 × M0는 자체 기준도 0.9배여서 M0와 경보가 같습니다. 색은 확률 보정의 보증을 뜻하지 않습니다.',fontsize=10,color='#475569')
        fig.subplots_adjust(left=.14,right=.985,bottom=.15,top=.85)
        fig.savefig(directory/'NULL_ALARM_BY_HORIZON.png',dpi=160)
        fig.savefig(directory/'NULL_ALARM_BY_HORIZON.svg')
        plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(15,9.3))
        compared=METHODS[1:]
        for ax,case in zip(axes.flat,('N0','N1','N2','N3')):
            chosen={r['candidate']:r for r in paired if r['case']==case and r['horizon']=='16384'}
            for j,method in enumerate(compared):
                r=chosen[method]
                mean,lo,hi=(100*float(r[k]) for k in ('mean_delta','CI_low','CI_high'))
                color='#b45309' if lo>0 else '#047857' if hi<0 else '#475569'
                ax.plot([lo,hi],[j,j],color=color,lw=2)
                ax.scatter([mean],[j],color=color,s=36,zorder=3)
                ax.text(.985,j,f'{mean:+.2f}',va='center',ha='right',transform=ax.get_yaxis_transform(),fontsize=9,color=color)
            ax.axvline(0,color='#334155',lw=1,ls='--')
            ax.set_yticks(range(7),LABELS[1:])
            ax.invert_yaxis()
            ax.set_xlim(-25,35)
            ax.set_title(f'{case} · BL80 경보율 {100*float(lookup[(case,16384,"M0")]["mean_across_fixed_folds"]):.1f}%',loc='left',fontsize=13)
            ax.set_xlabel('BL80 대비 오경보율 차이 (%p)')
            ax.grid(axis='x',alpha=.2)
            ax.spines[['top','right']].set_visible(False)
        fig.suptitle('장기 오경보의 증감: 16,384개 관측',fontsize=19,y=.98)
        fig.text(.025,.055,'점: 대응 평균 차이 · 선: 기존 2,000회 bootstrap의 95% 구간 · 갈색: 증가 구간 · 녹색: 감소 구간',fontsize=10)
        fig.text(.025,.026,'모든 배경을 표시했습니다. 구간은 각각의 비교에 대한 값이며, 전체 비교를 동시에 보장하지 않습니다. 수치는 새 경보 기준 선택에 쓰지 않았습니다.',fontsize=9.5,color='#475569')
        fig.subplots_adjust(left=.14,right=.98,top=.9,bottom=.135,wspace=.43,hspace=.36)
        fig.savefig(directory/'PAIRED_LONG_NULL_ALARM_CHANGE.png',dpi=160)
        fig.savefig(directory/'PAIRED_LONG_NULL_ALARM_CHANGE.svg')
        plt.close(fig)
        csv_write(relative+'/SOURCE_ALARM_RATES.csv',individual)
        csv_write(relative+'/SOURCE_PAIRED_CHANGES.csv',paired)
        save(relative+'/DONE.json',{'status':'COMPLETE','created_utc':utc_now(),'source_sha256':digest(__file__),
            'inputs_sha256':inputs,'outputs_sha256':{p.name:digest(p) for p in directory.iterdir() if p.is_file()},
            'backgrounds':4,'horizons':[1024,4096,16384],'models':METHODS,
            'new_statistics_or_calibration':False,'visual_review_required':True})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--snapshot',required=True)
    main(parser.parse_args().snapshot)
