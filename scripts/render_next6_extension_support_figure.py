from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from render_next6_extension_figures import *

def main():
    rr=rows('FRESH_NODE_STATE_OCCUPANCY.csv');hh=rows('H_REFERENCE_SUPPORT_SUMMARY.csv');fig,axes=plt.subplots(1,3,figsize=(16,8),gridspec_kw={'width_ratios':[3,3,3.5]});fig.suptitle('노드 상태의 점유와 H 보정 자료의 제약',fontsize=23,fontweight='bold')
    for phase,ax in enumerate(axes[:2]):
        a=np.array([[float(next(r for r in rr if r['partition']=='outer_train_fold0' and int(r['phase'])==phase and int(r['node'])==j)[k]) for k in ('low_fraction','normal_fraction','high_fraction')] for j in range(8)]);ax.imshow(a,vmin=0,vmax=1,cmap='Blues',aspect='auto');ax.set_xticks(range(3),['낮음','정상','높음']);ax.set_yticks(range(8),NODE_KO if phase==0 else ['']*8);ax.set_title('정상 phase' if phase==0 else '변화 이후 phase')
        for j in range(8):
            for k in range(3):ax.text(k,j,f'{a[j,k]:.0%}',ha='center',va='center',color='white' if a[j,k]>.5 else '#263b49',fontsize=11)
    vals=np.array([float(r['spread_floor_fraction']) for r in hh]);axes[2].barh(np.arange(8),vals,color=['#9aaebb']*6+['#c89452']*2);axes[2].set_yticks(range(8),['']*8);axes[2].set_ylim(7.5,-.5);axes[2].set_xlim(0,.5);axes[2].set_title('H 보정 폭 하한 사용 비율');axes[2].set_xlabel('전체 RP 6,400 series');axes[2].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1));clean(axes[2])
    for j,v in enumerate(vals):axes[2].text(v+.01,j,f'{v:.1%}',va='center',fontsize=11)
    fig.tight_layout(rect=(0,.14,1,.92));save(fig,'15_NODE_STATE_AND_H_REFERENCE_SUPPORT','왼쪽은 fold 0 학습 자료에서 두 구간의 현재 관측이 같은 phase인 경우이며 series/phase 비중을 맞췄습니다. 시차 4의 경계 참조는 별도 감사했습니다. 오른쪽은 전체 H 진단입니다. H 보정 구간은 25~125개(중앙값74)입니다. 하한 사용이 누락 원인이라는 증명은 아닙니다.')
if __name__=='__main__':main()
