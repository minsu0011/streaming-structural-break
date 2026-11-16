"""Measured bottleneck and mathematical-audit figures, without invented edges."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from render_next6_extension_figures import *

def memory():
    rr=rows('FROZEN_HEAD_MEMORY_SUMMARY.csv');fig,axes=plt.subplots(2,2,figsize=(15,10));fig.suptitle('충격 뒤 오래 남은 점수는 어느 입력 묶음에서 오는가',fontsize=21,fontweight='bold')
    groups=['J102_base','C205_base','node_temporal','graph_transition'];labels=['J102 기준 특징','C205 의존성','노드·시간 특징','그래프 전이 특징'];colors=['#c54f57','#407b9c','#9e79bd','#3a9e8f'];starts=[0,8,32,128,512,2048];ticks=['0–7','8–31','32–127','128–511','512–2047','2048+']
    for ax,case in zip(axes.ravel(),('O1','OB16','V2','D2')):
        for g,label,color in zip(groups,labels,colors):
            pp=[next(r for r in rr if r['scenario']==case and int(r['relative_start'])==lo and r['group']==g and r['metric']=='mean_signed_contribution') for lo in starts];y=np.array([float(r['mean']) for r in pp]);low=np.array([float(r['ci_lo']) for r in pp]);high=np.array([float(r['ci_hi']) for r in pp]);ax.errorbar(np.arange(6),y,yerr=np.vstack((y-low,high-y)),label=label,color=color,marker='o',ms=4,capsize=3,lw=1.5)
        pp=[next(r for r in rr if r['scenario']==case and int(r['relative_start'])==lo and r['group']==groups[0] and r['metric']=='actual_mean_prediction_effect') for lo in starts];ax.plot(np.arange(6),[float(r['mean']) for r in pp],color='#253541',ls='--',lw=1.5,label='전체 점수 차이')
        ax.axhline(0,color='#aaa',lw=.8);ax.set_xticks(np.arange(6),ticks,rotation=20);ax.set_title({'O1':'단일 충격 (O1)','OB16':'16시점 충격 (OB16)','V2':'지속 분산 증가 (V2)','D2':'지속 의존성 변화 (D2)'}[case]);ax.set_ylabel('동결 M3 출력에 대한 평균 기여');ax.set_xlabel('개입 이후 상대 시점');clean(ax)
    axes[0,0].legend(fontsize=9,ncol=2);fig.tight_layout(rect=(0,.09,1,.93));save(fig,'09_FROZEN_HEAD_SHOCK_MEMORY_ATTRIBUTION','각 조건 64개 동일 seed × 5개 동결 모델. 16개 입력 조합의 정확한 4묶음 Shapley 분해; 95% 구간은 seed 단위 재표집. 혼합 입력은 실현 불가능할 수 있으므로 계산 민감도이며 인과 기여가 아닙니다.')

def raw_null():
    rr=rows('COHERENT_RAW_NULL_SUMMARY.csv');fig,axes=plt.subplots(1,2,figsize=(13,5),gridspec_kw={'width_ratios':[3,4]});fig.suptitle('정상 원시 시계열에서 잘못된 관계 변화를 찾는가',fontsize=21,fontweight='bold')
    y=np.array([float(r['any_false_edge_rate']) for r in rr]);hi=np.array([float(r['ci_hi']) for r in rr]);axes[0].errorbar(np.arange(4),y,yerr=np.vstack((np.zeros(4),hi-y)),fmt='o',color='#407b9c',capsize=6);axes[0].set_xticks(np.arange(4),['Gaussian','AR','ARCH','AR + ARCH + tail'],rotation=15);axes[0].set_ylim(-.01,.12);axes[0].set_ylabel('하나 이상의 잘못된 연결 채택률');clean(axes[0])
    axes[1].axis('off');axes[1].text(.03,.98,'4개 정상 배경 × 40회 그래프 적합',fontsize=17,fontweight='bold',va='top');axes[1].text(.03,.80,'각 그래프: 512 series · H 2,048 · O 4,096\n실제 변화 없음 · 중간 시점에 분석용 가상 phase\n실제 상태망과 동일한 원시값 → 노드 엔진',fontsize=13,linespacing=1.6,va='top');axes[1].text(.03,.43,'160개 그래프 모두 채택 연결 0개\n배경별 95% 상한은 8.76%\n“오탐 가능성 0%”를 뜻하지 않습니다.',fontsize=15,color='#a45744',linespacing=1.5,va='top')
    fig.tight_layout(rect=(0,.12,1,.9));save(fig,'10_SAME_ENGINE_RAW_STATIONARY_NEGATIVE_CONTROLS','희소 상태 backoff의 수학적 일관성을 수정한 구현으로 재검증했습니다. Wilson 구간은 조건별 40회 실험의 오차 범위입니다. 실제 데이터에서 모든 종류의 오탐을 통제한다는 보장은 아닙니다.')

if __name__=='__main__':
    power();memory();raw_null()
