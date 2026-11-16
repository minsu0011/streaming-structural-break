from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from render_next6_extension_figures import *

def main():
    hr=rows('RANK_CHANNEL_CONFIRMATION_LIMITS.csv');rr=rows('SCALE_DIRECTION_CONFIRMATION_DIAGNOSTIC.csv')
    fig,axes=plt.subplots(1,3,figsize=(17,8),gridspec_kw={'width_ratios':[1.15,1,1]});fig.suptitle('분산 감소의 병목: 특징의 아래쪽 한계와 네 구간 확인',fontsize=22,fontweight='bold')
    x=np.arange(8);val=[float(r['negative_confirmation_mathematically_impossible_fraction']) for r in hr]
    axes[0].barh(x,val,color=['#9aaebb']*2+['#c89452']+['#9aaebb']*5);axes[0].set_yticks(x,NODE_KO);axes[0].set_ylim(7.5,-.5);axes[0].set_xlim(0,1.04);axes[0].set_title('어떤 입력도 음의 확인에 도달 못하는 H');axes[0].set_xlabel('RP 6,400개 H 중 비율');axes[0].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1));clean(axes[0])
    for j,v in enumerate(val):axes[0].text(v+.014,j,'<0.1%' if 0<v<.001 else f'{v:.1%}',va='center',fontsize=10)
    for ax,case,direction,title in zip(axes[1:],('V2','V4'),('positive','negative'),('분산 2배 · 양의 증거','분산 절반 · 음의 증거')):
        data=[next(r for r in rr if r['scenario']==case and r['horizon']=='1024' and int(r['node'])==j) for j in range(8)]
        one=[float(r[direction+'_single_block']) for r in data];four=[float(r[direction+'_four_new_blocks']) for r in data]
        ax.barh(x-.17,one,height=.30,color='#bccbd3',label='한 구간 이상 초과');ax.barh(x+.17,four,height=.30,color='#287f8e',label='네 구간 연속 같은 방향');ax.set_yticks(x,['']*8);ax.set_ylim(7.5,-.5);ax.set_xlim(0,1.04);ax.set_title(title);ax.set_xlabel('이미 평가한 동일 512개 seed');ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1));clean(ax)
        for j,v in enumerate(four):
            if v>=.02:ax.text(v+.015,j+.17,f'{v:.1%}',va='center',fontsize=10)
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),fontsize=9)
    fig.tight_layout(rect=(0,.23,1,.91));save(fig,'20_SCALE_CONTRACTION_CONFIRMATION_BOTTLENECK','오른쪽은 1,024시점 경로의 변화 이후 순수 원시 입력 구간만 본 기전 진단입니다. 사전 경보를 제외한 공식 최초 검출률과 다릅니다. 한 구간 확인기를 새로 제안하거나 적합하지 않았습니다. 꼬리 감소의 한계가 다른 노드까지 불가능하다는 뜻은 아닙니다.')

if __name__=='__main__':main()
