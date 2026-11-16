"""Matplotlib-only actual-result renderer; no raw data or model loading."""
from pathlib import Path
import sys,json,textwrap,argparse
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/next6'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch,Circle

plt.rcParams.update({'font.family':['Malgun Gothic','DejaVu Sans'],'axes.unicode_minus':False,'font.size':9,
    'figure.facecolor':'white','savefig.facecolor':'white','svg.fonttype':'none'})
OUT=BASE/'FIGURES';OUT.mkdir(exist_ok=True)

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def savefig(fig,name):
    for ext in ('png','svg'):fig.savefig(OUT/(name+'.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)

def graph_figure(d,tag):
    fig,ax=plt.subplots(figsize=(19,16));ax.axis('off');ax.set_xlim(-1.8,2.9);ax.set_ylim(-1.6,1.7)
    theta=np.linspace(np.pi/2,np.pi/2+2*np.pi,16,endpoint=False);xy=np.column_stack((1.15*np.cos(theta),1.15*np.sin(theta)))
    for e in d['edges']:
        i,j=e['source'],e['target'];gain=e.get('selection_gain',0);color='#b86d31' if 'MECHANICAL' in e['mechanical'] else '#267b9b'
        arrow=FancyArrowPatch(xy[i],xy[j],arrowstyle='-|>',mutation_scale=15,color=color,lw=1+min(gain*10,3),connectionstyle='arc3,rad=.1',shrinkA=23,shrinkB=23,alpha=.8)
        ax.add_patch(arrow);mid=(xy[i]+xy[j])/2
        ax.text(mid[0],mid[1],f"lag {e['lag']} · {gain:+.3f}",fontsize=7,color=color,bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'})
    for node,(x,y) in zip(d['nodes'],xy):
        j=node['id'];ax.add_patch(Circle((x,y),.075,facecolor='#e9f0f5',edgecolor='#32566b',lw=1.2));ax.text(x,y,str(j),ha='center',va='center',weight='bold')
        ax.text(x*1.24,y*1.2,node['korean'],ha='center',va='center',fontsize=8)
    ax.text(1.72,1.32,'실제 피처 좌표',weight='bold',fontsize=12)
    for j,n in enumerate(d['nodes']):
        ax.text(1.72,1.20-j*.152,f"{j:02d}  {n['korean']}\n"+textwrap.fill(n['feature_name'],43),fontsize=7,va='top')
    ax.set_title('피처의 예측적 선후관계망 — 실제 inner held-out gain으로 선택',loc='left',fontsize=19,pad=18)
    supports=[r['heldout_series'] for e in d['edges'] for v in e['inner_half_evidence'].values() for r in v]
    fig.text(.1,.06,f"{d['scope']} | training {d['training_series']:,} series, {d.get('training_sample_rows','see JSON')} sampled rows | {len(d['edges'])} cross edges, 16 nodes\n"
        +f"Edge별 내부 검증 support 범위: {min(supports) if supports else 0}–{max(supports) if supports else 0} series. 화살표 숫자=lag와 두 내부 검증 gain 중 최솟값.\n"
        +'갈색: 계산식/공유 필터로 설명 가능한 연결. 파랑: 공유 raw 입력 가능성. 색은 인과성 구분이 아닙니다.\n'
        +'불확실성: 각 edge의 두 내부 검증 gain·양수 series 비율은 동봉 JSON에 기록. outer 재현 여부와 TS-AUC 개선은 별도 검증합니다.',fontsize=10)
    savefig(fig,'01_PREDICTIVE_FEATURE_NETWORK_'+tag)

def transition_figure(d,tag):
    fig,axes=plt.subplots(4,2,figsize=(17,20));positions=np.array([[-1.,0],[0.,.9],[1.,0.]])
    for row,node in enumerate((2,5,7,12)):
        for col,phase in enumerate(('normal','post')):
            rec=next(x for x in d['transitions'] if x['node']==node and x['phase']==phase);p=np.array(rec['table']);ax=axes[row,col]
            ax.axis('off');ax.set_xlim(-1.7,1.7);ax.set_ylim(-.65,1.5)
            for i,xy in enumerate(positions):
                ax.add_patch(Circle(xy,.18,facecolor=('#dcecf7','#e2efe8','#f4dfd5')[i],edgecolor='#56636c'));ax.text(*xy,('낮음','정상','높음')[i],ha='center',va='center',fontsize=11)
                ax.text(xy[0],xy[1]-.29,f'유지 {p[i,i]:.2f}',ha='center',fontsize=9)
                for j,other in enumerate(positions):
                    if i==j:continue
                    ax.add_patch(FancyArrowPatch(xy,other,arrowstyle='-|>',mutation_scale=12,lw=.5+4*p[i,j],color='#3c788f',alpha=.4+.6*p[i,j],connectionstyle='arc3,rad=.22',shrinkA=22,shrinkB=22))
                    mid=.55*xy+.45*other;delta=other-xy;mid=mid+np.array([delta[1],-delta[0]])*.12
                    ax.text(*mid,f'{p[i,j]:.2f}',fontsize=8,bbox={'facecolor':'white','edgecolor':'none','alpha':.85})
            ax.set_title(f"{'정상 모형' if phase=='normal' else '변화 이후 모형'} · {node:02d} {rec['korean']}\n{rec['name']}",fontsize=10,loc='left')
            ax.text(0,-.52,f"조건: 선택 parent가 정상 | support(low/normal/high)={rec['context_independent_series']}\n"
                +f"조건 일치 {rec['conditioned_series']} series, {rec['conditioned_sample_rows']} sampled rows | 희소 context는 own-history backoff",ha='center',fontsize=8)
    fig.suptitle('동적 상태전이망: 같은 관측 상태에 정상·변화 모형을 모두 적용',fontsize=20,y=.995)
    fig.text(.07,.005,f"{d['scope']} | training {d['training_series']:,} series | 실제 factorized 모형의 조건부 전이 확률\n"
        +'선 굵기=확률. 각 시점에는 H와 현재까지의 관측만 사용합니다. tau로 정상/변화 모형을 골라 넣지 않습니다.\n'
        +'이 노드들은 동일 raw의 계산식 유래 피처입니다. 그림의 확률은 보정된 break 확률이나 인과효과가 아닙니다.\n'
        +'불확실성: 학습 series 200회 재표집한 조건부 전이 빈도 CI를 JSON에 별도 제공(모형 확률 CI로 혼동하지 않음). 확률 표시는 소수 둘째 자리 반올림.',fontsize=10)
    fig.tight_layout(rect=[0,.06,1,.98]);savefig(fig,'02_DYNAMIC_STATE_TRANSITION_NETWORK_'+tag)

def atlas_figures(d):
    labels=[str(i)+' '+k+'\n'+n for i,(k,n) in enumerate(zip(d['korean_meaning'],d['feature_names']))]
    lookup={(r['phase'],r['lag'],r['mode']):r for r in d['results']}
    support=read(BASE/'CORRELATION_ATLAS/PLOT_SUPPORT.json') if (BASE/'CORRELATION_ATLAS/PLOT_SUPPORT.json').exists() else {}
    phases=['normal_no_break','pre','onset','earlyPost','establishedPost','laterPost']
    for phase in phases:
        rec=lookup.get((phase,0,'rank'))
        if not rec:continue
        fig,ax=plt.subplots(figsize=(15,13));im=ax.imshow(rec['mean'],vmin=-1,vmax=1,cmap='RdBu_r')
        ax.set_xticks(range(16),labels,rotation=90,fontsize=6);ax.set_yticks(range(16),labels,fontsize=6)
        ax.set_title(f"동시 관계 MAP A · {phase} · {rec['series']:,} series · {support.get(phase+':0',{}).get('points','see JSON')} points\n방향 없는 within-series rank correlation",loc='left',fontsize=15)
        fig.colorbar(im,ax=ax,shrink=.6,label='series 동일 가중 평균 상관')
        fig.text(.03,.01,f"{d['scope']} | 각 cell support와 200회 series bootstrap CI: TRAIN_ATLAS.json\n모두 같은 raw 입력의 계산식 유래 피처. 상관은 경제적 인과관계가 아닙니다.",fontsize=9)
        fig.tight_layout(rect=[0,.055,1,1]);savefig(fig,'03_MAP_A_'+phase)
    for phase in ('onset','earlyPost','establishedPost','laterPost'):
        rec=lookup.get((phase,0,'partial'));normal=lookup.get(('normal_no_break',0,'partial'))
        if not rec or not normal:continue
        fig,ax=plt.subplots(figsize=(15,13));delta=np.array(rec['mean'])-np.array(normal['mean']);im=ax.imshow(delta,vmin=-.5,vmax=.5,cmap='RdBu_r')
        ax.set_xticks(range(16),labels,rotation=90,fontsize=6);ax.set_yticks(range(16),labels,fontsize=6)
        ax.set_title(f"탐색적 차이 MAP C · {phase} − normal · 비교 창 길이 다름\n시간 추세·공통 크기 통제 후 차이: 창 길이 효과가 포함됨",loc='left',fontsize=15)
        fig.colorbar(im,ax=ax,shrink=.6,label='부분 rank 상관 차이');fig.text(.03,.01,f"{d['scope']} | {rec['series']} event series / {normal['series']} null series\n같은 길이·시점 대조는 07_ALIGNMENT matched_not_yet_delta, 창 길이 함정은 08_WINDOW_LENGTH_CONFOUND 그림을 확인하세요.\n계산식 유래·공통 입력 가능성 유지. CI/support/points는 JSON. 이 지도만으로 변화 신호를 주장하지 않습니다.",fontsize=9)
        fig.tight_layout(rect=[0,.055,1,1]);savefig(fig,'04_MAP_C_'+phase)
    for lag in (1,2,4,8,16):
        rec=lookup.get(('normal_no_break',lag,'rank'))
        if not rec:continue
        fig,ax=plt.subplots(figsize=(15,13));im=ax.imshow(rec['mean'],vmin=-1,vmax=1,cmap='RdBu_r')
        ax.set_xticks(range(16),labels,rotation=90,fontsize=6);ax.set_yticks(range(16),labels,fontsize=6)
        ax.set_xlabel('후행 피처 j(t)');ax.set_ylabel(f'선행 피처 i(t−{lag})');ax.set_title(f"시차 관계 MAP B · lag {lag} · 정상 {rec['series']} series · {support.get('normal_no_break:'+str(lag),{}).get('points','see JSON')} pairs",loc='left',fontsize=15)
        fig.colorbar(im,ax=ax,shrink=.6);fig.text(.03,.01,f"{d['scope']} | 실제 연속 시차; series 경계를 넘지 않음 | support/CI는 JSON\n이 지도만으로 edge를 확정하지 않으며, 별도 own-history 대비 예측 이득 검증을 사용합니다.",fontsize=9)
        fig.tight_layout(rect=[0,.055,1,1]);savefig(fig,'05_MAP_B_lag'+str(lag))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--refresh-annotations',action='store_true');args=parser.parse_args()
    for path in (BASE/'STATE_TRANSITION_TABLES').glob('fold0_*.json'):
        tag=path.stem;d=read(path)
        if args.refresh_annotations or not (OUT/('02_DYNAMIC_STATE_TRANSITION_NETWORK_'+tag+'.png')).exists():graph_figure(d,tag);transition_figure(d,tag)
    atlas=BASE/'CORRELATION_ATLAS/TRAIN_ATLAS.json'
    if atlas.exists() and (args.refresh_annotations or not (OUT/'05_MAP_B_lag16.png').exists()):atlas_figures(read(atlas))
    print('NEXT6 result-based PNG/SVG figures:',len(list(OUT.glob('*.png'))),flush=True)

if __name__=='__main__':main()
