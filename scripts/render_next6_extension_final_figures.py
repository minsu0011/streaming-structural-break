from pathlib import Path
import sys,ast,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from render_next6_extension_figures import *

def synthetic():
    rr=rows('FRESH_VS_FROZEN_PAIRED_SYNTHETIC.csv');fig,axes=plt.subplots(1,2,figsize=(15,8));fig.suptitle('동일 경로에서 경보 감소와 변화 확인을 함께 비교합니다',fontsize=22,fontweight='bold')
    menus=[('alarm',[('N0','Gaussian 정상'),('N1','AR 정상'),('N2','ARCH 정상'),('N3','AR·ARCH·꼬리'),('O1','단일 충격'),('OB16','16시점 충격'),('REPEATED_SHOCK','반복 충격')]),('first_detection',[('V1','분산 1.5배'),('V2','분산 2배'),('V3','분산 0.7배'),('V4','분산 절반'),('D1','자기상관 증가'),('D2','자기상관 반전'),('RETURN_NORMAL','변화 후 정상 복귀')])]
    for ax,(metric,menu) in zip(axes,menus):
        for ref,offset,color in [('M0',-.13,'#407b9c'),('M3',.13,'#c89452')]:
            pp=[next(r for r in rr if r['scenario']==case and r['reference']==ref and r['metric']==metric and int(r['horizon'])==1024) for case,_ in menu];y=np.arange(len(pp))+offset;v=np.array([float(r['FRESH_minus_reference_mean']) for r in pp]);lo=np.array([float(r['ci_lo']) for r in pp]);hi=np.array([float(r['ci_hi']) for r in pp]);ax.errorbar(v,y,xerr=np.vstack((v-lo,hi-v)),fmt='o',color=color,ms=5,capsize=3,label='새 상태망 − '+ref)
        ax.axvline(0,color='#777',lw=1);ax.set_yticks(range(len(menu)),[label for _,label in menu]);ax.invert_yaxis();ax.set_xlabel('비율 차이와 95% paired bootstrap 구간');ax.legend(fontsize=10);ax.grid(axis='x',alpha=.15);ax.spines[['top','right']].set_visible(False);ax.set_title('경로 경보: 음수이면 감소' if metric=='alarm' else '첫 변화 확인: 양수이면 증가')
    fig.tight_layout(rect=(0,.12,1,.92));save(fig,'16_PAIRED_SHOCK_REJECTION_AND_DETECTION','각 조건 512개 seed, 1,024시점. 같은 seed의 5개 동결 fold 예측을 함께 재표집했습니다. 새 상태망은 보정된 사건확률이나 승격 후보가 아닙니다. 구간은 고정 예측에 조건부이며 적응적 개발 재사용을 보정하지 않습니다.')
    summary=rows('COUNTEREXAMPLE_RATE_AND_DELAY_COMPARISON.csv');fig,axes=plt.subplots(2,2,figsize=(14,10));fig.suptitle('관측 기간을 늘릴 때의 확인율과 오경보를 함께 봅니다',fontsize=21,fontweight='bold')
    for col,horizon in enumerate((1024,4096)):
        for row,(metric,cases,labels) in enumerate([('observed_null_alarm_rate',['N0','N2','O1','OB16'],['Gaussian','ARCH','단일 충격','16시점 충격']),('first_detection_rate',['V2','V4','D1','D2'],['분산 2배','분산 절반','상관 증가','상관 반전'])]):
            ax=axes[row,col]
            for method,offset,color in [('M0',-.2,'#407b9c'),('M3',0.,'#c89452'),('FRESH',.2,'#3a9e8f')]:
                pp=[next(r for r in summary if r['scenario']==case and r['method']==method and r['metric']==metric and int(r['horizon'])==horizon) for case in cases];v=np.array([float(r['mean_across_fixed_fold_models']) for r in pp]);lo=np.array([float(r['minimum_fixed_fold_model']) for r in pp]);hi=np.array([float(r['maximum_fixed_fold_model']) for r in pp]);ax.errorbar(np.arange(4)+offset,v,yerr=np.vstack((v-lo,hi-v)),fmt='o',color=color,capsize=4,ms=6,label=method)
            ax.set_xticks(range(4),labels);ax.set_ylim(-.025,1.025);ax.set_title(f'{horizon:,}시점 · '+('경보율' if row==0 else '변화 확인율'));ax.set_ylabel('비율');clean(ax)
    axes[0,0].legend(fontsize=10);fig.tight_layout(rect=(0,.1,1,.94));save(fig,'17_HORIZON_FALSE_ALARM_AND_DETECTION_TRADEOFF','점은 5개 고정 fold 모델의 평가 비율 평균, 세로선은 그 모델들 사이의 최솟값~최댓값입니다(신뢰구간 아님). FRESH는 한 고정 상태 규칙입니다. 모델 점수를 평균해 새 ensemble을 만들지 않았습니다. paired 불확실성은 별도 표와 그림16에 있습니다.')

def bootstrap():
    rr=rows('BOOTSTRAP_EDGE_SELECTION_FREQUENCY.csv');sizes=rows('BOOTSTRAP_GRAPH_SIZE_DISTRIBUTION.csv');keys=sorted({(int(r['source']),int(r['target']),int(r['lag'])) for r in rr if r['original_selected']=='True'});lookup={(int(r['fold']),int(r['source']),int(r['target']),int(r['lag'])):r for r in rr};tree=ast.parse((ROOT/'src/next6/nodes.py').read_text(encoding='utf-8'));names=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='KOREAN' for t in n.targets)))
    a=np.array([[float(lookup.get((f,*key),{}).get('selection_frequency',0)) for f in range(5)] for key in keys]);fig,axes=plt.subplots(1,2,figsize=(17,max(10,len(keys)*.47)),gridspec_kw={'width_ratios':[5,3]});fig.suptitle('원래 선택한 연결이 학습 자료 재표집에도 남는가',fontsize=22,fontweight='bold');axes[0].imshow(a,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto');axes[0].set_xticks(range(5),[f'fold {f}' for f in range(5)]);axes[0].set_yticks(range(len(keys)),[f'{names[s]} → {names[t]} · lag {lag}' for s,t,lag in keys],fontsize=8)
    for i,key in enumerate(keys):
        for f in range(5):
            original=lookup.get((f,*key),{}).get('original_selected')=='True';axes[0].text(f,i,f'{a[i,f]:.1%}'+('*' if original else ''),ha='center',va='center',fontsize=9,color='white' if a[i,f]>.55 else '#233b4a')
    data=[[int(r['edges']) for r in sizes if int(r['fold'])==f] for f in range(5)];axes[1].boxplot(data,tick_labels=[f'fold {f}' for f in range(5)],showmeans=True);axes[1].set_ylabel('재표집 그래프의 채택 연결 수');axes[1].set_title('각 fold 200개 그래프');clean(axes[1]);fig.tight_layout(rect=(0,.11,1,.95));save(fig,'18_ORIGINAL_GRAPH_GROUP_BOOTSTRAP_STABILITY','원래 내부 해시 절반 안에서 그룹 전체를 다시 뽑고 같은 선택 절차를 수행했습니다. *는 원래 해당 fold에서 선택한 연결입니다. 빈도는 참인 연결의 확률이 아니며, 안정적인 기계적 파생 관계도 포함합니다. 계수 부호와 Monte Carlo 구간은 CSV에 있습니다. 상자는 25~75%, 주황선은 중앙값, 초록 삼각형은 평균입니다. 수염은 1.5 IQR 안의 관측 범위, 원은 그 밖의 재표집 결과입니다.')

def latency():
    rr=rows('LATENCY_NEXT6_EXTENSION.csv');cold=rows('COLD_START_NEXT6_EXTENSION.csv');fig,axes=plt.subplots(1,2,figsize=(14,7));fig.suptitle('실제 스칼라 호출의 반복 비용과 첫 실행 비용',fontsize=22,fontweight='bold');labels=['기존 BL80','기존 M3','상태 코어 20열','설명용 27열','최종 빠른 27열'];v=np.array([float(r['scalar_median_us']) for r in rr]);p95=np.array([float(r['scalar_p95_us']) for r in rr]);axes[0].barh(range(5),v,color=['#829eaf','#829eaf','#adc8cc','#c89452','#3a9e8f']);axes[0].errorbar(v,range(5),xerr=np.vstack((np.zeros(5),p95-v)),fmt='none',ecolor='#4a6575',capsize=3);axes[0].set_yticks(range(5),labels);axes[0].invert_yaxis();axes[0].set_xlabel('1시점 처리 μs · 막대=중앙값, 끝선=p95');clean(axes[0])
    for i,value in enumerate(v):axes[0].text(p95[i]+.4,i,f'{value:.2f}',va='center')
    axes[0].set_xlim(0,max(p95)*1.16)
    value=np.array([float(r['model_load_and_prepare_seconds'])+float(r['cold_H_state_seconds'])+float(r['first_scalar_seconds']) for r in cold]);axes[1].barh(range(5),value,color='#abc1ce');axes[1].set_yticks(range(5),labels);axes[1].invert_yaxis();axes[1].set_xlabel('별도 프로세스의 모델 준비+H 초기화+첫 호출 (초)');clean(axes[1])
    for i,x in enumerate(value):axes[1].text(x+.35,i,f'{x:.2f}',va='center')
    axes[1].set_xlim(0,max(value)*1.16);fig.tight_layout(rect=(0,.14,1,.92));save(fig,'19_ACTUAL_SCALAR_RUNTIME_AND_COLD_START','같은 H와 4,096개 관측, 7회 순서 교대. 대규모 연구 작업 종료 후 측정했습니다. 설명용27열과 최종27열은 같은 수학·출력입니다. M0/M3/상태 코어는 역할과 계산이 다른 모델입니다. 첫 실행은 방법별 별도 프로세스·빈 Numba 캐시이며 일반적인 속도 보장은 아닙니다.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--section',choices=['synthetic','bootstrap','latency'],required=True);a=p.parse_args();globals()[a.section]()
