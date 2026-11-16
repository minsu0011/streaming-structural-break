from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from render_next6_extension_figures import *

def legacy():
    rr=rows('LEGACY_MEMORY_PRIMITIVE_SUMMARY.csv');fig,axes=plt.subplots(1,2,figsize=(14,6));fig.suptitle('오래 남는 충격 효과를 누적 최대값까지 추적했습니다',fontsize=21,fontweight='bold');starts=[0,8,32,128,512,2048];labels=['0–7','8–31','32–127','128–511','512–2047','2048+']
    groups=[('serial_lifetime_max','전 구간 누적 최대값','#c54f57'),('ARCH_original_51','ARCH 원래 51특징','#407b9c'),('serial_current_and_recent','현재·최근 분산 특징','#9e79bd'),('serial_decayed_max','감쇠 최대값','#3a9e8f')]
    for ax,case in zip(axes,('O1','OB16')):
        for group,label,color in groups:
            pp=[next(r for r in rr if r['scenario']==case and int(r['relative_start'])==lo and r['group']==group and r['metric']=='mean_signed_contribution') for lo in starts];y=np.array([float(r['mean']) for r in pp]);low=np.array([float(r['ci_lo']) for r in pp]);hi=np.array([float(r['ci_hi']) for r in pp]);ax.errorbar(range(6),y,yerr=np.vstack((y-low,hi-y)),color=color,label=label,marker='o',capsize=3,ms=4)
        ax.set_xticks(range(6),labels,rotation=20);ax.set_title('단일 충격' if case=='O1' else '16시점 연속 충격');ax.set_xlabel('개입 이후 상대 시점');ax.set_ylabel('동결 M3 출력에 대한 조건부 기여');ax.axhline(0,color='#aaa',lw=.8);clean(ax)
    axes[0].legend(fontsize=9);fig.tight_layout(rect=(0,.13,1,.9));save(fig,'11_LEGACY_LIFETIME_MAX_MEMORY_BOTTLENECK','기존 64개 seed를 그대로 추적한 원인 진단입니다. J102 안의 4묶음만 치환하고 나머지는 실제 충격 입력으로 고정했습니다. 95% seed 재표집 구간. 혼합 입력의 계산 민감도이며 인과 기여나 새 모델 성적이 아닙니다.')

def sparse():
    d=read('SPARSE_NULL_REPAIR_VISUAL_DATA.json');fig,axes=plt.subplots(1,2,figsize=(13,6));fig.suptitle('희소한 문맥 하나만 되돌리면 없애려던 관계가 다시 생깁니다',fontsize=19,fontweight='bold')
    for ax,key,title in zip(axes,('before_log_odds_phase_difference','after_log_odds_phase_difference'),('개별 문맥 backoff · 수정 전','자기 상태 층 전체 backoff · 수정 후')):
        a=np.array(d[key]);ax.imshow(a,vmin=-3.1,vmax=3.1,cmap='RdBu_r',aspect='auto');ax.set_title(title);ax.set_yticks(range(3),['이전 자기 상태: 낮음','이전 자기 상태: 정상','이전 자기 상태: 높음']);ax.set_xticks(range(4),['비교 1','비교 2','비교 3','비교 4']);ax.set_xlabel('부모–자식 인접 2×2 셀의 로그 오즈 차이')
        for i in range(3):
            for j in range(4):ax.text(j,i,f'{a[i,j]:+.3f}' if abs(a[i,j])>1e-8 else '0',ha='center',va='center',fontweight='bold',color='white' if abs(a[i,j])>1.6 else '#243541')
    fig.tight_layout(rect=(0,.18,1,.9));save(fig,'12_SPARSE_NULL_MATHEMATICAL_REPAIR','32개 series의 고정 합성 희소 표. 제거한 phase 상호작용이 없으면 두 phase의 부모–자식 로그 오즈 차이는 0이어야 합니다. 원래 최악 fixture를 그대로 재검사했습니다. 실제 RP 성적을 보고 문턱을 조정한 것이 아닙니다.')

def memory_clock():
    rr=rows('MACRO_STATE_CLOCK_AND_DWELL_DIAGNOSTIC.csv');fig,axes=plt.subplots(1,2,figsize=(14,6));fig.suptitle('상태 이름 외에도 구간 시계와 체류 시간이 영향을 줍니다',fontsize=21,fontweight='bold')
    for partition,label,color,ls in [('outer_train_fold0','학습','#407b9c','-'),('outer_heldout_fold0','held-out','#c54f57','--')]:
        pp=[r for r in rr if r['partition']==partition and r['previous_state']=='RECOVERING' and r['stratifier']=='dwell' and int(r['official_phase'])==0 and r['exit_probability']];names=['1','2-15','16-31','32-63'];y=[float(next(r for r in pp if r['stratum']==x)['exit_probability']) for x in names];axes[0].plot(range(4),y,label=label,color=color,ls=ls,marker='o');axes[0].set_xticks(range(4),names)
        pp=[r for r in rr if r['partition']==partition and r['previous_state']=='PERSISTENT' and r['stratifier']=='clock' and int(r['official_phase'])==0 and r['exit_probability']];axes[1].plot([int(r['stratum']) for r in pp],[float(r['exit_probability']) for r in pp],label=label,color=color,ls=ls,marker='o',ms=3)
    axes[0].set_xlabel('이전 회복 상태의 체류 시점 수');axes[0].set_ylabel('다음 시점에 회복 상태를 떠나는 비중');axes[0].set_title('회복 상태의 체류 시간');axes[1].set_xlabel('현재 online 시점 mod 16');axes[1].set_ylabel('지속 확인 상태를 떠나는 비중');axes[1].set_title('완성 구간 경계는 mod 16 = 15');axes[1].set_xticks(range(0,16,3))
    for ax in axes:clean(ax);ax.legend(fontsize=10)
    fig.tight_layout(rect=(0,.14,1,.9));save(fig,'13_OBSERVABLE_NETWORK_CLOCK_AND_DWELL','원래 fold 0 정상 phase에서 series/phase별 가중치를 먼저 맞춘 관측 빈도입니다. 7상태 그림은 평균 전이 설명이며 완전한 1차 Markov 생성 모델이 아닙니다. 추가로 조건화한 예측기나 후보를 학습하지 않았습니다.')

def rejection():
    rr=rows('COHERENT_GRAPH_CANDIDATE_DIAGNOSTIC.csv');fig,axes=plt.subplots(1,2,figsize=(14,6));fig.suptitle('실제 RP의 교차 관계 후보가 채택되지 않은 이유',fontsize=21,fontweight='bold')
    for fold in range(5):
        pp=[r for r in rr if int(r['fold'])==fold];x=[float(r['minimum_half_gain']) for r in pp];y=[float(r['minimum_half_positive_fraction']) for r in pp];axes[0].scatter(x,y,s=20,alpha=.65,label=f'fold {fold}')
    axes[0].axvline(.001,color='#c54f57',ls='--',lw=1.3);axes[0].axhline(.55,color='#c54f57',ls='--',lw=1.3);axes[0].set_xlabel('두 내부 검증 중 작은 full–null 로그 점수 이득');axes[0].set_ylabel('두 내부 검증 중 작은 양의 series 비중');axes[0].legend(fontsize=8,ncol=2);clean(axes[0])
    counts=rows('COHERENT_GRAPH_REJECTION_SUMMARY.csv');x=np.arange(5);axes[1].bar(x-.15,[int(r['both_halves_positive_gain']) for r in counts],.3,label='두 절반 모두 이득 > 0',color='#407b9c');axes[1].bar(x+.15,[int(r['both_all_gates']) for r in counts],.3,label='고정 기준 모두 통과',color='#c54f57');axes[1].set_xticks(x,[f'fold {f}' for f in range(5)]);axes[1].set_ylabel('후보 수 (각 fold 전체 52개)');axes[1].legend(fontsize=9);clean(axes[1]);fig.tight_layout(rect=(0,.13,1,.9));save(fig,'14_RP_CONDITIONAL_GRAPH_REJECTION_REASONS','선택은 원래 outer train 내부의 두 그룹 절반만 사용했습니다. 최소 이득 0.001·양의 비중 0.55를 양쪽에서 모두 요구한 고정 기준입니다. 0개라는 결과는 이 표현과 검사에서 근거를 못 얻었다는 뜻이며 모든 관계가 없다는 증명은 아닙니다.')

if __name__=='__main__':
    for fn in (legacy,sparse,memory_clock,rejection):fn();print(fn.__name__,flush=True)
