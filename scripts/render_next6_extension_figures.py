"""Publication-style figures from actual locked aggregate/derived results only."""
from pathlib import Path
import csv,json,math,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch,Circle,Rectangle
from matplotlib.colors import ListedColormap
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/next6_extension';OUT=BASE/'FIGURES';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':['Malgun Gothic','DejaVu Sans'],'axes.unicode_minus':False,'font.size':11,'savefig.facecolor':'white','svg.fonttype':'none'})
COLORS=['#95a0aa','#407b9c','#e3a02a','#9e79bd','#c54f57','#3a9e8f','#c3c8ce']
STATE_KO=['준비','정상 범위','순간 충격','추가 확인 대기','지속 관측 확인','회복 확인','기준 자료 부족']
NODE_KO=['위치·부호','크기','꼬리','비대칭','1시차 의존','4시차 의존','1시차 크기 의존','4시차 크기 의존']
def read(path):return json.loads((BASE/path).read_text(encoding='utf-8'))
def rows(path):
    with (BASE/path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(fig,name,note):
    fig.text(.02,.015,note,fontsize=9,color='#52606a',ha='left',va='bottom',wrap=True)
    fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(OUT/(name+'.svg'),bbox_inches='tight');plt.close(fig)
def clean(ax):
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)

def rules():
    fig,ax=plt.subplots(figsize=(14,6));ax.set_xlim(0,14);ax.set_ylim(0,6);ax.axis('off');fig.suptitle('새로운 관측이 들어와야 지속 상태를 확인합니다',fontsize=22,fontweight='bold')
    boxes=[(.3,3.2,2.3,1.5,'H 정상 기준','앞 60% 순위 기준\n뒤 40% 구간 보정'),(3.3,3.2,2.3,1.5,'현재 관측 → 8개 노드','순위·크기·꼬리·의존\n원 관측 영향: 4시차'),(6.3,3.2,2.3,1.5,'16시점 구간 완성','겹치지 않는 현재 관측\n미완성은 확인 제외'),(9.5,3.2,3.7,1.5,'서로 다른 4개 구간 확인','같은 노드·같은 방향의 증거\n4개 중 최솟값 > 고정 임계값')]
    for x,y,w,h,title,body in boxes:
        ax.add_patch(Rectangle((x,y),w,h,facecolor='#eef4f6',edgecolor='#7599aa',lw=1.2));ax.text(x+w/2,y+h-.3,title,ha='center',va='top',fontsize=12,fontweight='bold');ax.text(x+w/2,y+.55,body,ha='center',va='center',fontsize=11)
    for left,right in zip(boxes[:-1],boxes[1:]):ax.add_patch(FancyArrowPatch((left[0]+left[2]+.08,3.9),(right[0]-.08,3.9),arrowstyle='-|>',mutation_scale=18,color='#4e7186'))
    for k in range(4):
        x=6.3+k*1.7;ax.add_patch(Rectangle((x,1.8),1.5,.6,facecolor=COLORS[4],alpha=.2+k*.15));ax.text(x+.75,2.1,f'새 구간 {k+1}',ha='center',va='center')
    ax.text(.5,2.05,'하나의 충격을 반복 계산한 잔상은\n네 번의 새 관측 확인으로 세지 않습니다.',fontsize=13,va='center')
    ax.text(.5,.75,'충격 종료 후 128시점 이내, 대응 정상 경로와 모든 출력이 일치하는지 검증',fontsize=15,fontweight='bold',color='#257b70')
    save(fig,'01_FRESH_OBSERVATION_CONFIRMATION_RULES','구현 규칙을 그린 정의도입니다. 4개 구간의 통계적 독립성을 주장하지 않습니다(시차 원자료 최대 4시점 공유). 상태 점수는 변화 발생확률이 아닙니다.')

def network():
    rr=rows('MACRO_STATE_TRANSITIONS.csv');fig,axes=plt.subplots(1,2,figsize=(16,10));fig.suptitle('실제 학습 자료에서 관측한 동적 상태전이망',fontsize=23,fontweight='bold')
    positions={0:(-1.15,.95),1:(-.55,.15),2:(-1.,-.95),3:(.65,-.9),4:(1.2,.3),5:(.35,1.25)}
    for phase,ax in enumerate(axes):
        ax.set_xlim(-1.8,1.9);ax.set_ylim(-2.9,1.9);ax.set_aspect('equal');ax.axis('off');ax.set_title('정상 조건' if phase==0 else '변화 이후 조건',fontsize=18)
        table={(int(r['previous']),int(r['current'])):r for r in rr if int(r['fold'])==0 and int(r['phase'])==phase}
        for (a,b),r in table.items():
            if a==b or a not in positions or b not in positions or int(r['unique_training_series'])<8 or not r['jump_probability_unsmoothed']:continue
            p=float(r['jump_probability_unsmoothed'])
            if p<.03:continue
            start=np.array(positions[a]);end=np.array(positions[b]);arrow=FancyArrowPatch(start,end,connectionstyle='arc3,rad=.16',shrinkA=35,shrinkB=35,arrowstyle='-|>',mutation_scale=14,linewidth=1+4*p,color=COLORS[b],alpha=.65);ax.add_patch(arrow)
            delta=end-start;perp=np.array([-delta[1],delta[0]])/np.linalg.norm(delta);middle=(start+end)/2-.16*np.linalg.norm(delta)*perp
            ax.text(*middle,f'{p:.0%}\n({r["unique_training_series"]})',ha='center',va='center',fontsize=8,color='#243642',bbox=dict(boxstyle='round,pad=.15',fc='white',ec='none',alpha=.9))
        for state,pos in positions.items():
            ax.add_patch(Circle(pos,.25,facecolor=COLORS[state],edgecolor='white',lw=2,zorder=4));ax.text(*pos,STATE_KO[state].replace(' ', '\n'),ha='center',va='center',fontsize=10,color='white',fontweight='bold',zorder=5)
            r=table[state,state];n=int(r['source_state_training_series']);selfp=float(r['ordinary_probability_smoothed']);column=state%2;row=state//2
            ax.text(-1.55+column*1.9,-1.7-row*.36,f'{STATE_KO[state]}: {selfp:.1%}\n지원 {n} series',ha='left',va='top',fontsize=9,color=COLORS[state])
        ax.text(-1.55,-1.45,'상태별 자기 유지 확률 / 해당 상태의 학습 지원',fontsize=10,color='#52606a')
    save(fig,'02_OBSERVED_DYNAMIC_STATE_TRANSITION_NETWORK','fold 0 학습 5,120 series만 사용. 화살표는 상태를 떠났을 때의 조건부 비중, 괄호는 서로 다른 series 수입니다. 지원8 미만/비중3% 미만 연결을 생략했습니다. 기준 자료 부족 상태는 학습 관측이 없어 연결을 그리지 않았습니다. 전체 표는 MACRO_STATE_TRANSITIONS.csv.')

def evidence():
    d=read('FROZEN_EVIDENCE_DECOMPOSITION.json');fig,axes=plt.subplots(1,2,figsize=(13,5));fig.suptitle('예측 가능한 연결과 변화 구분에 유용한 연결을 분리',fontsize=20,fontweight='bold')
    labels=['자기 상태 정보','전체 전이 정보','부모 관계의 추가 정보'];values=[d['mean_TS_AUC'][k] for k in ('own_state_logratio','full_transition_logratio','conditional_parent_excess')]
    axes[0].barh(labels,values,color=['#407b9c','#9e79bd','#d39460']);axes[0].set_xlim(.45,.55);axes[0].axvline(.5,color='#555',ls='--',lw=1);axes[0].set_xlabel('고정 점수 단독의 5-fold 평균 TS-AUC')
    for y,v in enumerate(values):axes[0].text(v+.001,y,f'{v:.5f}',va='center',fontsize=11)
    axes[1].axis('off');axes[1].text(.02,.97,'전체 전이 − 자기 상태',fontsize=16,fontweight='bold',va='top');axes[1].text(.02,.79,f'{d["full_minus_own_TS_AUC"]:+.6f}\n95% paired CI\n[{d["paired_CI"][0]:+.6f}, {d["paired_CI"][1]:+.6f}]',fontsize=18,color='#ad4b50',va='top');axes[1].text(.02,.30,'새 조건부 관계망\n크기 변화 통제 후 채택 연결: 5개 fold 모두 0개',fontsize=13,va='top')
    fig.tight_layout(rect=(0,.12,1,.9));save(fig,'03_RELATIONSHIP_INFORMATION_DECOMPOSITION','원래 M3 최종 분류기의 성적과 다른, 구성 점수에 대한 사후 원인 진단입니다. 외부 점수를 이미 확인한 후속 분석이며 새 가중치·부호 반전·후보 선택에 사용하지 않았습니다.')

def example(case,seed=0):return read('extended_synthetic/fresh/'+case+'_EXAMPLES.json')['curves'][seed]
def responses():
    fig,axes=plt.subplots(2,2,figsize=(15,8),gridspec_kw={'height_ratios':[2,1]});cut=read('calibration/fresh/DONE.json')['thresholds']['FRESH'];fig.suptitle('일시적 충격 이후 상태와 증거가 회복되는가',fontsize=22,fontweight='bold');shown_states=set()
    for col,case in enumerate(('O1','OB16')):
        r=example(case);a=np.array(r['treatment']);b=np.array(r['control']);at=r['scheduled'];lo=max(0,at-64);hi=at+320;t=np.arange(lo,hi)-at
        ax=axes[0,col];ax.plot(t,a[lo:hi,1],color=COLORS[4],lw=2,label='충격 관측');ax.plot(t,b[lo:hi,1],color='#52606a',ls='--',lw=1.3,label='대응 정상');ax.axhline(cut,color='#777',ls=':',label='고정 임계값');ax.axvline(0,color='#bbb');ax.axvline(r['intervention_end']-at+128,color='#3a9e8f',ls='--',lw=1);ax.set_title('단일 충격' if case=='O1' else '16시점 연속 충격');ax.set_ylabel('4개 새 구간의 지속 증거');ax.legend(fontsize=9);clean(ax)
        shown_states.update(map(int,np.r_[a[lo:hi,0],b[lo:hi,0]]));axes[1,col].imshow(np.vstack((a[lo:hi,0],b[lo:hi,0])),aspect='auto',interpolation='nearest',cmap=ListedColormap(COLORS),vmin=0,vmax=6,extent=(lo-at,hi-at,1.5,-.5));axes[1,col].set_yticks([0,1],['충격 경로','대응 정상']);axes[1,col].set_xlabel('충격 위치 기준 상대 시점')
    fig.legend(handles=[Rectangle((0,0),1,1,color=COLORS[s],label=STATE_KO[s]) for s in sorted(shown_states)],loc='lower center',bbox_to_anchor=(.5,.075),ncol=4,fontsize=10)
    fig.tight_layout(rect=(0,.20,1,.92));save(fig,'04_TRANSIENT_SHOCK_AND_RECOVERY','각 사례는 사전에 고정한 seed 0입니다. 두 경로는 동일 H와 동일 잡음을 공유합니다. 초록 세로선 이후 모든 출력의 일치를 512개 paired series씩 별도 확인했습니다. 정상/충격은 대회 target을 지우는 상태가 아닙니다.')

def persistent():
    fig,axes=plt.subplots(2,3,figsize=(16,8),gridspec_kw={'height_ratios':[1.4,1]});cut=read('calibration/fresh/DONE.json')['thresholds']['FRESH'];fig.suptitle('지속 변화의 종류에 따라 분별력이 다릅니다',fontsize=21,fontweight='bold')
    for col,(case,title) in enumerate([('D2','자기상관 부호 반전'),('V2','분산 2배'),('V4','분산 절반')]):
        r=example(case);a=np.array(r['treatment'])[:1024];b=np.array(r['control'])[:1024];at=r['scheduled'];t=np.arange(len(a));ax=axes[0,col];ax.plot(t,a[:,1],color=COLORS[4],label='지속 증거',lw=1.6);ax.plot(t,b[:,1],color='#82939c',label='대응 정상',lw=1,ls='--');ax.axhline(cut,color='#444',ls=':');ax.axvline(at,color='#777',ls='--');ax.set_title(title);ax.set_ylabel('지속 증거');clean(ax)
        blocks=a[a[:,4]==1,12:];im=axes[1,col].imshow(blocks.T,aspect='auto',cmap='RdBu_r',vmin=-4,vmax=4,extent=(0,len(a),7.5,-.5));axes[1,col].set_yticks(range(8),NODE_KO if col==0 else ['']*8);axes[1,col].set_xlabel('현재까지의 online 시점');axes[1,col].axvline(at,color='#777',ls='--')
    axes[0,0].legend(fontsize=8);fig.subplots_adjust(top=.87,bottom=.14,left=.12,right=.88,wspace=.25,hspace=.3);cax=fig.add_axes([.915,.15,.012,.25]);fig.colorbar(im,cax=cax,label='H 상대 구간값')
    save(fig,'05_PERSISTENT_DEPENDENCE_SCALE_UP_SCALE_DOWN','seed 0을 결과와 무관하게 사용했습니다. 분산 감소에 대한 낮은 검출력을 숨기지 않습니다. 세로선의 tau는 사후 그림 표시이며 실시간 입력이 아닙니다.')

def transport():
    rr=rows('CALIBRATION_TRANSPORT_SUMMARY.csv');fig,axes=plt.subplots(2,2,figsize=(13,9));fig.suptitle('정상 기준 H가 달라지면 경보율도 달라집니다',fontsize=22,fontweight='bold')
    for ax,case in zip(axes.ravel(),('N0','N1','N2','N3')):
        for horizon,color in ((1024,'#407b9c'),(4096,'#c54f57')):
            for fraction,ls in ((0.,'-'),(.01,'--')):
                p=sorted([r for r in rr if r['background']==case and int(r['horizon'])==horizon and float(r['H_contamination_fraction'])==fraction],key=lambda r:int(r['H_length']));x=[int(r['H_length']) for r in p];y=np.array([float(r['alarm_rate']) for r in p]);lo=np.array([float(r['ci_lo']) for r in p]);hi=np.array([float(r['ci_hi']) for r in p]);ax.errorbar(x,y,yerr=np.vstack((y-lo,hi-y)),color=color,ls=ls,marker='o',ms=4,capsize=3,label=f'{horizon}시점 / H '+('1% 충격' if fraction else '정상'))
        ax.axhline(.05,color='#777',ls=':',lw=1);ax.set_title(case);ax.set_xlabel('H 길이');ax.set_ylabel('변화 없는 경로의 경보율');clean(ax)
    axes[0,0].legend(fontsize=8);fig.tight_layout(rect=(0,.09,1,.92));save(fig,'06_H_REFERENCE_AND_HORIZON_TRANSPORT','각 점 512개 series, 95% Wilson 구간. 임계값은 H=2048·1,024시점 calibration에서 고정했습니다. 새 H 길이와 오염 결과에 맞춰 재보정하지 않았습니다. 5% 선은 설계 목표이며 anytime 보장이 아닙니다.')

def power():
    rr=rows('COHERENT_INTERACTION_POWER_SUMMARY.csv');fig,axes=plt.subplots(1,2,figsize=(14,6),gridspec_kw={'width_ratios':[5,3]});fig.suptitle('관계 변화 검사기의 검출력과 잘못된 연결 채택',fontsize=21,fontweight='bold');labels=[];matrix=[];null=[]
    for occupancy in ('uniform','central80percent'):
        for n in (64,256,1024):
            labels.append(('균등 상태' if occupancy=='uniform' else '정상 상태 80%')+f' / {n} series');matrix.append([float(next(r for r in rr if r['kind']=='changed_link' and r['occupancy']==occupancy and int(r['groups'])==n and float(r['coupling'])==c)['true_edge_selection_rate']) for c in (0,.15,.3,.6,.9)]);null.append([float(next(r for r in rr if r['kind']==kind and r['occupancy']==occupancy and int(r['groups'])==n and float(r['coupling'])==c)['any_false_edge_rate']) for kind,c in [('changed_link',0.),('stable_link',.6),('marginal_shift',.6)]])
    for ax,mat,xtitle in zip(axes,(matrix,null),(['0','.15','.30','.60','.90'],['독립 정상','안정된 예측 연결','크기만 변화'])):
        ax.imshow(mat,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto');ax.set_xticks(range(len(xtitle)),xtitle,rotation=15);ax.set_yticks(range(6),labels if ax is axes[0] else ['']*6)
        for i,row in enumerate(mat):
            for j,v in enumerate(row):ax.text(j,i,f'{v:.0%}' if v in (0,1) else f'{v:.1%}',ha='center',va='center',color='white' if v>.5 else '#213541',fontweight='bold')
    axes[0].set_title('정답 연결 0 → 4의 채택률 / 결합 강도');axes[1].set_title('하나 이상의 잘못된 연결 채택률');fig.tight_layout(rect=(0,.12,1,.9));save(fig,'07_KNOWN_INTERACTION_POWER_AND_NEGATIVE_CONTROLS','희소 상태의 null 일관성 수정 후 같은 42조건 × 40회 재검증. 실제 주식 자료나 새 대회 분할이 아닙니다. 95% 구간은 COHERENT_INTERACTION_POWER_SUMMARY.csv. 강한 대조군 성공이 RP에서 관계 변화가 없다는 증명은 아닙니다.')

def phase():
    rr=rows('RP_PHASE_OCCUPANCY_AGGREGATE.csv');names=['no_break','pre','onset','earlyPost','establishedPost','laterPost'];vals=[];labels=[]
    for name in names:
        pp=next(r for r in rr if r['phase']==name);vals.append([float(pp[f'state_{j}_mean_series_fraction']) for j in range(7)]);labels.append(name+f' (n={pp["series"]})')
    fig,ax=plt.subplots(figsize=(13,6));left=np.zeros(6)
    for state in range(7):
        x=np.array(vals)[:,state];ax.barh(labels,x,left=left,color=COLORS[state],label=STATE_KO[state]);left+=x
    ax.set_xlim(0,1);ax.set_xlabel('series별 비중을 먼저 계산한 후 평균');handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc='lower center',bbox_to_anchor=(.5,.095));ax.set_title('실제 held-out 경로의 관측 상태 비중',fontsize=20,fontweight='bold');fig.tight_layout(rect=(0,.28,1,1));save(fig,'08_RP_OBSERVABLE_STATE_BY_EVENT_PHASE','공식 tau 기준 고정 phase. 없는 구간은 생성하지 않았습니다. 관측 상태가 정상으로 돌아가도 breakOccurred target은 계속 1입니다. 상태 점수의 진단 TS-AUC는 0.53520이며 BL80 대체 모델로 승격하지 않았습니다.')

if __name__=='__main__':
    for fn in (rules,network,evidence,responses,persistent,transport,power,phase):fn();print(fn.__name__,flush=True)
