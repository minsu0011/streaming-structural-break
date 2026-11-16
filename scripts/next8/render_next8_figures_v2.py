"""Eight evidence-only scientific figures; run with the existing figure venv."""
from pathlib import Path
import sys
import csv
import json
import hashlib
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next8
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from src.next8.common import BASE,read_json,digest,atomic,utc_now,activity,note

DETAILED=(0,6,9,11,15,21,51,64,77,81,93,105)
COLORS={'M0':'#202a35','M1':'#929aa2','M2':'#984f9f','M3':'#c38c20','M4':'#32917c','M5':'#1468bc','M6':'#c24b45'}
COMMON='H60 fit → 동일 엔진 H40 calibration → Online, 정규화 F 범위 ±12. 모든 피처는 같은 원시 시계열을 공유합니다.'


def read_verified(directory,name):
    directory=Path(directory);receipt=read_json(directory/'DONE.json')
    assert digest(directory/name)==receipt['files_sha256'][name]
    return read_json(directory/name)


def caption(fig,lines):
    fig.text(.055,.025,'\n'.join([*lines,COMMON]),ha='left',va='bottom',fontsize=8,color='#394754')


def empty(ax,message):
    ax.text(.5,.5,message,ha='center',va='center',transform=ax.transAxes,fontsize=12,linespacing=1.7,color='#54616d')
    ax.set_xticks([]);ax.set_yticks([])


def atlas_figure(features):
    fig,ax=plt.subplots(figsize=(14,8));fig.subplots_adjust(left=.12,right=.94,top=.88,bottom=.21)
    domains=[f'TIME_{k}' for k in range(7)]+[f'H_LENGTH_{k}' for k in range(5)]
    values=np.full((len(DETAILED),len(domains)),np.nan);support=np.zeros(values.shape,dtype=int);stars=np.zeros(values.shape,dtype=bool)
    for ji,j in enumerate(DETAILED):
        for di,domain in enumerate(domains):
            rows=[r for f in range(5) for r in features[f,j]['atlas'] if r['mode']=='RAW_F' and r['domain']==domain and r['status']=='SUPPORTED']
            support[ji,di]=len(rows)
            if rows:
                values[ji,di]=np.mean([r['TS_AUC'] for r in rows])-.5
                stars[ji,di]=max(sum(r['ci_low']>.5 for r in rows),sum(r['ci_high']<.5 for r in rows))>=4
    cm=plt.get_cmap('RdBu_r').copy();cm.set_bad('#e6e9ec')
    im=ax.imshow(values,vmin=-.12,vmax=.12,cmap=cm,aspect='auto')
    for (y,x),v in np.ndenumerate(values):
        text='미지원' if not np.isfinite(v) else f'{v:+.3f}'+('*' if stars[y,x] else '')+f'\n{support[y,x]}/5'
        ax.text(x,y,text,ha='center',va='center',fontsize=7,color='white' if np.isfinite(v) and abs(v)>.085 else '#24303a')
    ax.set_yticks(range(len(DETAILED)),[f'F{j:03d}' for j in DETAILED]);ax.set_xticks(range(len(domains)),domains,rotation=40,ha='right')
    ax.axvline(6.5,color='white',lw=3);fig.colorbar(im,ax=ax,shrink=.8,label='Raw F same-time AUC - 0.5')
    ax.set_title('01 | 시간·H 배경별 실제 피처 방향: 사전 지정 12개',loc='left',fontsize=17,pad=18)
    caption(fig,['HELD-OUT outer 5-fold, stride 8. 셀은 지원되는 fold 평균과 fold 수입니다. 153개 전체 표는 별도 보존합니다.',
        '*: 4개 이상 fold의 개별 근사 군집 CI가 같은 방향으로 0.5를 벗어남. 최종 불변/가변 인증을 뜻하지 않습니다.',
        '지원: ≥64 시계열 및 각 정답 ≥32 시계열, 같은 시점 비교 가능. 시점/H 구간은 사전 고정했습니다.'])
    return fig


def relationship_figure(features,certification):
    fig,(ax,bx)=plt.subplots(1,2,figsize=(14,7),gridspec_kw={'width_ratios':[1.25,1]});fig.subplots_adjust(bottom=.20,top=.86,wspace=.35)
    for method,offset,color in [('O0',-.12,'#727d89'),('O1',.12,'#1468bc')]:
        means=[]
        for ji,j in enumerate(DETAILED):
            vals=[features[f,j]['fixed_refit'].get('ranks',{}).get(method,{}).get('TS_AUC') for f in range(5)]
            good=[v for v in vals if v is not None];means.append(np.mean(good) if good else np.nan)
            ax.scatter(np.full(len(good),ji+offset),good,s=16,alpha=.5,color=color)
        ax.plot(np.arange(len(DETAILED))+offset,means,'o-',color=color,label=method+' fold 평균',lw=1.1,ms=5)
    ax.axhline(.5,color='#b6bdc5',ls='--');ax.set_xticks(range(len(DETAILED)),[f'F{j:03d}' for j in DETAILED],rotation=55)
    ax.set_ylabel('Held-out same-time TS-AUC');ax.set_title('단순 선형 O0 / 고정 비선형 O1');ax.legend(fontsize=9)
    units=certification['conditional_units']
    supported=[r for r in units if r.get('direct_difference_ci_low') is not None]
    if not supported:empty(bx,'지원되는 조건부 방향 대비 없음\n선택·지원 기준을 통과하지 않은 비교는\n0 효과나 성공 사례로 채우지 않았습니다.')
    else:
        for i,r in enumerate(supported):
            x=r['direct_state_difference'];lo=r['direct_difference_ci_low'];hi=r['direct_difference_ci_high']
            bx.plot([lo,hi],[i,i],color='#1468bc');bx.plot(x,i,'o',color='#1468bc')
        bx.axvline(0,color='#b6bdc5',ls='--');bx.set_yticks(range(len(supported)),[f"F{r['feature']:03d} {r['orientation_model']} {r['state_pair']}" for r in supported],fontsize=7)
        bx.set_xlabel('조건 간 raw F AUC 차이 / paired 95% CI')
    bx.set_title('동결된 상태 조건부 방향 대비')
    fig.suptitle('02 | 고정 비선형 관계와 조건부 관계를 분리',fontsize=17,x=.07,ha='left')
    caption(fig,['HELD-OUT. 왼쪽 작은 점은 개별 fold이며 오차막대로 해석하지 않습니다. 오른쪽은 실제 반복 규칙의 군집 bootstrap CI입니다.',
        'O0/O1은 해당 outer TRAIN으로 맞춘 방향 모형입니다. 조건부 대조는 공통 TRAIN F 범위와 고정·상태 전용·트리 대조군을 요구합니다.'])
    return fig


def timeline_figure(data):
    fig,axes=plt.subplots(6,1,figsize=(14,11),sharex=True);fig.subplots_adjust(left=.13,right=.96,top=.91,bottom=.14,hspace=.18)
    t=data['time'];axes[0].plot(t,data['F'],color='#1468bc',lw=1.)
    for value in (data['physical_F_H_low'],data['physical_F_H_high']):axes[0].axhline(value,color='#9da7b1',ls=':',lw=1)
    axes[0].set_ylabel('관측 F')
    axes[1].step(t,data['value_sign'],where='post',color='#697d91');axes[1].set_ylabel('VALUE_SIGN');axes[1].set_yticks([-1,0,1])
    axes[2].step(t,np.asarray(data['observed_movement_direction'])-1,where='post',color='#32917c');axes[2].set_ylabel('관측된\n증감 h=1');axes[2].set_yticks([-1,0,1])
    axes[3].step(t,data['physical_F_H_state'],where='post',label='F[t] 관측 후 물리 상태',color='#202a35')
    axes[3].step(t,data['predicted_graph_state_known_before_point'],where='post',label='F[t] 관측 전 T3 조건',color='#c38c20',alpha=.8)
    axes[3].set_ylabel('물리 상태 /\n예측 조건');axes[3].set_yticks([0,1,2,3],['LOW','NORMAL','HIGH','UNKNOWN']);axes[3].legend(fontsize=8,ncol=2)
    axes[4].step(t,data['own_or_H_condition_known_before_point'],where='post',color='#984f9f');axes[4].set_ylabel('관측 전\n자기/H 상태');axes[4].set_yticks([0,1,2,3])
    beta=np.asarray([np.nan if v is None else v for v in data['linear_beta_chosen_before_point']])
    if np.isfinite(beta).any():axes[5].step(t,beta,where='post',color='#c24b45');axes[5].axhline(0,color='#c0c6cc')
    else:axes[5].text(.5,.5,'선택된 가변 피처 방향 규칙 없음',ha='center',va='center',transform=axes[5].transAxes)
    axes[5].set_ylabel('사전 선택\n선형 β');axes[5].set_xlabel('Absolute Online time t')
    chosen='TRAIN 선택 변수' if data['TRAIN_selected_variable'] else '사전 지정 조사 피처 / 가변 인증 주장 없음'
    fig.suptitle(f"03 | F{data['feature']:03d}, series {data['series_id']}, H={data['H_length']} — {chosen}",fontsize=16,x=.06,ha='left')
    caption(fig,['HELD-OUT outer 0 첫 ID, 최대 첫 512개 관측. 관측된 증감과 미래 움직임의 예측을 구분했습니다.',
        'T3 현재 조건은 t-1 발행 h1 예측: 자기 F[t-1], 부모 F[t-2]. UNKNOWN은 사건 확인이 아닙니다.',
        'β는 선형 성분이며 고정 비선형 항이 있으면 전체 국소 방향을 미리 안다고 주장하지 않습니다. 단일 사례는 인증 근거가 아닙니다.'])
    return fig


def graph_figure(graphs,movement):
    fig,axes=plt.subplots(2,3,figsize=(14,10));fig.subplots_adjust(top=.9,bottom=.15,hspace=.2)
    angles=np.linspace(0,2*np.pi,len(DETAILED),endpoint=False);positions={j:(np.cos(a),np.sin(a)) for j,a in zip(DETAILED,angles)}
    panels=[(f'outer {f}: TRAIN 선택',g['edges']) for f,g in enumerate(graphs)]+[('held-out 반복 검증 요약',movement['summary_graph_edges'])]
    for ax,(title,edges) in zip(axes.ravel(),panels):
        for j,(x,y) in positions.items():
            ax.scatter(x,y,s=460,color='#ecf2f7',edgecolor='#8796a4',zorder=3);ax.text(x,y,f'F{j:03d}',ha='center',va='center',fontsize=8,zorder=4)
        for edge in edges:
            ax.annotate('',xy=positions[edge['destination']],xytext=positions[edge['source']],
                arrowprops={'arrowstyle':'->','color':'#1468bc','lw':1.3,'shrinkA':13,'shrinkB':13,'connectionstyle':'arc3,rad=.08'})
        if not edges:ax.text(0,0,'교차 간선 0\n기준 완화 없음',ha='center',va='center',color='#54616d',fontsize=11)
        ax.set_title(title+f' ({len(edges)} edges)',fontsize=11);ax.set(xlim=(-1.3,1.3),ylim=(-1.3,1.3));ax.set_aspect('equal');ax.axis('off')
    fig.suptitle('04 | 실제 선택된 희소 선행 예측 관계망',fontsize=17,x=.06,ha='left')
    caption(fig,['TRAIN 선택과 HELD-OUT 요약을 구분했습니다. 부모 lag 1/4/16, 피처당 최대 2개, 전체 최대 12개입니다.',
        '최종 교차 요인은 자기 과거·H·계산 계보 및 정확한 부모 제거 대조를 통과해야 합니다. 간선 굵기는 인과 강도가 아닙니다.',
        '계산식 파생 관계와 원시 입력 공유는 별도 계보 표에 보존했습니다. 빈 망은 실제 결과이며 가상의 간선을 추가하지 않았습니다.'])
    return fig


def transition_figure(table,interval,movement):
    fig,(ax,bx)=plt.subplots(1,2,figsize=(14,7),gridspec_kw={'width_ratios':[1,1.3]});fig.subplots_adjust(top=.86,bottom=.2,wspace=.36)
    p=np.asarray(interval['point_probability']);lo=np.asarray(interval['ci_low']);hi=np.asarray(interval['ci_high'])
    im=ax.imshow(p,vmin=0,vmax=1,cmap='Blues')
    for (a,b),value in np.ndenumerate(p):ax.text(b,a,f'{value:.3f}\n[{lo[a,b]:.3f}, {hi[a,b]:.3f}]',ha='center',va='center',fontsize=9,color='white' if value>.65 else '#202a35')
    ax.set_xticks([0,1,2],['LOW','NORMAL','HIGH']);ax.set_yticks([0,1,2],[f'{n} (n={s})' for n,s in zip(['LOW','NORMAL','HIGH'],table['own_training_group_support'])])
    ax.set_xlabel('다음 시점 F 상태');ax.set_ylabel('현재 F 상태 / TRAIN 그룹 수');ax.set_title('outer 0 F000 h1: 자기 전이확률')
    r=next(r for r in movement['transition_tests'] if r['feature']==0 and r['horizon']==1)
    labels=[]
    for i,c in enumerate(r['controls']):
        labels.append(c['reference'].replace('_state',''))
        if c['status']=='SUPPORTED':bx.plot([c['ci_low'],c['ci_high']],[i,i],color='#1468bc');bx.plot(c['mean_gain'],i,'o',color='#1468bc')
        else:bx.text(.03,i,'비교 불가 / 동일 망 또는 지원 부족',transform=bx.get_yaxis_transform(),fontsize=8,color='#687581',va='center')
    bx.set_yticks(range(len(labels)),labels);bx.axvline(0,color='#a7b0b8',ls='--');bx.set_xlabel('Reference log loss - T3 log loss');bx.set_title('HELD-OUT proper-loss 개선과 95% CI')
    bx.set_ylim(-.35,len(labels)-.5)
    fig.suptitle('05 | 상태 전이확률과 예측 성능의 불확실성을 분리',fontsize=17,x=.06,ha='left')
    caption(fig,[f"왼쪽: TRAIN {interval['groups']}개 원래 그룹, 2,000 bootstrap. 상태 정의 고정 조건부의 셀별 95% 구간입니다.",
        '오른쪽: 원래 outer fold별 그룹 bootstrap, 동일 표본 비교. T4가 동일하면 추가 망 가치의 증거로 세지 않습니다.',
        '작은 상태표는 전체 AR/ARCH/EWMA 기억을 포함한 완전한 Markov 상태가 아닙니다. 지원 <64 상태는 실행 시 prior로 물러납니다.'])
    return fig


def forecast_figure(movement):
    fig,axes=plt.subplots(1,3,figsize=(15,9),sharey=True);fig.subplots_adjust(top=.87,bottom=.17,wspace=.25)
    for ax,h in zip(axes,(1,4,16)):
        for ji,j in enumerate(DETAILED):
            own=next(r for r in movement['own_dynamics'] if r['feature']==j and r['horizon']==h)
            cross=next(r for r in movement['factor_ablations'] if r['feature']==j and r['horizon']==h and r['domain']=='ALL' and r['reference']=='D5_delta' and r['candidate']=='D5_CROSS_delta' and r['metric']=='MSE')
            for offset,r,color,label in [(-.12,own,'#1468bc','D0 - D1 자기 과거'),(.12,cross,'#c38c20','D5 - Cross 교차 추가')]:
                if r['status']=='SUPPORTED':
                    ax.plot([r['ci_low'],r['ci_high']],[ji+offset]*2,color=color,lw=1.6)
                    ax.plot(r['mean_gain'],ji+offset,'o',color=color,ms=4,label=label if ji==0 else None)
        ax.set_title(f'horizon {h}');ax.axvline(0,color='#a7b0b8',ls='--');ax.set_xlabel('Reference MSE - candidate MSE');ax.grid(axis='y',alpha=.15)
    axes[0].set_yticks(range(len(DETAILED)),[f'F{j:03d}' for j in DETAILED]);axes[0].invert_yaxis();axes[0].legend(fontsize=8,loc='best')
    fig.suptitle('06 | 자기 이력 모형과 교차 피처 추가의 오차 대조',fontsize=17,x=.055,ha='left')
    caption(fig,['HELD-OUT 5-fold, 전체 시계열 그룹 평균 손실과 2,000 bootstrap 95% CI. 오른쪽이면 해당 대조군보다 개선입니다.',
        'D0는 현재 값 유지입니다. 이를 이긴 것만으로 기억의 추가 가치를 입증하지 않습니다. 별도 H 중심값 대조 결과를 함께 확인해야 합니다.',
        'h1/4/16을 모두 보존했습니다. D5는 H 및 공유 계산 입력을 포함합니다. 자기 예측성이 구조변화 분별력이나 새로운 원인을 뜻하지 않습니다.'])
    return fig


def cases_figure(cases):
    fig,axes=plt.subplots(2,2,figsize=(14,10));fig.subplots_adjust(top=.89,bottom=.17,hspace=.4,wspace=.25)
    titles={'success':'성공: D1 오차가 D0보다 ≥10% 작음','failure':'실패: D1 오차가 D0보다 ≥10% 큼','delay_or_miss':'실제 반전 놓침 / 뒤늦은 예측','false_alarm':'반전 오경보'}
    for ax,key in zip(axes.ravel(),titles):
        c=cases['cases'][key];ax.set_title(titles[key],fontsize=12)
        if c.get('status')=='NO_QUALIFYING_CASE_UNDER_FROZEN_RULE':empty(ax,'고정 기준을 만족하는 사례 없음');continue
        ax.plot(c['trace_time'],c['trace_F'],color='#202a35',lw=1,label='실제 관측 F')
        ax.plot(c['forecast_target_time'],c['predicted_future_F'],'.-',color='#1468bc',lw=.9,ms=3,label='발행 시점의 D1 미래 F 예측')
        event=c.get('miss_time',c.get('false_forecast_time'))
        if event is not None:ax.axvline(event,color='#c24b45',ls='--',lw=1)
        if c.get('first_later_forecast_time') is not None:ax.axvline(c['first_later_forecast_time'],color='#c38c20',ls=':',lw=1)
        ax.set_xlabel(f"t | F{c['feature']:03d}, h={c['horizon']}, {c['scope']}, ID={c['series_id']}, H={c['H_length']}",fontsize=8)
        ax.set_ylabel('정규화 F');ax.legend(fontsize=7,loc='best');ax.grid(alpha=.12)
    fig.suptitle('07 | 실제 움직임 예측의 성공·실패·놓침·오경보',fontsize=17,x=.06,ha='left')
    caption(fig,['HELD-OUT 설명용 사례: 사전 고정 feature/horizon/fold/ID/time 순서의 첫 적격 사례이며 가장 큰 효과를 고르지 않았습니다.',
        '성공·실패는 현재 값 유지 대비 오차 기준입니다. 고정 H 중심값보다 좋은 예측이라는 뜻은 아닙니다. 별도 중심값 대조를 보존했습니다.',
        '모든 180개 피처·horizon·fold 패널의 적격 수를 보존했습니다. 예측값은 발행 t가 아니라 목표 t+h에 맞춰 그렸습니다.',
        '빨강: 놓친 실제 반전 또는 오경보 발행. 노랑: 이후 첫 >0.5 예측이며 같은 사건을 확인했다고 단정하지 않습니다. 단일 사례는 인증이 아닙니다.'])
    return fig


def model_figure(model_summary,curve_path):
    fig,(ax,bx)=plt.subplots(1,2,figsize=(14,7));fig.subplots_adjust(top=.85,bottom=.22,wspace=.3)
    points=defaultdict(list)
    with Path(curve_path).open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if r['early_only']!='False' or r['method'] not in ('M0','M3','M4','M5','M6') or not r['auc']:continue
            points[r['method'],int(r['time_online'])].append(float(r['auc']))
    for method in ('M0','M3','M4','M5','M6'):
        times=sorted(t for m,t in points if m==method and len(points[m,t])>=4)
        ax.plot(times,[np.mean(points[method,t]) for t in times],color=COLORS[method],lw=1.2 if method in ('M0','M5') else .8,alpha=.85,label=method)
    ax.axhline(.5,color='#b7c0c8',ls='--');ax.set_xlabel('Absolute Online time');ax.set_ylabel('같은 시점 AUC의 지원 fold 평균');ax.set_title('동일 시점 / 동일 held-out 예측 행');ax.legend(fontsize=8)
    records=[r for r in model_summary['comparisons'] if r['reference']=='M0' and r['candidate'] in ('M1','M2','M3','M4','M5','M6')]
    for i,r in enumerate(records):
        if r['status']=='SUPPORTED':bx.plot([r['ci_low'],r['ci_high']],[i,i],color=COLORS[r['candidate']],lw=2);bx.plot(r['mean_delta'],i,'o',color=COLORS[r['candidate']])
        else:bx.text(.03,i,'지원 부족',transform=bx.get_yaxis_transform())
    bx.set_yticks(range(len(records)),[r['candidate'] for r in records]);bx.axvline(0,color='#9ba6af',ls='--');bx.axvline(.0012,color='#c24b45',ls=':',label='최소 평균 Δ .0012')
    bx.set_xlabel('Δ TS-AUC vs matched BL80');bx.set_title('원래 평가 권위: equal outer-fold 평균');bx.legend(fontsize=8)
    fig.suptitle('08 | BL80과 새 모델의 같은 시점 비교',fontsize=17,x=.06,ha='left')
    caption(fig,['HELD-OUT RP6400. 왼쪽은 비교 가능한 fold ≥4인 시점의 설명용 곡선이며 독립 시점 표본이나 공식 최종 평균이 아닙니다.',
        '오른쪽은 동일 원래 그룹의 2,000 paired bootstrap 95% CI입니다. 정상 연구의 BL80 평균 권위는 0.6203652100698587입니다.',
        'M6는 TRAIN 내부에서 허용된 경우에만 BL80 95% + M5 5%입니다. 같은 BL 비율 대조군·초기 학습 일반화 결과는 표에 별도 보존했습니다.'])
    return fig


def main():
    plt.rcParams.update({'font.family':['Malgun Gothic','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    directory=BASE/'FIGURES_V2';directory.mkdir(parents=True,exist_ok=True)
    if any(directory.iterdir()):raise RuntimeError('Existing figures preserved; rendering revision must use a new explicit output directory')
    feature={}
    for fold in range(5):
        for j in DETAILED:
            p=BASE/'feature_evaluation'/f'outer_{fold}'/f'F{j:03d}'/'DONE.json'
            scope=read_json(p.parent.parent/'DONE.json');assert digest(p)==scope['files_sha256'][p.parent.name+'/DONE.json'];feature[fold,j]=read_json(p)
    certification=read_verified(BASE/'feature_certification','NORMAL.json');movement=read_verified(BASE/'movement_summary','NORMAL.json')
    models=read_verified(BASE/'model_summary','NORMAL.json');data=BASE/'figure_data'
    for receipt_name in ('DATA_DONE.json','REAL_CASES_DONE.json'):assert read_json(data/receipt_name)['status']=='COMPLETE'
    for n,sha in read_json(data/'DATA_DONE.json')['files_sha256'].items():assert digest(data/n)==sha
    assert digest(data/'REAL_CASES.json')==read_json(data/'REAL_CASES_DONE.json')['cases_sha256']
    release=read_json(BASE/'PREDICTION_FREEZE_NEXT8.json');graphs=[]
    for fold in range(5):
        p=BASE/'parent_selection'/f'outer_{fold}'/'GRAPH.json';assert digest(p)==release['files_sha256'][p.relative_to(BASE).as_posix()];graphs.append(read_json(p))
    tasks=[('01_DIRECTION_ATLAS',lambda:atlas_figure(feature)),('02_FIXED_VS_CONDITIONAL',lambda:relationship_figure(feature,certification)),
        ('03_VALUE_MOVEMENT_STATE_DIRECTION',lambda:timeline_figure(read_json(data/'TIMELINE.json'))),('04_SPARSE_PREDICTIVE_GRAPH',lambda:graph_figure(graphs,movement)),
        ('05_TRANSITION_UNCERTAINTY',lambda:transition_figure(read_json(data/'TRANSITION.json'),read_json(data/'TRANSITION_UNCERTAINTY.json'),movement)),
        ('06_OWN_VS_CROSS_FORECAST',lambda:forecast_figure(movement)),('07_REAL_SUCCESS_FAILURE_CASES',lambda:cases_figure(read_json(data/'REAL_CASES.json'))),
        ('08_BL80_SAME_TIME_COMPARISON',lambda:model_figure(models,data/'SAME_TIME_CURVES.csv'))]
    files={}
    with activity('Render eight observed scientific figures','FIGURE_RENDER','B4_MOVEMENT_VS_DISCRIMINATION'):
        with PdfPages(directory/'NEXT8_EIGHT_FIGURES.pdf') as combined:
            for name,build in tasks:
                fig=build();fig.savefig(directory/(name+'.png'),dpi=160);fig.savefig(directory/(name+'.pdf'));combined.savefig(fig);plt.close(fig)
                for suffix in ('.png','.pdf'):files[name+suffix]=digest(directory/(name+suffix))
        files['NEXT8_EIGHT_FIGURES.pdf']=digest(directory/'NEXT8_EIGHT_FIGURES.pdf')
    atomic(directory/'DONE.json',{'status':'COMPLETE','timestamp':utc_now(),'figure_count':8,'files_sha256':files,
        'matplotlib_version':matplotlib.__version__,'numpy_version':np.__version__,'source_sha256':digest(__file__),
        'real_artifact_data_only':True,'no_fictional_discovery_edges_or_filled_missing_results':True,'case_policy_sha256':digest(BASE/'FIGURE_CASE_POLICY_NEXT8.json')})
    note('Complete eight scientific figures','Actual heldout evidence, uncertainty and empty results retained','B4_MOVEMENT_VS_DISCRIMINATION')


if __name__=='__main__':main()
