"""Render only small numerical result JSON; no competition raw access."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/next6';OUT=BASE/'FIGURES'
plt.rcParams.update({'font.family':['Malgun Gothic','DejaVu Sans'],'axes.unicode_minus':False,'svg.fonttype':'none','font.size':10})

def save(fig,name):
    for ext in ('png','svg'):fig.savefig(OUT/(name+'.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)

def cases():
    d=json.loads((BASE/'CASE_STUDIES_NEXT6.json').read_text(encoding='utf-8'));names={'success':'탐지 성공','delayed':'탐지 지연','false_alarm':'오경보','miss':'변화 누락'}
    labels=[f'{i:02d} {k}\n{n}' for i,(k,n) in enumerate(zip(d['korean_meaning'],d['feature_names']))]
    for category,title in names.items():
        e=d['examples'].get(category)
        if e is None:
            fig,ax=plt.subplots(figsize=(12,4));ax.axis('off');ax.text(.05,.7,title+' 사례 없음',fontsize=22);ax.text(.05,.35,'고정한 fold 0·해시 순서·임계값 기준에 해당하는 사례가 없습니다.\n다른 기준으로 사례를 바꾸지 않았습니다.');save(fig,'06_HELDOUT_'+category);continue
        fig,(top,bottom)=plt.subplots(2,1,figsize=(18,14),gridspec_kw={'height_ratios':[1,3]},sharex=True)
        colors={'M0':'#56636c','M1':'#a88222','M2':'#437f91','M3':'#b64848'}
        for m,p in e['scores'].items():top.plot(p,label=m,color=colors[m],lw=1.2)
        top.axhline(e['thresholds']['M3'],color=colors['M3'],ls=':',label='M3 고정 임계값');top.legend(ncol=5,fontsize=10);top.set_ylabel('검출 점수');top.set_ylim(0,1)
        cmap=ListedColormap(['#67a2c5','#eeeeec','#ce806b','#444444']);norm=BoundaryNorm([-.5,.5,1.5,2.5,3.5],4)
        im=bottom.imshow(e['states'],aspect='auto',interpolation='nearest',cmap=cmap,norm=norm,extent=[-.5,e['points']-.5,15.5,-.5]);bottom.set_yticks(range(16),labels,fontsize=8);bottom.set_xlabel('현재까지 관측한 online 시점 t')
        for ax in (top,bottom):
            if e['tau']>=0:ax.axvline(e['tau'],color='#222222',ls='--',lw=1.2)
            if e['first_alarm'] is not None:ax.axvline(e['first_alarm'],color='#b64848',ls=':',lw=1.2)
        bar=fig.colorbar(im,ax=bottom,ticks=[0,1,2,3],fraction=.025,pad=.015);bar.ax.set_yticklabels(['낮음','정상','높음','퇴화·미확정'])
        fig.suptitle(f"{title} · 실제 held-out series {e['dataset_id']} · {e['points']} points\nM3로 고정한 사례 선정, M0·M1·M2도 함께 표시",fontsize=20)
        fig.text(.03,.008,f"{d['scope']} | 단일 사례, 점별 신뢰구간 없음. 전체 불확실성은 PAIRED_UNCERTAINTY_NEXT6.csv\n"
            +'모든 노드는 같은 raw 입력의 계산식 유래 피처입니다. 검은 파선은 사후 표시한 공식 tau이며 모델 입력이 아닙니다.\n'
            +'상태 경계는 이 series의 H pseudo-online에서만 계산합니다. 전체 길이·미래 관측·정답을 runtime에 전달하지 않습니다.',fontsize=10)
        fig.subplots_adjust(left=.36,right=.95,top=.9,bottom=.09,hspace=.1);save(fig,'06_HELDOUT_'+category)

def alignment():
    d=json.loads((BASE/'CORRELATION_ATLAS/PSEUDO_AND_NOT_YET.json').read_text(encoding='utf-8'));records={r['name']:r for r in d['results']};labels=[f'{i} {k}' for i,k in enumerate(d['korean_meaning'])]
    for name in ('pseudo_null_pre','pseudo_null_onset','matched_not_yet_delta_pre','matched_not_yet_delta_onset','matched_not_yet_delta_earlyPost','matched_not_yet_delta_establishedPost','matched_not_yet_delta_laterPost'):
        r=records.get(name)
        if not r:continue
        fig,ax=plt.subplots(figsize=(12,11));im=ax.imshow(r['mean'],vmin=-.5,vmax=.5,cmap='RdBu_r');ax.set_xticks(range(16),labels,rotation=90,fontsize=8);ax.set_yticks(range(16),labels,fontsize=8)
        ax.set_title(f"{name}\nouter 0 학습 구간 · {r['series']} series · 추세/공통 크기 통제",loc='left',fontsize=15);fig.colorbar(im,ax=ax,shrink=.6)
        fig.text(.02,.008,'같은 절대 시점의 not-yet 대조 또는 정답·최종 길이와 독립인 pseudo-event.\n동일 raw 파생 피처이며 인과관계 아님. cell별 support·200회 series bootstrap CI·실제 피처 이름은 원본 JSON.\n대조 series 재사용을 포함한 기술통계이며 선행 신호의 확증이 아닙니다.',fontsize=9)
        fig.tight_layout(rect=[0,.08,1,1]);save(fig,'07_ALIGNMENT_'+name)

def window_confound():
    a=json.loads((BASE/'CORRELATION_ATLAS/TRAIN_ATLAS.json').read_text(encoding='utf-8'));b=json.loads((BASE/'CORRELATION_ATLAS/PSEUDO_AND_NOT_YET.json').read_text(encoding='utf-8'))
    atlas={(r['phase'],r['lag'],r['mode']):r for r in a['results']};matched={r['name']:r for r in b['results']};phases=['pre','onset','earlyPost','establishedPost','laterPost'];i,j=9,6
    event=[atlas[p,0,'partial']['mean'][i][j] for p in phases];pseudo=[matched['pseudo_null_'+p]['mean'][i][j] for p in phases];delta=[matched['matched_not_yet_delta_'+p]['mean'][i][j] for p in phases]
    low=[matched['matched_not_yet_delta_'+p]['CI_low'][i][j] for p in phases];high=[matched['matched_not_yet_delta_'+p]['CI_high'][i][j] for p in phases]
    fig,axes=plt.subplots(1,2,figsize=(17,7));ax=axes[0];ax.plot(phases,event,'o-',label='실제 변화 정렬');ax.plot(phases,pseudo,'s-',label='변화 없는 raw의 가상 시점');ax.axhline(atlas['normal_no_break',0,'partial']['mean'][i][j],ls='--',color='#777',label='정상 전체 길이')
    ax.set_title('짧은 창에서는 변화 없는 경우도 큰 상관');ax.set_ylabel('추세·공통 크기 통제 부분 순위 상관');ax.legend();ax.tick_params(axis='x',rotation=25)
    ax=axes[1];ax.errorbar(phases,delta,yerr=[np.array(delta)-low,np.array(high)-delta],fmt='o',capsize=5,color='#397f91');ax.axhline(0,ls='--',color='#777');ax.set_title('같은 시점·같은 길이의 not-yet 대조와 차이');ax.set_ylabel('변화 series - H 매칭 not-yet series');ax.tick_params(axis='x',rotation=25)
    fig.suptitle('창 길이 효과를 구조변화 관계로 오해하지 않기\n09 원 에너지 상방 ↔ 06 에너지 장기 지속',fontsize=19)
    fig.text(.04,.012,'outer 0 학습 구간의 설명용 사후 선택 pair. node09 next4_log_semitail_raw_energy_upper_current / node06 serial_variance__raw_energy_ewma32\n'
        +'같은 raw·필터에서 유래한 관계. 각 phase support는 원본 JSON. 오른쪽은 200회 event-series bootstrap의 기술적 CI(대조 재사용 포함).\n'
        +'전체 길이 정상 지도와 8시점 onset 지도의 직접 차이만으로 변화 신호를 주장할 수 없습니다. edge 선택이나 모델을 바꾸지 않았습니다.',fontsize=10)
    fig.tight_layout(rect=[0,.14,1,.86]);save(fig,'08_WINDOW_LENGTH_CONFOUND')

if __name__=='__main__':
    if (BASE/'CASE_STUDIES_NEXT6.json').exists():cases()
    alignment()
    window_confound()
