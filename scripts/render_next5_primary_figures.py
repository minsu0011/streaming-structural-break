"""Standalone research figures from already-frozen aggregate results only."""
from pathlib import Path
import csv
import hashlib
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def rows(path):
    with path.open(encoding='utf-8',newline='') as stream:return list(csv.DictReader(stream))


def main():
    base=ROOT/'artifacts/next5';out=base/'analysis_figures/primary_v1';out.mkdir(parents=True,exist_ok=False)
    sources={};catalog_path=base/'selection_uncertainty/CATALOG_UNCERTAINTY.csv';catalog=rows(catalog_path);sources[str(catalog_path.relative_to(ROOT))]=sha(catalog_path)
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'savefig.facecolor':'white','axes.titleweight':'bold','figure.facecolor':'white'})
    colors={'S':'#1976a3','P':'#d77b23','W':'#4a9062','E':'#9466ad','R':'#c34e6a','M':'#7e7b37','X':'#303f70'}
    figures=[];fig,ax=plt.subplots(figsize=(12,10));positions=np.arange(len(catalog))
    for position,row in zip(positions,catalog):
        mean=float(row['estimate_delta']);lower=float(row['pointwise_ci_lower']);upper=float(row['pointwise_ci_upper'])
        ax.errorbar(mean,position,xerr=[[mean-lower],[upper-mean]],fmt='o',color=colors[row['family']],capsize=3,lw=1.6)
    ax.set_yticks(positions,[row['candidate'] for row in catalog]);ax.invert_yaxis();ax.axvline(0,color='#888888',lw=1)
    ax.axvline(.0012,color='#c23838',lw=1.5,ls='--',label='사전 승격 기준 +0.0012')
    ax.set_xlabel('ResearchPool BL80 대비 5-fold 평균 TS-AUC 차이');ax.set_title('NEXT5: 19개 full 후보의 점수와 고정 OOF 불확실성',loc='left',pad=18)
    ax.grid(axis='x',alpha=.18);ax.legend(loc='lower right',frameon=False)
    fig.text(.02,.02,'2,000회 whole-series paired bootstrap. 후보 설계·선택의 적응성과 학습 불확실성을 보정한 구간이 아닙니다.\nX2 실제 개선 +0.001195650019 < +0.001200000000. M2는 screen 실패로 full 점수가 없습니다.',fontsize=9)
    fig.tight_layout(rect=[0,.07,1,1]);figures.append(('PRIMARY_CANDIDATE_INTERVALS',fig))
    selected=['S4B_UNIFIED_SCALE_SPECIALIST','P1_TWO_TIMESCALE_CONFIRMATION','X2_SCALE_TAIL'];labels=['S4B 통합 scale','P1 지속성 확인','X2 scale × tail']
    tables={}
    for cid in selected:
        path=base/'forensic'/cid/'PAIRED_FRACTURES.csv';tables[cid]={r['scope']:r for r in rows(path)};sources[str(path.relative_to(ROOT))]=sha(path)
    fig,axes=plt.subplots(1,2,figsize=(14,6));menus=[(['time_0_9','time_10_24','time_25_49','time_50_99','time_100_199','time_200_399','time_400_plus'],['0–9','10–24','25–49','50–99','100–199','200–399','400+'],'온라인 시간'),
        (['age_0_4','age_5_9','age_10_19','age_20_49','age_50_plus'],['0–4','5–9','10–19','20–49','50+'],'변화 이후 경과 시간')]
    for ax,(keys,ticks,title) in zip(axes,menus):
        for k,(cid,label) in enumerate(zip(selected,labels)):
            mean=np.array([float(tables[cid][key]['estimate_mean_delta']) for key in keys]);low=np.array([float(tables[cid][key]['ci_2_5']) for key in keys]);high=np.array([float(tables[cid][key]['ci_97_5']) for key in keys])
            ax.errorbar(np.arange(len(keys))+(k-1)*.16,mean,yerr=[mean-low,high-mean],fmt='o',color=colors[cid[0]],capsize=2,label=label,ms=5)
        ax.axhline(0,color='#888888',lw=1);ax.set_xticks(np.arange(len(keys)),ticks,rotation=30);ax.set_title(title);ax.grid(axis='y',alpha=.18);ax.set_ylabel('TS-AUC 차이')
    axes[0].legend(frameon=False,fontsize=9);fig.suptitle('개선이 나타나는 시점과 변화 직후의 불확실성',x=.04,ha='left',fontweight='bold')
    fig.text(.03,.02,'같은 시점의 양성–음성 쌍을 비교합니다. break-age 구간은 해당 양성 시점과 같은 시점의 모든 음성 비교군을 유지합니다.\n구간별 추정치의 단순 합은 전체 점수가 아닙니다. 별도 pair-weighted 기여도는 합이 전체 개선과 정확히 일치함을 검증했습니다.',fontsize=9)
    fig.tight_layout(rect=[0,.1,1,.93]);figures.append(('TIME_AND_BREAK_AGE',fig))
    family_best=[]
    for family in 'SPWERMX':family_best.append(next(row for row in catalog if row['family']==family))
    mechanisms=[('mechanism_DEPENDENCE_DOMINANT','DEPENDENCE (1,227)'),('mechanism_SCALE_DOMINANT','SCALE (106)'),('mechanism_MIXED','MIXED (170)'),
        ('mechanism_MEAN_DOMINANT','MEAN (16)'),('mechanism_TAIL_DOMINANT','TAIL (64)'),('mechanism_WEAK_OR_AMBIGUOUS','WEAK (1,600)')]
    fig,ax=plt.subplots(figsize=(13,6));values=np.zeros((7,len(mechanisms)));support=np.zeros_like(values,dtype=bool)
    for i,row in enumerate(family_best):
        path=base/'forensic'/row['candidate']/'PAIRED_FRACTURES.csv';table={r['scope']:r for r in rows(path)};sources[str(path.relative_to(ROOT))]=sha(path)
        for j,(key,_) in enumerate(mechanisms):values[i,j]=float(table[key]['estimate_mean_delta']);support[i,j]=table[key]['supported']=='True'
    maximum=max(.02,float(np.max(np.abs(values))));picture=ax.imshow(values,cmap='RdBu',vmin=-maximum,vmax=maximum,aspect='auto')
    for i in range(7):
        for j in range(len(mechanisms)):
            text=f'{values[i,j]:+.4f}'+(' *' if not support[i,j] else '')
            ax.text(j,i,text,ha='center',va='center',fontsize=10,color='white' if abs(values[i,j])>.6*maximum else '#111111')
    ax.set_yticks(np.arange(7),[r['candidate'] for r in family_best]);ax.set_xticks(np.arange(len(mechanisms)),[m[1] for m in mechanisms],rotation=20,ha='right')
    ax.set_title('계열 대표 후보의 관측 mechanism별 TS-AUC 차이',loc='left',pad=16);fig.colorbar(picture,ax=ax,label='기준 대비 차이')
    fig.text(.02,.02,'* 지원 기준 100개 양성 시계열 미달: 안정적 결론에 사용하지 않습니다.\nMechanism은 고정된 사후 관측 분류이며 잠재 생성 원인의 정답 또는 모델 입력이 아닙니다. 각 구간은 공통 무변화 비교군을 포함합니다.',fontsize=9)
    fig.tight_layout(rect=[0,.1,1,1]);figures.append(('MECHANISM_SUPPORT',fig))
    with PdfPages(out/'PRIMARY_RESEARCH_FIGURES.pdf') as pdf:
        for name,figure in figures:
            figure.savefig(out/(name+'.png'),dpi=180,bbox_inches='tight');pdf.savefig(figure,bbox_inches='tight');plt.close(figure)
    receipt={'status':'COMPLETE_INTERIM_RESEARCH_FIGURES','scope':'Frozen PRIMARY OOF and diagnostic aggregate tables; ALT, synthetic and final audit remain separate.',
        'matplotlib_version':matplotlib.__version__,'source_sha256':sha(Path(__file__)),'input_sha256':sources,
        'files_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()},'raw_or_Shadow_data_loaded':False}
    (out/'RESULTS.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('NEXT5 RESEARCH FIGURES',len(figures),flush=True)


if __name__=='__main__':main()
