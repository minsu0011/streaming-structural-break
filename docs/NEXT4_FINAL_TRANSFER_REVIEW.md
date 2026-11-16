# NEXT4 최종 과학적 채택 검토

작성 UTC 2026-09-10T18:18:06.109704+00:00. 연구 단계의 실제 파일 검증을 완료한 시점의 판단이다. **별도 final audit와 fresh raw 재현은 이후 실제 수행해야 하며, 이 문서는 감사 완료를 선행 주장하지 않는다.**

주 성능 모델 BL2_C205_DEPENDENCE_80의 0.6232684128173095를 유지한다. 탐색 최고 B2의 원래 paired CI 하한 -0.000018936916538은 실패로 남긴다. D2_DISCRETE_SCALE_MIXTURE는 제한된 SCALE 강건성 역할의 최종 감사 대상으로 지정한다. 새 performance champion이나 generation은 아니다.

후보 평균 0.6235961690191665, 기준 대비 Δ 0.00032775620185694355, 전체 CI [-0.000273254935296, 0.000909263558055], P(Δ>0) 0.8545. 전체 CI가 0을 포함하므로 평균 개선을 확증하지 않았다.

고정 우선순위는 original SCALE CI 하한, original 전체 gain, ID 순이다. 사후 가장 좋은 ALT나 합성 점수로 순서를 바꾸지 않았다. A2/C3/D3는 고정 네 분산 감소 반례를 모두 통과하지 못했다. B1은 이 네 반례를 통과했지만 original SCALE CI 하한이 더 낮고 H 제외 SCALE 점 추정치도 음수였다.

| 검증 | 전체 Δ | 전체 95% CI | SCALE Δ | SCALE 95% CI |
|---|---:|---|---:|---|
| PRIMARY | 0.000327756201857 | [-0.000273254935296, 0.000909263558055] | 0.00893582821438 | [0.00536269396578, 0.0127629249066] |
| ALT_1 | 0.000431819282534 | [-0.000162679023791, 0.00100653707327] | 0.0087447033613 | [0.00493874519242, 0.0129113460597] |
| ALT_2 | 0.000419280952505 | [-0.000147062479339, 0.00102020881276] | 0.00812152836036 | [0.00395812450582, 0.0126786904745] |
| ALT_3 | 0.000544028053188 | [-6.58078084302e-05, 0.00118791751857] | 0.00993382653819 | [0.00534279003431, 0.0154670688634] |
| LEAVE_ONE_H_REGIME_OUT | -0.000215433674867 | [-0.00111361225639, 0.000719166240329] | 0.00442824954371 | [0.00166864286516, 0.00798753129072] |

SCALE에는 변화 시계열 132개와 null 4056개가 있다. ALT와 H 제외 학습은 같은 DEV 8000개의 다른 grouped partition이다. 독립 데이터가 아니다. H 전체의 matched 최소 Δ는 -0.00268446316194다.

| 원래 기전 | Δ | 95% CI |
|---|---:|---|
| mechanism_DEPENDENCE_DOMINANT | -0.000762643135639 | [-0.00158905207824, 5.39433861992e-05] |
| mechanism_MIXED | 0.00457570980978 | [0.00257091600656, 0.00677582420774] |
| mechanism_WEAK_OR_AMBIGUOUS | 8.81027732083e-05 | [-0.000807934175011, 0.000955055709993] |
| mechanism_observed_scale_increase | 0.00512601266825 | [0.000823508415612, 0.00973866790611] |

raw SD 증가/감소 subset은 53/79개여서 각 100개 미만이다. 이 방향 구분은 latent conditional variance의 정답이 아니다. MEAN 21, TAIL 86도 지원 표본 기준에 못 미친다.

실제 all-DEV 파일의 고정 반례 확인은 512개 matched generating-seed pair를 사용했다. 각 시나리오와 seed, null 보정 방식, 기존 learner/혼합은 그대로이며 새 fit은 해당 후보의 전체 DEV 학습 1회뿐이다. 다음 Δ는 합성 TS-AUC로 실제 DEV 점수와 합산하지 않는다.

| 실제 파일 반례 | Δ | 95% CI |
|---|---:|---|
| V3 | 0.0198439824568 | [0.0180872253747, 0.0215903673799] |
| V4 | 0.0193668392711 | [0.0176446737163, 0.0211967796337] |
| CB_N3_070 | 0.0172859676968 | [0.0155826019094, 0.0190446267324] |
| CB_N3_050 | 0.0110996186166 | [0.00959251938853, 0.0127773985593] |

| 실제 파일 이상치 후 지표 | 후보−BL80 | 95% CI |
|---|---:|---|
| O1 mean_signed_excess | -0.0192979267538 | [-0.0198678358247, -0.0186838110241] |
| O1 mean_signed_excess_final_64 | -0.0183730810786 | [-0.0189722246295, -0.0177300503561] |
| OB16 mean_signed_excess | 0.00476280282419 | [0.00394614533338, 0.00559941734587] |
| OB16 mean_signed_excess_final_64 | 0.00691903814368 | [0.00603897082722, 0.00784542842073] |

음수 excess 차이는 해당 고정 반례에서 score 오염이 줄었다는 뜻이다. 검출력 보존이나 모든 이상치에 대한 보장을 뜻하지 않는다. fold-0의 256쌍과 실제 all-DEV 파일의 512쌍 통계를 혼합하지 않았다. 128개 TreeSHAP seed의 분해는 별도 진단이며 인과적 feature 제거 실험이 아니다.

실제 파일의 반복 이상치 OB16은 평균 excess Δ +0.00476280282419, 마지막 64점 +0.00691903814368로 악화됐다. 두 CI도 양수다. fold-0에서 평균 Δ -0.00367609088426이었던 개선이 전체 학습 파일에서는 유지되지 않았다. 단발 이상치 개선과 반복 이상치 악화를 함께 보고하며, 전반적인 outlier repair라고 부르지 않는다. α=.01의 OB16 outlier alarm fraction도 0.650390625 대 BL80 0.525390625였고, α=.05에서는 0.986328125 대 0.939453125였다.

고정 합성 TS-AUC에서도 모든 조건이 좋아진 것은 아니다. 실제 파일은 AR 변화 D1에서 Δ -0.00717549926562, mixed X1에서 -0.00466391236264, ARCH A1에서 -0.00176538745381, tail T1에서 -0.00131586908997이었다. 강건성 역할의 근거는 미리 고정한 실제 SCALE 기준과 감소 반례 확인이며, 새로운 사후 gate를 만들거나 이 손실을 숨기지 않는다.

| 16384점 null | α | 후보 alarm fraction | BL80 alarm fraction |
|---|---:|---:|---:|
| N0 | 0.01 | 0.080078125 | 0.07421875 |
| N0 | 0.05 | 0.3359375 | 0.330078125 |
| N1 | 0.01 | 0.15234375 | 0.146484375 |
| N1 | 0.05 | 0.388671875 | 0.3984375 |
| N2 | 0.01 | 0.078125 | 0.08203125 |
| N2 | 0.05 | 0.478515625 | 0.46875 |
| N3 | 0.01 | 0.154296875 | 0.146484375 |
| N3 | 0.05 | 0.515625 | 0.546875 |

독립 1024점 null maximum으로 정한 threshold를 긴 horizon에 고정했으므로 anytime 보장은 없다. 부호 반전 결과도 기존 DEV stress이며 새로운 ensemble/augmentation 후보가 아니다.

실제 entry는 300개 prefix 및 DEV 전체 현재값 추론 대조, 20개 역순 reset, 첫 None, 미래 길이/미래 suffix trap을 통과했다. cold read-only import는 2개 새 Python 프로세스에서 확인했다. 고유 mutable NumPy state 배열은 25821 bytes다. 연구 단계 2회 paired 직접 runtime 중앙 비율은 {'ten_million_ratio': 1.8760807062020808, 'scalar_median_ratio': 1.8840579710144927, 'amortized_ratio': 1.678413980429725}이며 동일 시점 BL80과 비교했다. Final audit에서 직접 측정과 raw→새 feature→동일 fit→추론 재현을 별도로 요구한다.

SEAL 네 개는 미개봉이며 NEXT4 reduced/public/cloud 사용은 0이다. DEV saturation은 HIGH로 유지한다. 이 채택은 SCALE의 제한된 support와 고정 진단 범위에 한정한다.

근거 SHA-256:

- `artifacts/next4/gate_reviews/research_final/RESULTS.json`: `c9a77928ae683588a640727def8c39d32fa7c4e97514ef2eb01981f11fb0bff9`
- `artifacts/next4/experiments/D2_DISCRETE_SCALE_MIXTURE/full/RESULTS.json`: `a52b9dd371958441be1ed6168822795fa82373485ff6504c09c8de10aed54859`
- `artifacts/next4/bootstrap/D2_DISCRETE_SCALE_MIXTURE/RESULTS.json`: `a94375a94f9fcb538cb7cafbed85b324fcee0504f4d9248efdce0c2c9123741f`
- `artifacts/next4/forensic/D2_DISCRETE_SCALE_MIXTURE/RESULTS.json`: `e47659be70c087a058b5d49824ec84920055f7eff887582880108ebd3a377cba`
- `artifacts/next4/robustness/D2_DISCRETE_SCALE_MIXTURE/RESULTS.json`: `3d446a84d36e06e4ffca16c5ef1363c2649da33097c8d8855b98af4d3d096964`
- `artifacts/next4/robustness_forensic/D2_DISCRETE_SCALE_MIXTURE/RESULTS.json`: `7c811b5559cc5990a44effde255a5dfdddc6958b294006dc3ba38f626723f89c`
- `artifacts/next4/finalist_nominations/D2_DISCRETE_SCALE_MIXTURE.json`: `d7bd160b5725e3c30972a79f7989ea8445e361cb92a3cf0f6c07cdb5b0d62094`
- `artifacts/next4/refits/D2_DISCRETE_SCALE_MIXTURE/nominated_all_dev_v1/RESULTS.json`: `ae9e922c2b1a96dd90e82e23594d4fa56dd430c7a9292589f4841a14bb58d1d3`
- `artifacts/next4/finalist_validation/D2_DISCRETE_SCALE_MIXTURE/API_D2_DISCRETE_SCALE_MIXTURE/RESULTS.json`: `3dbcdd1cf9c1fbfb52aeb7ba1b9e68cd53fb08706cf8d4dd3397c0304839e436`
- `artifacts/next4/finalist_validation/D2_DISCRETE_SCALE_MIXTURE/COLD/RESULTS.json`: `ae97bc680536484bbfa377967ae7db4433c1970522524a78519685818b64651d`
- `artifacts/next4/latency/D2_DISCRETE_SCALE_MIXTURE_DIRECT/RESULTS.json`: `1104ce4f53293fc5204599cb718bd6304503b5c382c7e9a325166c166e15dddf`
- `artifacts/next4/synthetic/candidates/D2_DISCRETE_SCALE_MIXTURE_all_dev/RESULTS.json`: `468ceede40ad1927c70822afa56eb34ac13eb1fd7fc0a52e0873429d5232c41c`
- `artifacts/next4/memory_diagnostics/D2_DISCRETE_SCALE_MIXTURE_all_dev/RESULTS.json`: `3b6fe17d4a2b0145340274f068b4651e16db4a37f3881d4117e8b357e2d466f6`
- `artifacts/next4/sign_reflection/RESULTS.json`: `7ab45a74b0892dba977b78915289d7b7f80cee0632a4902a29ab07ebd7644c72`
- `artifacts/next4/synthetic_nominations/A1_INNOVATION_PORTMANTEAU_full_fold0.json`: `65ec6b54e480d2ff23961d072c9d6d9e7535d977c5b0e2b579ef098520b9ec8b`
- `artifacts/next4/synthetic/candidates/A1_INNOVATION_PORTMANTEAU_full_fold0/RESULTS.json`: `9591e5124b12b175d7b6c4f52de716ca5837fc902c70ae1e1154867a31300588`
- `artifacts/next4/synthetic_nominations/A2_MAGNITUDE_PORTMANTEAU_full_fold0.json`: `bff732a42ea1141c48e1d0ba0dd52303e926c74daeaf932b113c86c60a5d3499`
- `artifacts/next4/synthetic/candidates/A2_MAGNITUDE_PORTMANTEAU_full_fold0/RESULTS.json`: `9f018d056936bd4cd6682f3eb23e94351ab245a3493e041ebdbcd546f5e44b90`
- `artifacts/next4/synthetic_nominations/A3_SIGN_PORTMANTEAU_full_fold0.json`: `5c8c8607a68567fc879f81505225a0042805e21361e864bb55c9c4427e31bcca`
- `artifacts/next4/synthetic/candidates/A3_SIGN_PORTMANTEAU_full_fold0/RESULTS.json`: `aa2b2df01a1fcd9cc549d93272d26e02a7573cefc56abc2f2e7aa36bb42d3b76`
- `artifacts/next4/synthetic_nominations/B1_SPLIT_ENERGY_BRANCHES_full_fold0.json`: `01128faa62ec3f672a743befee626bb67c537257d9882fe85d24d0c28c2a04dd`
- `artifacts/next4/synthetic/candidates/B1_SPLIT_ENERGY_BRANCHES_full_fold0/RESULTS.json`: `dce5aa294e373512a9763d238b8aed9181031dad67891a14da5f9ff86d349c42`
- `artifacts/next4/synthetic_nominations/B2_LOG_ENERGY_SEMITAIL_all_dev.json`: `c7bddb34afd86caf1cf5b597648c1ae4eb563c7fce018163804ffca6aa500b3a`
- `artifacts/next4/synthetic/candidates/B2_LOG_ENERGY_SEMITAIL_all_dev/RESULTS.json`: `42b4f3af6a2e2c9cc34b06c244f558eb53ea1acbed96cdb5198ec3a21abf8822`
- `artifacts/next4/synthetic_nominations/C1_FIXED_BIN_HELLINGER_full_fold0.json`: `149da16f3df3b0667f230543cd34c28bc02c5a390c28a7f17058ab207cc770c8`
- `artifacts/next4/synthetic/candidates/C1_FIXED_BIN_HELLINGER_full_fold0/RESULTS.json`: `d66922c9b878ba21bd3cdfd021432386c72641e0d33798ed5bbbaa16144ec6e9`
- `artifacts/next4/synthetic_nominations/C2_FIXED_BIN_JS_full_fold0.json`: `a347653cf4be12130d208488330fd00e39524ac65d7de8f6ba9466f878093c1d`
- `artifacts/next4/synthetic/candidates/C2_FIXED_BIN_JS_full_fold0/RESULTS.json`: `0b3eba449ef65250eb9d101f0acd3123dbd06fc1f33da5338f4699de575d1eb0`
- `artifacts/next4/synthetic_nominations/C3_SHAPE_RATIO_DISTRIBUTION_full_fold0.json`: `9e5aa46b952f62ece84ed86618ceb1d97134ed23436932d6a0dd30ca91853a1b`
- `artifacts/next4/synthetic/candidates/C3_SHAPE_RATIO_DISTRIBUTION_full_fold0/RESULTS.json`: `b8335e55666c2cd605b65517580925fae001d1594a4659de6cd53c1ccccbb732`
- `artifacts/next4/synthetic_nominations/D1_DISCRETE_MEAN_MIXTURE_full_fold0.json`: `806d70a55325c6c4cba29ba30e96533bda3d823a28a5b87353154bf4718020c6`
- `artifacts/next4/synthetic/candidates/D1_DISCRETE_MEAN_MIXTURE_full_fold0/RESULTS.json`: `eded199f98709c27dd485e3e6b95d2d7b340c44945d542f5b8a5956a354d1520`
- `artifacts/next4/synthetic_nominations/D2_DISCRETE_SCALE_MIXTURE_all_dev.json`: `b69130d13dc998d8d99b4343122c370d8ed5264ae7858ee1fcb35c929bc44ba1`
- `artifacts/next4/synthetic_nominations/D2_DISCRETE_SCALE_MIXTURE_full_fold0.json`: `140c368b7641d90e9c40f447b159ea0c42a8371876191d8c9dd7ebdc78c6ce75`
- `artifacts/next4/synthetic/candidates/D2_DISCRETE_SCALE_MIXTURE_full_fold0/RESULTS.json`: `d506ea86c18eef189d8f90093676f1bd8989b40eeb4c82960ee231df3c505935`
- `artifacts/next4/synthetic_nominations/D3_HAZARD_MIXTURE_full_fold0.json`: `69ab4c693aa5fc58f93d2ef9bdf547de8732287989087bcae230e5b6fcd5eea7`
- `artifacts/next4/synthetic/candidates/D3_HAZARD_MIXTURE_full_fold0/RESULTS.json`: `3dc778ac0e291d16896639c3ecf25e0e6ae1f624919b719f1276cc3235d7eac9`
- `artifacts/next4/synthetic_nominations/H1_FIXED_LINEAR_STATE_READOUT_full_fold0.json`: `2405991a582e204662ecba5a2aed5fc25642e58d0cb7366a4353bdd9f69de452`
- `artifacts/next4/synthetic/candidates/H1_FIXED_LINEAR_STATE_READOUT_full_fold0/RESULTS.json`: `8e5f8a2bbd9011884f7a2051943e73b9e39b69e9170609a9dd073081db9f3cef`
- `artifacts/next4/forensic/D2_DISCRETE_SCALE_MIXTURE/PAIRED_FRACTURES.csv`: `f51a031e25cfce02a98a60d08c5c6454d3671982a4442ebd3d619e499c59d968`
- `artifacts/next4/synthetic/candidates/D2_DISCRETE_SCALE_MIXTURE_all_dev/PAIRED_SYNTHETIC.csv`: `fd8742a5f25e29dd4c4edee3efbbfc5af35006db4693f4b6d1189ba71a7589b6`
- `artifacts/next4/memory_diagnostics/D2_DISCRETE_SCALE_MIXTURE_all_dev/PAIRED_OUTLIER_EXCESS.csv`: `9ab8caa1928083e821271552a758d1d16bb81946d6d1ec5a7802bddeb03ef7d9`
- `artifacts/next4/synthetic/candidates/D2_DISCRETE_SCALE_MIXTURE_all_dev/LONG_NULL.csv`: `97a60a294a73e815acd1f14e322d951a4f02be0281aebda8e133a203fbb95d73`
