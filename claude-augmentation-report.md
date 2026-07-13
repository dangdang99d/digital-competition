# Prompt Template Augmentation 실험 정리 (seongmin)

Granite 라우터 학습에 사용한 label-preserving prompt augmentation 코드와 실험 결과 정리.

## 구현 방식

핵심 모듈은 `src/action_router/augmentation.py`의 `build_augmented_granite_examples()`.

아이디어: **같은 클래스의 실제 train-fold 샘플에서 `session_meta`/`history`를 그대로 복사하고, `current_prompt`만 클래스별 고정밀 템플릿으로 교체**한 synthetic 예제를 학습 데이터에 추가한다. 라벨은 보존되고, source 샘플을 train fold에서만 뽑기 때문에 validation leakage가 없다.

두 가지 모드:

| 모드 | 템플릿 | 설명 |
|---|---|---|
| `contrastive` | `PROMPT_TEMPLATES` | 한/영 혼합 템플릿. exploration 4클래스(`read_file`, `grep_search`, `list_directory`, `glob_pattern`) 중심이고 `run_bash`, `run_tests`, `lint_or_typecheck`, `ask_user`, `plan_task`, `web_search`까지 10클래스 템플릿 보유 |
| `translation` | `TRANSLATION_PROMPT_TEMPLATES` | 한국어 프롬프트를 가진 샘플만 source로 골라(`_looks_korean_prompt`) 영어 intent paraphrase로 교체. ko→en 번역 스타일 augmentation |

템플릿의 `{path}`, `{symbol}`, `{dir}`, `{pattern}`, `{command}` 슬롯은 `SLOTS` 딕셔너리의 현실적인 값(`src/auth.ts`, `useAuth`, `**/*_test.go` 등)으로 랜덤 채움. 클래스별 생성 개수는 `min(max_aug_per_class, source_pool × template 수)`로 제한.

### 사용법

`scripts/train_granite_router.py` (full fine-tune), `scripts/train_granite_lora_router.py` (LoRA) 양쪽에서 CLI로 제어:

```bash
python scripts/train_granite_router.py \
  --augmentation exploration_contrastive \   # 또는 exploration_translation / none
  --aug-classes read_file,grep_search,list_directory,glob_pattern \
  --aug-max-per-class 250
```

## 실험 결과 (GroupKFold 5, session id group, logit bias 튜닝 후 Macro-F1)

베이스라인: Granite h16/l512 full fine-tune fold0 tuned = **0.73906**, LoRA fold0 tuned = **0.75337**

| 날짜 | 실험 | tuned F1 | 결론 |
|---|---|---|---|
| 07-08 | exploration 4클래스, 1000/class | 0.73465 | ❌ 과다 투여 — baseline보다 하락 |
| 07-08 | exploration 4클래스, 250/class | **0.74149** | ✅ full fine-tune fold0 신기록 (+0.00243). read/glob F1 개선 |
| 07-08 | 10클래스 확장, 250/class | 0.73035 | ❌ 넓게 뿌리면 raw/tuned 모두 저하 |
| 07-08 | `grep_search` 단독, 250/class | 0.73960 | ❌ `grep→read` 오분류가 351→562로 오히려 악화 |
| 07-08 | exploration 250/class, fold1 | 0.75104 | ⚠️ raw는 소폭 개선이나 tuned는 fold1 baseline(0.75159) 미달 |
| 07-08 | LoRA + exploration 250/class | 0.74881 | ❌ LoRA baseline(0.75337) 미달 — LoRA로 전이 안 됨 |
| 07-10 | translation 스타일, 100/class | 0.73972 | ❌ raw(0.73548)는 개선되나 tuned에서 exploration250 미달, `grep→read` 혼동 악화 |

## 교훈

1. **Dose-sensitive**: 1000/class는 해롭고 250/class가 sweet spot. 템플릿 augmentation은 소량만.
2. **좁게 타겟해야 함**: 가장 혼동이 심한 exploration 경계(read/grep/list/glob)만 노릴 때 효과. 실행/계획/검색 클래스로 넓히면 실패.
3. **단일 클래스 augmentation은 경계를 한쪽으로 밀 뿐**: grep 단독 augmentation은 `read→grep`은 줄였지만 `grep→read`를 크게 악화.
4. **LoRA로 전이 안 되고 fold1에서 불안정**: fold0 full fine-tune 한정 신호로 취급. 아직 제출 경로 아님.
5. **translation augmentation은 raw-F1 신호만 있음**: bias 튜닝을 거치면 이득이 사라짐. 재시도한다면 훨씬 좁은 클래스 특정 설정 + confusion 우선 검증 필요.

## 파일 구성

```
src/action_router/augmentation.py   # 템플릿 + build_augmented_granite_examples()
src/action_router/features.py       # render_granite_sample (직렬화, augmentation 의존성)
src/action_router/constants.py      # LABEL2ID
scripts/train_granite_router.py     # full fine-tune 학습 스크립트 (--augmentation 옵션)
scripts/train_granite_lora_router.py# LoRA 학습 스크립트 (--augmentation 옵션)
```
