# M-PAGER

**Multi-hop Patient-Aware Graph Enhanced Retrieval**

CREST 임상 가이드라인으로부터 환자 조건(condition)이 결합된 의료 지식 그래프를 구축하는 5-stage 파이프라인. **UMLS 1-hop subgraph**를 골격으로 하고, **LLM이 추출한 4종 구조화 조건**을 엣지 속성으로 부착해 Neo4j에 적재합니다.

| 항목 | 내용 |
| --- | --- |
| 입력 | CREST `xml/` (권고문 + 강도) + `primary/` (HTML 컨텍스트) |
| KG 골격 | UMLS REST API `/relations` 1-hop |
| 의미 분류 | 15 Semantic Group · 127 Semantic Type (TUI) |
| 조건 4종 | `numeric_threshold` / `categorical_state` / `medication_history` / `temporal_condition` |
| 출력 | Neo4j `(:Concept)-[:RELATES {conditions, evidence, ...}]->(:Concept)` |
| 진입점 | [pipeline.py](pipeline.py) — `stage0~4 / all` 서브커맨드 |

---

## 1. 디렉터리 구조

```
M-PAGER/
├── pipeline.py             # 통합 CLI
├── config.py               # 경로·키·동시성·배치 설정
├── cli_utils.py            # 로깅·JSON I/O·prerequisite 체크
├── crest_parser.py         # Stage 0
├── entity_extractor.py     # Stage 1 (LLM NER)
├── semantic_types.py       # 127 Semantic Type + 예시
├── umls_client.py          # UMLS REST (rate-limited, thread-safe)
├── entity_matcher.py       # Stage 2-A (cascading match)
├── subgraph_builder.py     # Stage 2-B (1-hop triples)
├── condition_augmenter.py  # Stage 3 (조건 추출/검증)
├── neo4j_builder.py        # Stage 4 (UNWIND + MERGE, idempotent)
└── output/                 # stage별 산출 JSON
```

---

## 2. Stage별 핵심 정리

### Stage 0 — CREST Parsing
- `xml/`의 `recommendation` 속성 마크업에서 권고문 + strength 추출 (부모-자식 동일 등급 중복 제거)
- `primary/` HTML은 nav/script 제거 후 4,000자로 잘라 `guideline_context` 결합
- 출력: [output/stage0_recommendations.json](output/stage0_recommendations.json) — `{guideline_id, strength, text, tag, guideline_context}`

### Stage 1 — Entity Candidate Extraction (LLM)
- 127 Semantic Type 프롬프트 + few-shot, **recall 우선** NER
- `surface_form` / `normalized_form` / `semantic_group` / `TUI` 출력 → normalized_form 기준 dedup
- 출력: [output/stage1_entity_candidates.json](output/stage1_entity_candidates.json) — unique entity + `source_guidelines[]`

### Stage 2 — UMLS Layer Construction
- **2-A (matching)**: `Exact → NormalizedString → Words` cascading. 약어/구어 사전·suffix·괄호·하이픈 변형으로 후보 확장. TUI ↔ semantic group 일치 필터로 노이즈 차단
- **2-B (1-hop subgraph)**: matched CUI를 **head**, related entity를 **tail**로 통일. `additionalRelationLabel` 우선 사용, `(head_cui, relation, tail_id)` dedup
- 출력: [output/stage2_umls_matched.json](output/stage2_umls_matched.json), [output/stage2_umls_layer_triples.json](output/stage2_umls_layer_triples.json)

### Stage 3 — Condition Augmentation
- **CUI 기반 매칭**: `head_cui → match_results → source_guidelines → recommendations` (preferred name 직접 비교 회피). non-CUI tail은 name fallback
- **Bidirectional filter**: head와 tail이 **모두** 같은 rec에 등장할 때만 LLM 호출 → 비용 절감의 1차 게이트
- **Pair-cache + group dedup**: `(head_cui, tail_id, head_name, tail_name)` 동일 그룹은 LLM 1회 호출 결과 공유
- **검증**: 4종 type별 required field + `condition_logic ∈ {AND, OR, NOT}` 검사. 누락된 `triple_index`는 small-chunk retry, 실패 시 `parse_failed=True`
- 출력: [output/stage3_condition_augmented_triples.json](output/stage3_condition_augmented_triples.json) — triple에 `conditions[]`, `condition_logic`, `condition_source`, `recommendation_strength`, `has_conditions`, `parse_failed`, `conditions_json` 추가

### Stage 4 — Neo4j KG Construction
- **스키마**
  - 노드: `(:Concept {id, name, is_cui})` — non-CUI tail은 `<root_source>:<tail_id>`로 namespacing (HCPCS `C1300` 등 충돌 방지)
  - 엣지: `[:RELATES {relation, has_conditions, conditions_json, condition_logic, guideline_id, evidence_level, evidence_texts, recommendation_strength, ...}]`
  - dedup key (MERGE): `(head, tail, relation, guideline_id)`
- **적재**: `UNWIND $rows ... MERGE` 배치 (500 rows/batch). 제약/인덱스 자동 생성. `parse_failed=True` 또는 head/tail id 누락 triple은 skip
- 출력: [output/stage4_neo4j_summary.json](output/stage4_neo4j_summary.json) — ingest 카운트 + `nodes`, `edges`, `edges_with_conditions` 등 통계

---

## 3. 실행 방법

```bash
# 단일 stage
python pipeline.py stage0 --xml-dir ./crest/xml --primary-dir ./crest/primary
python pipeline.py stage1 --openai-key sk-...
python pipeline.py stage2 --umls-key ...
python pipeline.py stage3 --max-triples 10
python pipeline.py stage4 --neo4j-password ... [--clear]

# 전체 (기본 0~3, --end-stage 4로 Neo4j까지)
python pipeline.py all --umls-key ... --openai-key ...
python pipeline.py all --end-stage 4 --neo4j-password ...

# 백그라운드
nohup python -u pipeline.py stage3 > ./output/stage3.log 2>&1 &
```

---

## 4. 주요 환경변수 / Config

| 변수 | 기본값 |
| --- | --- |
| `UMLS_API_KEY` / `OPENAI_API_KEY` | — |
| `LLM_MODEL` | `gpt-5.4-mini` |
| `CREST_XML_DIR` / `CREST_PRIMARY_DIR` | `./crest/xml` / `./crest/primary` |
| `OUTPUT_DIR` | `./output` |
| `NEO4J_URI` / `USER` / `PASSWORD` | `bolt://localhost:7687` / `neo4j` / `neo4j` |
| `UMLS_MAX_WORKERS` / `LLM_MAX_WORKERS` | 8 / 8 |
| `STAGE3_LLM_CHUNK_SIZE` / `NEO4J_BATCH_SIZE` | 12 / 500 |

---

## 5. RunPod에서 Neo4j KG 구축 — Step 0~10 핵심 명령어

```bash
# Step 0 — SSH 접속
ssh root@<runpod-ip> -p <ssh-port> -i ~/.ssh/id_ed25519

# Step 1 — 시스템 / Python
apt-get update && apt-get install -y git curl python3-pip python3-venv

# Step 2 — 저장소 클론
cd /workspace && git clone https://github.com/<your-org>/M-PAGER.git && cd M-PAGER

# Step 3 — venv + 의존성
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt

# Step 4 — API 키 + CREST 코퍼스 배치
export UMLS_API_KEY="..."  OPENAI_API_KEY="sk-..."
mkdir -p crest/xml crest/primary
# (scp 등으로 코퍼스 업로드)

# Step 5 — Neo4j Docker 기동
docker run -d --name neo4j-mpager -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/<password> -e NEO4J_PLUGINS='["apoc"]' \
  -v /workspace/neo4j/data:/data neo4j:5.14
export NEO4J_URI="bolt://localhost:7687" NEO4J_USER="neo4j" NEO4J_PASSWORD="<password>"

# Step 6 — Stage 0: CREST 파싱
python pipeline.py stage0 --xml-dir ./crest/xml --primary-dir ./crest/primary

# Step 7 — Stage 1: Entity 추출
python pipeline.py stage1 --max-workers 8

# Step 8 — Stage 2: UMLS 매칭 + 1-hop subgraph
python pipeline.py stage2 --max-workers 8

# Step 9 — Stage 3: Condition 추출 (백그라운드)
nohup python -u pipeline.py stage3 --batch-size 12 --max-workers 8 \
  > ./output/stage3.log 2>&1 &

# Step 10 — Stage 4: Neo4j 적재 + 검증
python pipeline.py stage4 --neo4j-password "$NEO4J_PASSWORD" --clear
docker exec -it neo4j-mpager cypher-shell -u neo4j -p "$NEO4J_PASSWORD" \
  "MATCH (n:Concept) RETURN count(n);
   MATCH ()-[r:RELATES]->() WHERE r.has_conditions RETURN count(r);"
```

> **한 번에 0~4까지**: `python pipeline.py all --end-stage 4 --neo4j-clear`

---