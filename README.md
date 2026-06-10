# 서울시 데이터 분석 경진대회 — 치매 노인 실종자 발견 확률 모델

서울시 50m 격자 단위 **치매 노인 실종 시 발견 확률** 모델. 13개 feature 가중합 + 보수적 임계치 기반 사건별 우선 수색 격자 추천.


## 📂 프로젝트 구조

```
seoul-startup-competition/
├─ code/                              # 처리 파이프라인 (실행 순서대로 번호)
│  ├─ 03_unify_elderly_facilities.py     # 노인복지시설 4개 파일 통합
│  ├─ 04_geocode_*.py                    # 카카오 지오코딩
│  ├─ 05_coord_to_grid.py                # (구) 1km 격자 — 보조용
│  ├─ 06_aggregate_*.py                  # (구) 1km 밀도
│  ├─ 07_unify_cctv_points.py            # CCTV 30개 파일 → 9,535개 점
│  ├─ 08_build_50m_grid.py               # 50m 격자 시스템
│  ├─ 09_cctv_coverage_features.py       # 시설별 반경 CCTV 커버리지
│  ├─ 10_extract_road_network.py         # OSM 도로망 → 교차로/복잡도
│  ├─ 11_zoning_to_grid.py               # 용도지역 → 격자 분류
│  ├─ 12_unify_rest_areas.py             # 공원/쉼터 통합
│  ├─ 13_geocode_senior_centers.py       # 경로당 좌표 부착
│  ├─ 14_rest_areas_grid_features.py     # 휴식지 격자 feature
│  ├─ 15_cctv_facility_grid_features.py  # CCTV·시설 격자 feature
│  ├─ 16_population_to_grid.py           # 인구 → 격자 broadcast
│  ├─ 17_discovery_probability_model.py  # ⭐ 7-feature 종합 모델
│  ├─ 18_preprocess_input_data.py        # ML 모델용 final_data.csv 생성 파이프라인
│
├─ data/
│  ├─ raw/             # ❌ gitignore (각자 다운로드 — 아래 안내 참고)
│  ├─ interim/         # 부분 commit (큰 파일은 재생성)
│  └─ processed/       # ✅ 모델 산출물 (commit)
│     ├─ grid50_master_features.csv         (32 cols × 121,051 grids)
│     ├─ grid50_discovery_probability.csv   ⭐ 모델 출력
│     ├─ final_data.csv                     ML 파이프라인 최종 입력 데이터 (18_PROPROCESS_INPUT_DATA 실행시 생성)
|
├─ UQ111_용도지역(도시지역)_202602/            # 서울시 용도지역 관련 데이터셋 (.shp 데이터셋 입력)
│
└─ main.ipynb               # 메인 실행 코드
```

## 🚀 환경 셋업

```bash
# Python 3.10+ 권장
cd dimentia_wandering
python3 -m venv .venv
source .venv/bin/activate
pip install pandas openpyxl numpy geopandas shapely pyproj osmnx networkx folium scipy
```

## 🔑 카카오 API 키 (지오코딩에 필요)

경로당 좌표 부착 등에 카카오 Local API 사용. 본인 키로 환경변수 설정:

```bash
export KAKAO_REST_API_KEY="여기에_본인_키"
```

키 발급: https://developers.kakao.com/console → 앱 추가 → REST API 키 복사

## 📥 원본 데이터 다운로드 안내

`data/raw/` 폴더는 gitignore 처리되어 있습니다. 프로젝트 내 파이프라인(`18_preprocess_input_data.py` 등) 실행에 필요한 아래 데이터들을 폴더 구조에 맞춰 저장하세요:

```
data/raw/
├─ elderly_facilities/      # 노인복지시설 4개 (서울 열린데이터광장)
│  ├─ 01_노인주거복지시설.csv
│  ├─ 02_노인의료복지시설.csv
│  ├─ 03_노인의료복지시설현황.xlsx
│  └─ 04_노인여가복지시설.csv
├─ cctv/                    # CCTV 위치 데이터 (구별 + 시 전체)
│  └─ 서울시 불법주정차_전용차로 위반 단속 CCTV 위치정보.csv  (마스터, 4,652점)
│  └─ + 강북·금천·은평·영등포·관악 일반 CCTV
├─ zoning/shp파일/          # 용도지역 폴리곤 (공공데이터포털 / VWorld)
│  ├─ UPIS_C_UQ111.shp + .dbf + .prj + .shx + .sbn + .sbx + .shp.xml
├─ rest_areas/              # 휴식지 4종
│  ├─ 01_seoul_parks.csv
│  ├─ 02_seoul_senior_centers.csv
│  ├─ 03_seoul_cool_shelter.csv
│  └─ 04_seoul_warm_shelters.csv
└─ population/
   └─ 주민등록인구(내국인+각+세별_구별)*.csv
```

각 데이터 출처는 `docs/` 또는 코드 상단 주석 참고.

## ⏯️ 파이프라인 실행

```bash
source .venv/bin/activate
export KAKAO_REST_API_KEY="..."

# 순서대로 실행 (각 스크립트가 직전 출력에 의존)
python code/03_unify_elderly_facilities.py        # 시설 통합
python code/04_geocode_elderly_facilities_kakao.py  # 시설 좌표
python code/07_unify_cctv_points.py               # CCTV 통합
python code/08_build_50m_grid.py                  # 50m 격자
python code/10_extract_road_network.py            # 도로망 (5~10분, OSM 다운로드)
python code/11_zoning_to_grid.py                  # 용도지역
python code/12_unify_rest_areas.py                # 휴식지
python code/13_geocode_senior_centers.py          # 경로당 좌표
python code/14_rest_areas_grid_features.py
python code/15_cctv_facility_grid_features.py
python code/16_population_to_grid.py
python code/17_discovery_probability_model.py     # ⭐ 모델 종합
python code/18_preprocess_input_data.py           # ML 모델용 final_data.csv 빌드
```
