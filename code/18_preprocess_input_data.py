"""
18_preprocess_input_data.py
--------------------
mlproject2 final_data.csv 생성 파이프라인

실행:
    python code/18_preprocess_input_data.py

필요 파일 (기준 폴더: data/):
    processed/grid50_master_features.csv
    raw/표고_5000_서울/N3P_F002.shp
    raw/population/TB_T_RSPOP_ADMI.txt
    raw/merged.txt
    raw/BND_ADM_DONG_PG/BND_ADM_DONG_PG.shp
    raw/seoul_missing_by date.xlsx
    raw/전국치매센터표준데이터.csv
    raw/보건복지부_시군구별 치매현황_20251231.csv
    raw/zoning/shp파일/UPIS_C_UQ111.shp

출력:
    processed/final_data.csv
"""

from __future__ import annotations
import math
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import KDTree, cKDTree

# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────
BASE     = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
RAW_DIR  = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT   = PROCESSED_DIR / "final_data.csv"

MASTER_FEATURES = PROCESSED_DIR / "grid50_master_features.csv"
ELEVATION_SHP   = DATA_DIR / "표고_5000_서울" / "N3P_F002.shp"
POP_TXT         = DATA_DIR / "TB_T_RSPOP_ADMI.txt"
METRICS_TXT     = DATA_DIR / "merged.txt"
DONG_SHP        = DATA_DIR / "BND_ADM_DONG_PG" / "BND_ADM_DONG_PG.shp"
MISSING_XLSX    = DATA_DIR / "seoul_missing_by date.xlsx"
DEMENTIA_CENTER = RAW_DIR / "전국치매센터표준데이터.csv"
DEMENTIA_POP    = RAW_DIR / "보건복지부_시군구별 치매현황_20251231.csv"
UPIS_SHP        = BASE / "UQ111_용도지역(도시지역)_202602" / "shp파일" / "UPIS_C_UQ111.shp"

# ─────────────────────────────────────────────
# 서울 그리드 스펙
# ─────────────────────────────────────────────
SEOUL_BBOX  = {"min_lat": 37.41, "max_lat": 37.72, "min_lon": 126.74, "max_lon": 127.20}
CELL_SIZE_M = 50.0
LAT_STEP    = CELL_SIZE_M / 111_000.0
LON_STEP    = CELL_SIZE_M / (111_320.0 * math.cos(math.radians(37.5)))
N_ROWS      = math.ceil((SEOUL_BBOX["max_lat"] - SEOUL_BBOX["min_lat"]) / LAT_STEP)
N_COLS      = math.ceil((SEOUL_BBOX["max_lon"] - SEOUL_BBOX["min_lon"]) / LON_STEP)

# ─────────────────────────────────────────────
# 서울 구→동 매핑
# ─────────────────────────────────────────────
GU_DONG_DICT = {
    '노원구': ['공릉1동','공릉2동','하계1동','하계2동','월계1동','월계2동','월계3동','월계4동','중계1동','중계2동','중계3동','중계4동','중계본동','상계1동','상계2동','상계3동','상계4동','상계5동','상계6·7동','상계8동','상계9동','상계10동'],
    '도봉구': ['방학1동','방학2동','방학3동','창1동','창2동','창3동','창4동','창5동','쌍문1동','쌍문2동','쌍문3동','쌍문4동','방학동','도봉1동','도봉2동'],
    '강북구': ['번1동','번2동','번3동','번동','미아동','송중동','송천동','삼각산동','인수동','수유1동','수유2동','수유3동','우이동'],
    '성북구': ['성북동','삼선동','동선동','돈암1동','돈암2동','안암동','보문동','정릉1동','정릉2동','정릉3동','정릉4동','길음1동','길음2동','종암동','월곡1동','월곡2동','장위1동','장위2동','장위3동','석관동'],
    '성동구': ['왕십리2동','왕십리도선동','마장동','사근동','행당1동','행당2동','응봉동','금호1가동','금호2·3가동','금호4가동','옥수동','성수1가1동','성수1가2동','성수2가1동','성수2가3동','송정동','용답동'],
    '광진구': ['중곡1동','중곡2동','중곡3동','중곡4동','능동','구의1동','구의2동','구의3동','광장동','자양1동','자양2동','자양3동','자양4동','화양동','군자동'],
    '중랑구': ['면목본동','면목2동','면목3·8동','면목4동','면목5동','면목7동','상봉1동','상봉2동','중화1동','중화2동','묵1동','묵2동','망우3동','망우본동','신내1동','신내2동'],
    '동대문구': ['회기동','청량리동','제기동','용신동','전농1동','전농2동','답십리1동','답십리2동','장안1동','장안2동','이문1동','이문2동','휘경1동','휘경2동'],
    '종로구': ['청운효자동','사직동','삼청동','부암동','평창동','무악동','교남동','가회동','종로1·2·3·4가동','종로5·6가동','이화동','혜화동','창신1동','창신2동','창신3동','숭인1동','숭인2동'],
    '중구': ['소공동','회현동','명동','필동','장충동','광희동','을지로동','신당동','다산동','약수동','청구동','동화동','황학동','중림동'],
    '용산구': ['후암동','용산2가동','남영동','청파동','원효로1동','원효로2동','효창동','용문동','한강로동','이촌1동','이촌2동','이태원1동','이태원2동','한남동','서빙고동','보광동'],
    '마포구': ['아현동','공덕동','도화동','용강동','대흥동','염리동','노고산동','신수동','현석동','구수동','창전동','상암동','합정동','망원1동','망원2동','연남동','성산1동','성산2동','서교동'],
    '서대문구': ['천연동','북아현동','충현동','신촌동','연희동','홍제1동','홍제2동','홍제3동','홍은1동','홍은2동','남가좌1동','남가좌2동','북가좌1동','북가좌2동'],
    '은평구': ['녹번동','불광1동','불광2동','갈현1동','갈현2동','구산동','대조동','응암1동','응암2동','응암3동','역촌동','신사1동','신사2동','증산동','수색동','진관동'],
    '강서구': ['염창동','등촌1동','등촌2동','등촌3동','화곡1동','화곡2동','화곡3동','화곡4동','화곡6동','화곡8동','화곡본동','가양1동','가양2동','가양3동','발산1동','우장산동','공항동','방화1동','방화2동','방화3동'],
    '양천구': ['신정1동','신정2동','신정3동','신정4동','신정6동','신정7동','목1동','목2동','목3동','목4동','목5동'],
    '구로구': ['신도림동','구로1동','구로2동','구로3동','구로4동','구로5동','가리봉동','고척1동','고척2동','개봉1동','개봉2동','개봉3동','오류1동','오류2동','수궁동','항동'],
    '금천구': ['시흥1동','시흥2동','시흥3동','시흥4동','시흥5동','가산동','독산1동','독산2동','독산3동','독산4동'],
    '영등포구': ['여의동','당산1동','당산2동','도림동','문래동','양평1동','양평2동','신길1동','신길3동','신길4동','신길5동','신길6동','신길7동','대림1동','대림2동','대림3동'],
    '동작구': ['노량진1동','노량진2동','상도1동','상도2동','상도3동','상도4동','대방동','신대방1동','신대방2동','흑석동','사당1동','사당2동','사당3동','사당4동','사당5동'],
    '관악구': ['보라매동','은천동','성현동','중앙동','청림동','행운동','낙성대동','청룡동','인헌동','남현동','서원동','신원동','서림동','난곡동','난향동','신림동','조원동','대학동','삼성동','신사동','미성동','강현동','봉천동'],
    '서초구': ['방배본동','방배1동','방배2동','방배3동','방배4동','양재1동','양재2동','내곡동','염곡동','신원동','잠원동','반포본동','반포1동','반포2동','반포3동','반포4동','서초1동','서초2동','서초3동','서초4동'],
    '강남구': ['신사동','압구정동','논현1동','논현2동','청담동','삼성1동','삼성2동','대치1동','대치2동','대치3동','대치4동','역삼1동','역삼2동','도곡1동','도곡2동','개포1동','개포2동','개포4동','일원본동','일원1동','일원2동','수서동','세곡동'],
    '송파구': ['풍납1동','풍납2동','거여1동','거여2동','마천1동','마천2동','방이1동','방이2동','오륜동','오금동','송파1동','송파2동','석촌동','삼전동','가락본동','가락1동','가락2동','문정1동','문정2동','장지동','위례동','잠실본동','잠실2동','잠실3동','잠실4동','잠실6동','잠실7동'],
    '강동구': ['강일동','상일동','명일1동','명일2동','고덕1동','고덕2동','암사1동','암사2동','암사3동','천호1동','천호2동','천호3동','성내1동','성내2동','성내3동','둔촌1동','둔촌2동','길동'],
}
DONG_TO_GU    = {dong: gu for gu, dongs in GU_DONG_DICT.items() for dong in dongs}
ALL_SEOUL_DONGS = list(DONG_TO_GU.keys())

# ─────────────────────────────────────────────
# 용도지역 가중치
# ─────────────────────────────────────────────
ZONE_WEIGHTS = {
    'SALES':          {'상업지역': 3.0, '준주거지역': 1.5, '제3종일반주거': 1.0, '제2종일반주거': 0.8, '제1종일반주거': 0.2},
    'POPULATION':     {'제3종일반주거': 2.5, '제2종일반주거': 2.0, '제2종일반주거_저층': 1.5, '준주거지역': 1.5, '제1종일반주거': 1.0, '상업지역': 0.5},
    'INFRASTRUCTURE': {'상업지역': 2.0, '준주거지역': 1.5, '제3종일반주거': 1.2, '제2종일반주거': 0.5, '제1종일반주거': 0.2},
    'STORE':          {'상업지역': 3.5, '준주거지역': 1.5, '제3종일반주거': 1.0, '제2종일반주거': 0.5, '제1종일반주거': 0.2},
    'DEPOSIT':        {'상업지역': 2.5, '제3종일반주거': 2.0, '준주거지역': 1.8, '제2종일반주거': 1.2, '제1종일반주거': 0.8},
}
TARGET_COLS = ['SALES', 'INFRASTRUCTURE', 'STORE', 'POPULATION', 'DEPOSIT']


# ═══════════════════════════════════════════════
# 헬퍼
# ═══════════════════════════════════════════════

def read_csv_auto(path: Path, **kwargs) -> pd.DataFrame:
    for enc in ('utf-8-sig', 'utf-8', 'cp949', 'euc-kr'):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, **kwargs)


def standardize_zone(text) -> str:
    text = "" if pd.isna(text) else str(text)
    if '상업' in text: return '상업지역'
    if '준주거' in text: return '준주거지역'
    if '3종' in text: return '제3종일반주거'
    if '2종' in text: return '제2종일반주거_저층' if '7층' in text else '제2종일반주거'
    if '1종' in text: return '제1종일반주거'
    if '공업' in text: return '공업지역'
    if '녹지' in text: return '녹지지역'
    return '기타'


# ═══════════════════════════════════════════════
# 파이프라인 단계
# ═══════════════════════════════════════════════

def step1_generate_grid() -> pd.DataFrame:
    """서울 50m 그리드 생성"""
    print("  [1/8] 서울 50m 그리드 생성 중...")
    rows = []
    for r in range(N_ROWS):
        for c in range(N_COLS):
            max_lat = SEOUL_BBOX["max_lat"] - r * LAT_STEP
            min_lat = max_lat - LAT_STEP
            min_lon = SEOUL_BBOX["min_lon"] + c * LON_STEP
            max_lon = min_lon + LON_STEP
            rows.append({
                "grid_id_50m": f"G50-{r:05d}-{c:05d}",
                "center_lon":  (min_lon + max_lon) / 2,
                "center_lat":  (min_lat + max_lat) / 2,
                "row_idx": r,
                "col_idx": c,
            })
    df = pd.DataFrame(rows)
    print(f"     완료: {len(df):,}개 그리드")
    return df


def step2_merge_master_features(grid_df: pd.DataFrame) -> pd.DataFrame:
    """grid50_master_features 병합"""
    print("  [2/8] grid50_master_features 병합 중...")
    features = pd.read_csv(MASTER_FEATURES, low_memory=False)
    df = pd.merge(grid_df, features, on='grid_id_50m', how='left').fillna(0)
    print(f"     완료: {df.shape}")
    return df


def step3_add_elevation_slope(df: pd.DataFrame) -> pd.DataFrame:
    """표고 SHP → KDTree로 각 그리드에 elevation/slope 매핑 (Untitled.ipynb 방식)"""
    print("  [3/8] 표고/경사 매핑 중...")
    spot_gdf = gpd.read_file(ELEVATION_SHP).to_crs(epsg=4326)
    spot_coords = np.array(list(zip(spot_gdf.geometry.x, spot_gdf.geometry.y)))
    tree = KDTree(spot_coords)

    grid_coords = df[['center_lon', 'center_lat']].values
    _, idx = tree.query(grid_coords, k=1)

    df['elevation'] = spot_gdf.iloc[idx]['HEIGHT'].values
    df['slope'] = np.abs(np.gradient(df['elevation'].values))
    print(f"     완료: elevation/slope 추가")
    return df


def step4_add_metrics_and_population(df: pd.DataFrame) -> pd.DataFrame:
    """상권지표 + 인구 + 행정동 shp 조인 → 서울 필터 + GU_NM 추가"""
    print("  [4/8] 상권지표 + 인구 + 행정동 조인 중...")

    # 상권지표
    metrics = pd.read_csv(METRICS_TXT, sep='|', quotechar='`', encoding='utf-16', low_memory=False)
    metrics = metrics[metrics['ADSTRD_CD'] != 'ADSTRD_CD'].copy()
    for col in TARGET_COLS:
        metrics[col] = pd.to_numeric(metrics[col], errors='coerce').astype('float32')
    dong_metrics = metrics.groupby('ADSTRD_CD').agg({
        'SALES': 'median', 'INFRASTRUCTURE': 'mean',
        'STORE': 'sum', 'POPULATION': 'median', 'DEPOSIT': 'mean'
    }).reset_index()

    # 동 이름 매핑
    pop_raw = pd.read_csv(POP_TXT, sep='|', quotechar='`', low_memory=False)
    pop_raw['ADMI_CD'] = pop_raw['ADMI_CD'].astype(str)
    pop_map = (pop_raw[['ADMI_CD', 'ADMI_NM']].drop_duplicates()
               .rename(columns={'ADMI_CD': 'ADSTRD_CD', 'ADMI_NM': 'ADM_NM'}))
    dong_metrics = dong_metrics.merge(pop_map, on='ADSTRD_CD', how='left').fillna(0)

    # 행정동 shp로 그리드 → ADM_NM 조인
    seoul_dong = gpd.read_file(DONG_SHP).to_crs(epsg=4326)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df['center_lon'], df['center_lat']),
        crs="EPSG:4326"
    )
    joined = gpd.sjoin(gdf, seoul_dong[['ADM_NM', 'geometry']], how='left', predicate='within')
    joined = joined[~joined.index.duplicated(keep='first')]
    df = pd.DataFrame(joined.drop(columns=['geometry', 'index_right'], errors='ignore'))

    # 상권지표 병합
    df = df.merge(dong_metrics, on='ADM_NM', how='left')

    # 서울 필터 + GU_NM
    df = df[df['ADM_NM'].isin(ALL_SEOUL_DONGS)].copy().reset_index(drop=True)
    df['GU_NM'] = df['ADM_NM'].map(DONG_TO_GU)

    print(f"     완료: {df.shape}")
    return df


def step5_add_missing_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """실종건수 → 그리드별 분배"""
    print("  [5/8] 실종 분포 계산 중...")

    missing_raw = pd.read_excel(MISSING_XLSX, header=None)
    missing_sum = missing_raw.iloc[2:27].copy()
    missing_sum.columns = ['지역', '실종건수']
    missing_sum['실종건수'] = pd.to_numeric(missing_sum['실종건수'], errors='coerce').fillna(0).astype(int)

    df = df.fillna(0)
    df = df.merge(missing_sum, left_on='GU_NM', right_on='지역', how='left')

    df['weight']        = df['pop_65plus_per_grid'] / (df['INFRASTRUCTURE'] + 1)
    df['gu_weight_sum'] = df.groupby('GU_NM')['weight'].transform('sum')
    df['missing_estimated'] = (
        (df['weight'] / df['gu_weight_sum'].replace(0, np.nan)) * df['실종건수']
    ).fillna(0)

    print(f"     완료: {df.shape}")
    return df


def step6_add_dementia_center(df: pd.DataFrame) -> pd.DataFrame:
    """서울 치매센터 → 가장 가까운 그리드에 매핑"""
    print("  [6/8] 치매센터 매핑 중...")

    centers = read_csv_auto(DEMENTIA_CENTER)
    addr_col = [c for c in centers.columns if '주소' in c][0]
    centers['시도'] = centers[addr_col].astype(str).str[:2]
    centers = centers[centers['시도'] == '서울'].dropna(subset=['위도', '경도'])

    tree = cKDTree(df[['center_lat', 'center_lon']].values)
    _, idx = tree.query(centers[['위도', '경도']].values)
    counts = pd.Series(idx).value_counts()
    df['치매센터수'] = df.index.map(counts).fillna(0).astype(int)

    print(f"     완료: 서울 치매센터 {len(centers)}개 매핑")
    return df


def step7_add_upis_zoning(df: pd.DataFrame) -> pd.DataFrame:
    """UPIS 용도지역 조인 + 가중치 적용"""
    print("  [7/8] UPIS 용도지역 조인 + 가중치 적용 중...")

    upis = gpd.read_file(UPIS_SHP)
    upis['표준용도'] = upis['DGM_NM'].apply(standardize_zone)

    df['geometry'] = df.apply(lambda r: Point(r['center_lon'], r['center_lat']), axis=1)
    points = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326").to_crs(upis.crs)

    joined = gpd.sjoin(points, upis[['표준용도', 'geometry']], how='left', predicate='within')
    grouped = joined.groupby(joined.index)['표준용도'].apply(
        lambda x: [v for v in x if pd.notna(v)]
    )
    zone_cols_df = grouped.apply(pd.Series).rename(columns=lambda x: f'표준용도{x+1}')
    result = points.join(zone_cols_df)
    zone_cols = [c for c in result.columns if c.startswith('표준용도')]

    for col in TARGET_COLS:
        wmap = ZONE_WEIGHTS.get(col, {})
        temp = pd.DataFrame(index=result.index)
        for z_col in zone_cols:
            temp[z_col] = result[z_col].map(wmap).fillna(0.0)
        result[f'{col}_w'] = temp.mean(axis=1)
        weight_sum = result.groupby('ADM_NM')[f'{col}_w'].transform('sum')
        result[col] = np.where(weight_sum > 0,
                               result[col] * (result[f'{col}_w'] / weight_sum), 0)

    drop_cols = [f'{col}_w' for col in TARGET_COLS] + ['geometry']
    if 'index_right' in result.columns:
        drop_cols.append('index_right')
    df = pd.DataFrame(result.drop(columns=drop_cols, errors='ignore'))

    print(f"     완료: {df.shape}")
    return df


def step8_add_dementia_population(df: pd.DataFrame) -> pd.DataFrame:
    """보건복지부 치매 현황 → 그리드별 추정 치매환자 분배"""
    print("  [8/8] 치매 인구 분배 중...")

    dementia = read_csv_auto(DEMENTIA_POP)
    cols = dementia.columns.tolist()
    gu_col  = [c for c in cols if '시군구' in c or '지역' in c][0]
    sex_col = [c for c in cols if '성별' in c][0]
    cnt_col = [c for c in cols if '추정치매환자수' in c or '합계' in c][0]

    male   = (dementia[dementia[sex_col] == '남'][[gu_col, cnt_col]]
              .rename(columns={gu_col: '지역', cnt_col: '남성추정치매환자수'}))
    female = (dementia[dementia[sex_col] == '여'][[gu_col, cnt_col]]
              .rename(columns={gu_col: '지역', cnt_col: '여성추정치매환자수'}))

    df = df.merge(male,   left_on='GU_NM', right_on='지역', how='left').drop(columns=['지역'], errors='ignore')
    df = df.merge(female, left_on='GU_NM', right_on='지역', how='left').drop(columns=['지역'], errors='ignore')

    active = df.groupby('GU_NM')['pop_65plus_per_grid'].transform(lambda x: (x > 0).sum())
    for col in ['남성추정치매환자수', '여성추정치매환자수']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df[col] = np.where(df['pop_65plus_per_grid'] > 0,
                           df[col] / active.replace(0, np.nan), 0)
        df[col] = df[col].fillna(0)

    df['추정치매환자수_총합'] = df['남성추정치매환자수'] + df['여성추정치매환자수']

    print(f"     완료: {df.shape}")
    return df


# ═══════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════

def main():
    print("=" * 55)
    print("  final_data.csv 빌드 파이프라인 시작")
    print("=" * 55)

    df = step1_generate_grid()
    df = step2_merge_master_features(df)
    df = step3_add_elevation_slope(df)
    df = step4_add_metrics_and_population(df)
    df = step5_add_missing_distribution(df)
    df = step6_add_dementia_center(df)
    df = step7_add_upis_zoning(df)
    df = step8_add_dementia_population(df)

    # 중간 계산 컬럼 정리
    drop_final = ['weight', 'gu_weight_sum', '실종건수', '지역']
    df = df.drop(columns=[c for c in drop_final if c in df.columns])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False, encoding='utf-8-sig')

    print("=" * 55)
    print(f"  저장 완료 : {OUTPUT}")
    print(f"  Shape    : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print("=" * 55)


if __name__ == "__main__":
    main()