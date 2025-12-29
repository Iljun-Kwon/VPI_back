import pandas as pd
import re
from dateutil import parser

def parse_published_at(series: pd.Series) -> pd.Series:
    """
    object, str, int, datetime 혼합 시리즈를 모두 datetime64[ns]로 바꿔줍니다.
    끝에 붙은 Z나 ±hh:mm 시간대 표시를 제거하고,
    유닉스 타임스탬프(초)도 파싱, 실패 시 NaT 반환.
    """
    # 1) 시리즈 복사 & 문자열화
    s = series.copy()
    # 날짜/시간 객체는 그대로 두고, 나머지는 str로 변환
    # (Timestamp → str → 다시 parse되는 비용을 줄이려면 아래 조건문 생략 가능)
    s_str = s.astype(str).str.strip()

    # 2) 보이지 않는 제어문자 제거
    s_clean = s_str.apply(lambda x: re.sub(r"[^\x20-\x7E]", "", x))

    # 3) 끝에 붙은 Z 또는 ±HH:MM 제거
    s_clean = s_clean.str.replace(r"(Z|[+-]\d{2}:\d{2})$", "", regex=True)

    # 4) None/nan/NaT → 실제 결측값
    s_clean = s_clean.replace({"None": None, "nan": None, "NaT": None})

    # 5) 숫자(정수/소수) 형태면 유닉스 초로 파싱 시도
    is_num = s_clean.str.match(r"^\d+(\.\d+)?$", na=False)
    dt_numeric = pd.to_datetime(s_clean[is_num].astype(float), unit="s", errors="coerce")

    # 6) 나머지는 일반 문자열 파싱 (dateutil 기반)
    def _parse(x):
        try:
            return parser.parse(x)
        except Exception:
            return pd.NaT

    dt_strings = s_clean[~is_num].apply(_parse)

    # 7) 합치기 & 모두 datetime64[ns]로
    dt = pd.Series(index=s_clean.index, dtype="datetime64[ns]")
    dt.loc[is_num] = dt_numeric.values
    dt.loc[~is_num] = dt_strings.values

    return dt