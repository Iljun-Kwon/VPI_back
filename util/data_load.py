# data/data_loader.py
import pandas as pd
from config.settings import supabase_client
import time

def fetch_all_rows_paginated(table_name: str) -> pd.DataFrame:
    """
    Supabase 테이블에서 페이지네이션을 통해 모든 행을 가져옵니다.

    Args:
        table_name: 데이터를 가져올 테이블 이름

    Returns:
        Pandas DataFrame
    """
    PAGE_SIZE = 1000
    all_data = []
    current_page = 0

    print(f"[🔄] '{table_name}' 테이블 데이터 로딩 중...")

    while True:
        start_index = current_page * PAGE_SIZE
        end_index = start_index + PAGE_SIZE - 1

        try:
            response = supabase_client.table(table_name).select('*', count='exact').range(start_index, end_index).execute()

            # API 응답에서 데이터와 전체 개수 추출
            data = response.data
            total_count = response.count

            if not data:
                break

            all_data.extend(data)

            # 진행 상황 표시
            print(f"\r   ... {len(all_data)} / {total_count} 행 로드 완료", end="")

            if len(data) < PAGE_SIZE:
                break

            current_page += 1
            time.sleep(0.1)  # API 서버 부하를 줄이기 위한 약간의 딜레이

        except Exception as e:
            print(f"\n[❌] '{table_name}' 테이블 로딩 중 에러 발생: {e}")
            # 부분적으로라도 로드된 데이터가 있으면 반환
            break

    print(f"\n[✅] '{table_name}' 테이블 로딩 완료. 총 {len(all_data)} 행.")
    return pd.DataFrame(all_data)

def fetch_all_rows_keyset(table_name: str, id_col: str = "id", page_size: int = 10000, where=None) -> pd.DataFrame:
    """
    Faster, timeout-resistant keyset pagination:
    SELECT * FROM table WHERE id > last_id ORDER BY id ASC LIMIT page_size
    """
    all_rows = []
    last_id = None
    total = 0
    print(f"[🔄] '{table_name}' (keyset) 로딩 시작... page_size={page_size}")

    while True:
        q = supabase_client.table(table_name).select("*").order(id_col, desc=False).limit(page_size)
        if last_id is not None:
            q = q.gt(id_col, last_id)
        if where:
            # where is a list of callables like: lambda q: q.gte("collected_at","2025-01-01")
            for f in where:
                q = f(q)

        try:
            res = q.execute()
            batch = res.data or []
        except Exception as e:
            print(f"\n[❌] '{table_name}' keyset 페이지 로딩 중 에러: {e}  (재시도 대기)")
            time.sleep(1.0)
            continue  # retry same page

        if not batch:
            break

        all_rows.extend(batch)
        total += len(batch)
        last_id = batch[-1][id_col]
        print(f"\r   ... {total} 행 로드 완료", end="")
        # tiny backoff to be friendly to API
        time.sleep(0.05)

    print(f"\n[✅] '{table_name}' 로딩 완료. 총 {total} 행.")
    return pd.DataFrame(all_rows)

def fetch_all_rows_offset(table_name: str, page_size: int = 5000, include_count: bool = False) -> pd.DataFrame:
    """
    Offset pagination. Avoid include_count=True for big tables to prevent COUNT(*) timeouts.
    """
    all_rows = []
    current_page = 0
    print(f"[🔄] '{table_name}' (offset) 로딩 시작... page_size={page_size}")

    while True:
        start = current_page * page_size
        end = start + page_size - 1
        try:
            # ⚠️ count='exact' removed by default to avoid timeouts
            sel = supabase_client.table(table_name).select('*', count='exact' if include_count else None)
            res = sel.range(start, end).execute()
            batch = res.data or []
            if not batch:
                break
            all_rows.extend(batch)
            current_page += 1

            if include_count and res.count is not None:
                print(f"\r   ... {len(all_rows)} / {res.count} 행 로드 완료", end="")
            else:
                print(f"\r   ... {len(all_rows)} 행 로드 완료", end="")
        except Exception as e:
            print(f"\n[❌] '{table_name}' offset 페이지 로딩 중 에러: {e}  (재시도 대기)")
            time.sleep(1.0)
            continue  # retry same page

        time.sleep(0.05)

    print(f"\n[✅] '{table_name}' 로딩 완료. 총 {len(all_rows)} 행.")
    return pd.DataFrame(all_rows)

def fetch_all_data():
    """
    Supabase에서 videos, video_snapshots, channel_snapshots 테이블의 모든 데이터를 가져와 반환합니다.
    """
    print("\n--- [🚀] Supabase 전체 데이터 로딩 시작 ---")

    #df_videos = fetch_all_rows_paginated('videos')
    #df_video_snapshots = fetch_all_rows_paginated('video_snapshots')
    #df_channel = fetch_all_rows_paginated('channels')
    #df_channel_snapshots = fetch_all_rows_paginated('channel_snapshots')
    df_videos = fetch_all_rows_offset('videos', page_size=5000, include_count=False)
    df_channel = fetch_all_rows_offset('channels', page_size=5000, include_count=False)
    df_channel_snapshots = fetch_all_rows_offset('channel_snapshots', page_size=10000, include_count=False)
    df_video_snapshots = fetch_all_rows_keyset('video_snapshots', id_col="id", page_size=20000)

    if df_videos.empty or df_video_snapshots.empty or df_channel.empty or df_channel_snapshots.empty:
        print("[⚠️] 하나 이상의 테이블에서 데이터를 가져오지 못했습니다.")
        return None, None, None, None

    print("\n--- [✅] Supabase 전체 데이터 로딩 성공 ---")

    # 💾 Save DataFrames as CSV
    df_videos.to_csv("videos.csv", index=False, encoding="utf-8-sig")
    df_video_snapshots.to_csv("video_snapshots.csv", index=False, encoding="utf-8-sig")
    df_channel.to_csv("channels.csv", index=False, encoding="utf-8-sig")
    df_channel_snapshots.to_csv("channel_snapshots.csv", index=False, encoding="utf-8-sig")

    return df_videos, df_video_snapshots, df_channel, df_channel_snapshots


if __name__ == '__main__':
    # 스크립트 직접 실행 시 테스트
    fetch_all_data()