# config/settings.py
import os
from dotenv import load_dotenv
import supabase

# .env 파일에서 환경 변수 로드
# 이 파일이 실행되는 위치를 기준으로 .env 파일을 찾습니다.
# 보통 프로젝트 루트에 .env를 두므로, 경로를 적절히 조정해야 할 수 있습니다.
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

# Supabase 클라이언트 설정
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase URL과 Key를 .env 파일에 설정해주세요.")

supabase_client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)

# YouTube API 키 설정
# .env 파일에서 'API_KEY_'로 시작하는 모든 키를 가져옵니다.
API_KEYS = [v for k, v in os.environ.items() if k.startswith('API_KEY_')]

# 모델 및 스케일러 저장 경로
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'mlp_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.pkl')