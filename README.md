**server**
python terminal
conda activate vpi-back
uvicorn app:app --host 0.0.0.0 --port 5001 --reload

**ngrok**
ngrok http 5001


local
uvicorn app:app --host 127.0.0.1 --port 5001
