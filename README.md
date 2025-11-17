# VPI
**A project to predict video's view using early data**

## Turning on the Server using ngrok
**Server**  
python terminal  
conda activate vpi-back  
uvicorn app:app --host 0.0.0.0 --port 5001 --reload  
  
**Ngrok**  
ngrok http 5001  

## Turning on the Server on local server
conda activate vpi-back  
uvicorn app:app --host 127.0.0.1 --port 5001    
