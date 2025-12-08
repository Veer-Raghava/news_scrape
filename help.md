### install requirements:
(ignore hdb) pip install -r requirements.txt
run docker first using:
 docker run -d -p 6333:6333 -v qdrant-storage:/qdrant/storage qdrant/qdrant

 ##  run_everything.py but before that
 make sure you are running:
 1. net stat mongodb or manually open mongo and connect to localhost ( compass )
 2. node server.js (for jwt)
 3. npm run dev (both on frontend and backend) (first backend next frontend) -- we are using 5174 port make sure you open the same port

### run event_cluster_rebuild.py

 ### start py server using:
 uvicorn project.scripts.ml_service:app --reload --port 8000