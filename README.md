# ped4you-cross-detection
cross detection for virtual ballot

<pre>
pip install -r requirements.txt
</pre>

inference with inference.py

## How to Deploy
1. `fly launch` to create a new app
2. `fly deploy` to deploy the app
3. `fly scale memory 512` to scale up the memory since the default is 256MB will crash the app
