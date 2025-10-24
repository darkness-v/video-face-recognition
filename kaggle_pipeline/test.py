import requests, json, os
BASE='http://localhost:8000'
SERVICE_TOKEN = "2rHSwGQqoMcyRsJa34OTp5dZTmW_4e99AEtZGe23ZUFrRAdsr"
headers={'Authorization': f'Bearer {SERVICE_TOKEN}'} if SERVICE_TOKEN else {}
resp = requests.post(BASE+'/ping', headers=headers)
print(resp.status_code, resp.text)
