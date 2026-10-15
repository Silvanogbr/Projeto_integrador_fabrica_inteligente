
import requests

url = "http://127.0.0.1:5000/predict"

dados = {
    "temperatura_c": 85.0,
    "vibracao_mm_s": 12.5,
    "pressao_bar": 95.0
}

resposta = requests.post(url, json=dados)

print("Status:", resposta.status_code)
print("Resposta:", resposta.json())
