
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Carregar o modelo treinado
modelo = joblib.load("modelo_random_forest.pkl")


@app.route("/")
def inicio():
    return jsonify({
        "projeto": "MetalTech - Fábrica Inteligente",
        "status": "API funcionando!"
    })


@app.route("/predict", methods=["POST"])
def prever():
    dados = request.get_json(silent=True)

    if not isinstance(dados, dict):
        return jsonify({
            "erro": "Envie os dados no formato JSON."
        }), 400

    sensores = [
        "temperatura_c",
        "vibracao_mm_s",
        "pressao_bar"
    ]

    if not all(sensor in dados for sensor in sensores):
        return jsonify({
            "erro": "Informe temperatura_c, vibracao_mm_s e pressao_bar."
        }), 400

    try:
        entrada = pd.DataFrame(
            [[float(dados[sensor]) for sensor in sensores]],
            columns=sensores
        )
    except (TypeError, ValueError):
        return jsonify({
            "erro": "Os valores dos sensores devem ser numéricos."
        }), 400

    if entrada.isna().any().any():
        return jsonify({
            "erro": "Os valores não podem estar vazios ou ser inválidos."
        }), 400

    previsao = int(modelo.predict(entrada)[0])

    return jsonify({
        "falha_prevista": previsao,
        "resultado": (
            "Falha detectada"
            if previsao == 1
            else "Nenhuma falha prevista"
        )
    })


if __name__ == "__main__":
    app.run(debug=True)
