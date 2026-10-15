import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Carregar os dados
df = pd.read_csv("dados_sensores_metaltech.csv")

# 2. Organizar os dados por máquina e horário
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(
    ["id_maquina", "timestamp"]
).reset_index(drop=True)

# 3. Tratar valores inválidos
sensores = [
    "temperatura_c",
    "vibracao_mm_s",
    "pressao_bar"
]

df.loc[df["temperatura_c"] == 999.9, "temperatura_c"] = np.nan
df.loc[df["vibracao_mm_s"] < 0, "vibracao_mm_s"] = np.nan

# 4. Preencher valores ausentes por máquina
for coluna in sensores:
    df[coluna] = df.groupby("id_maquina")[coluna].transform(
        lambda serie: serie.interpolate(limit_direction="both")
    )

# 5. Separar entradas e resultado esperado
df = df.dropna(subset=["falha_motor"])

X = df[sensores]
y = df["falha_motor"].astype(int)

# 6. Dividir em 70% para treino e 30% para teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# 7. Conferir a divisão
print("Preparação concluída!")
print("Dados para treinamento:", X_treino.shape)
print("Dados para teste:", X_teste.shape)

print("\nFalhas no treinamento:")
print(y_treino.value_counts())

print("\nFalhas no teste:")
print(y_teste.value_counts())

# 8. Padronizar os dados para a Regressão Logística
scaler = StandardScaler()

X_treino_pad = scaler.fit_transform(X_treino)
X_teste_pad = scaler.transform(X_teste)

# 9. Criar e treinar os modelos
modelos = {
    "Regressão Logística": LogisticRegression(
        class_weight="balanced",
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        class_weight="balanced",
        random_state=42
    )
}

resultados = []

for nome, modelo in modelos.items():

    if nome == "Regressão Logística":
        modelo.fit(X_treino_pad, y_treino)
        previsoes = modelo.predict(X_teste_pad)
    else:
        modelo.fit(X_treino, y_treino)
        previsoes = modelo.predict(X_teste)

    resultados.append({
        "Modelo": nome,
        "Acurácia": accuracy_score(y_teste, previsoes),
        "Precisão": precision_score(
            y_teste, previsoes, zero_division=0
        ),
        "Recall": recall_score(
            y_teste, previsoes, zero_division=0
        ),
        "F1-score": f1_score(
            y_teste, previsoes, zero_division=0
        )
    })

    print(f"\n{'=' * 40}")
    print(nome)
    
    print("Matriz de confusão:")
    print(confusion_matrix(y_teste, previsoes))

    # Gerar matriz de confusão visual
    ConfusionMatrixDisplay.from_predictions(
        y_teste,
        previsoes,
        display_labels=["Sem falha", "Com falha"],
        cmap="Blues"
    )

    plt.title(f"Matriz de Confusão - {nome}")
    plt.tight_layout()

    nome_arquivo = (
        "matriz_logistica.png"
        if nome == "Regressão Logística"
        else "matriz_random_forest.png"
    )

    plt.savefig(nome_arquivo, dpi=150)
    plt.close()

    print(f"Matriz salva: {nome_arquivo}")


# 10. Comparar os resultados
tabela = pd.DataFrame(resultados)

print("\nCOMPARAÇÃO DOS MODELOS")
print(tabela.round(4).to_string(index=False))

# Salvar o modelo Random Forest
joblib.dump(
    modelos["Random Forest"],
    "modelo_random_forest.pkl"
)

# Salvar o modelo de Regressão Logística
joblib.dump(
    modelos["Regressão Logística"],
    "modelo_regressao_logistica.pkl"
)

# Salvar o scaler usado pela Regressão Logística
joblib.dump(
    scaler,
    "scaler_regressao_logistica.pkl"
)

print("\nModelos e scaler salvos com sucesso!")

# Selecionar somente os sensores
X_cluster = df[sensores]

# Padronizar os dados para o agrupamento
scaler_cluster = StandardScaler()
X_cluster_pad = scaler_cluster.fit_transform(X_cluster)

# Criar dois grupos
kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

grupos = kmeans.fit_predict(X_cluster_pad)

# Avaliar a separação dos grupos
silhueta = silhouette_score(X_cluster_pad, grupos)

print("\n=========================================\n")
print("AGRUPAMENTO K-MEANS")
print("Quantidade de grupos:", len(set(grupos)))
print("Registros em cada grupo:")
print(pd.Series(grupos).value_counts().sort_index())
print(f"Coeficiente de silhueta: {silhueta:.4f}")

# Salvar o modelo e o scaler
joblib.dump(kmeans, "modelo_kmeans.pkl")
joblib.dump(scaler_cluster, "scaler_kmeans.pkl")

print("Modelo K-Means e scaler salvos com sucesso!")
