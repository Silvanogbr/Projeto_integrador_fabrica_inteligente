
import cv2

# Carregar a imagem da peça industrial
imagem = cv2.imread("peca.jpg")

imagem_defeito = cv2.imread("peca_defeito.jpg")

if imagem_defeito is None:
    print("Erro ao carregar a imagem da peça com defeito.")
    raise SystemExit

if imagem is None:
    print("Erro: não foi possível carregar a imagem.")
    print("Confira se o arquivo peca.jpg está na pasta do projeto.")
    raise SystemExit

# Converter a imagem para escala de cinza
cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Aplicar desfoque para reduzir ruídos
blur = cv2.GaussianBlur(cinza, (7, 7), 0)

# Detectar bordas com Canny
bordas = cv2.Canny(blur, 70, 180)

# Identificar os contornos
contornos, _ = cv2.findContours(
    bordas,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# Desenhar os contornos encontrados
resultado = imagem.copy()
cv2.drawContours(resultado, contornos, -1, (0, 255, 0), 2)

# Exibir a quantidade de contornos
print("Quantidade de contornos encontrados:", len(contornos))

# Salvar as imagens processadas
cv2.imwrite("imagem_cinza.png", cinza)
cv2.imwrite("imagem_bordas.png", bordas)
cv2.imwrite("peca_contornos.png", resultado)

print("Processamento concluído!")
print("Imagens salvas na pasta do projeto.")
print("\n=========================================\n")

# Processar a imagem da peça com defeito
cinza_defeito = cv2.cvtColor(imagem_defeito, cv2.COLOR_BGR2GRAY)

blur_defeito = cv2.GaussianBlur(cinza_defeito, (7, 7), 0)

bordas_defeito = cv2.Canny(blur_defeito, 70, 180)

contornos_defeito, _ = cv2.findContours(
    bordas_defeito,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

resultado_defeito = imagem_defeito.copy()

cv2.drawContours(
    resultado_defeito,
    contornos_defeito,
    -1,
    (0, 0, 255),
    2
)

cv2.imwrite("imagem_defeito_bordas.png", bordas_defeito)
cv2.imwrite("peca_defeito_contornos.png", resultado_defeito)

print("Contornos encontrados na peça com defeito:", len(contornos_defeito))
print("Imagens da peça com defeito salvas!")

# Comparação inicial das bordas detectadas
total_bordas_boa = cv2.countNonZero(bordas)
total_bordas_defeito = cv2.countNonZero(bordas_defeito)

print("\n--- Comparação das imagens ---")
print("Bordas na peça boa:", total_bordas_boa)
print("Bordas na peça com defeito:", total_bordas_defeito)

diferenca = total_bordas_defeito - total_bordas_boa
print("Diferença na quantidade de pixels de borda:", diferenca)

percentual = (diferenca / total_bordas_boa) * 100

print(f"Aumento percentual de bordas: {percentual:.2f}%")


# Classificação inicial da peça
limite_anomalia = 30.0

print("\n--- Análise da peça ---\n")

if percentual >= limite_anomalia:
    print("Resultado: POSSÍVEL ANOMALIA DETECTADA")
    print("A peça apresentou um aumento significativo de bordas.\n")
else:
    print("Resultado: SEM ANOMALIA IDENTIFICADA PELO CRITÉRIO")
    print("A quantidade de bordas ficou abaixo do limite definido.\n")
