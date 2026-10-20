# Fábrica Inteligente: manutenção preditiva e inspeção visual

Protótipo desenvolvido para o Projeto Integrador da MetalTech Soluções Industriais S.A. O projeto explora o uso de dados de sensores para apoiar a identificação de risco de falha em motores e o processamento de imagens para auxiliar a inspeção de peças. Também disponibiliza uma API REST para consultar o classificador e uma interface visual para analisar imagens.

> **Escopo:** este repositório é um protótipo acadêmico. As previsões e medidas de imagem são indicadores experimentais e não substituem validação de engenharia, inspeção humana, intertravamentos ou sistemas de segurança industrial.

## Sumário

- [Objetivos e contexto](#objetivos-e-contexto)
- [Arquitetura do projeto](#arquitetura-do-projeto)
- [Etapa 1: dados industriais e preparação](#etapa-1-dados-industriais-e-preparação)
- [Etapa 2: modelos preditivos e avaliação](#etapa-2-modelos-preditivos-e-avaliação)
- [Etapa 3: visão computacional e interface](#etapa-3-visão-computacional-e-interface)
- [API REST](#api-rest)
- [Considerações técnicas, éticas e regulatórias](#considerações-técnicas-éticas-e-regulatórias)
- [Plano de monitoramento MLOps por seis meses](#plano-de-monitoramento-mlops-por-seis-meses)
- [Instalação e execução](#instalação-e-execução)
- [Estrutura dos arquivos](#estrutura-dos-arquivos)
- [Limitações e próximos passos](#limitações-e-próximos-passos)

## Objetivos e contexto

A proposta da atividade parte de um cenário fabril com sensores IoT que medem temperatura, vibração e pressão, além de registros de ocorrência de falha (`falha_motor`). O desafio é transformar essas medições em informação útil para manutenção preditiva e usar visão computacional como apoio à inspeção de qualidade.

Manutenção preditiva procura identificar sinais associados a uma falha antes que ela provoque uma parada não planejada. Um modelo estatístico pode priorizar inspeções e ajudar a equipe a planejar intervenções; sua saída é uma estimativa baseada nos dados disponíveis, não uma garantia de que uma falha ocorrerá ou deixará de ocorrer.

O repositório implementa três frentes: preparação e treinamento com o CSV, inspeção de imagens estáticas com OpenCV e consulta do modelo Random Forest por API. A aplicação Streamlit apresenta uma interface para explorar o processamento visual.

## Arquitetura do projeto

| Componente | Arquivos principais | Função |
|---|---|---|
| Preparação e treinamento | `treinamento.py`, `dados_sensores_metaltech.csv` | Limpa valores selecionados, separa treino e teste, treina classificadores e K-Means e salva artefatos. |
| Análise exploratória | `tratamento.ipynb` | Notebook para inspeção e tratamento exploratório do conjunto de dados. |
| Classificação por sensores | `api.py`, `modelo_random_forest.pkl` | Expõe a previsão do Random Forest por HTTP/JSON. |
| Inspeção de imagem por script | `visao_computacional.py`, `peca.jpg`, `peca_defeito.jpg` | Gera imagens em tons de cinza, bordas e contornos e compara contagens de pixels de borda. |
| Interface de análise visual | `app_visual.py` | Permite enviar uma imagem e visualizar bordas, contornos, regiões com concentração de bordas e métricas descritivas. |

As bibliotecas centrais são Python, Pandas, NumPy, scikit-learn, Matplotlib, OpenCV, Flask e Streamlit. Os modelos são serializados em arquivos `.pkl` com Joblib.

## Etapa 1: dados industriais e preparação

### Inspeção e qualidade dos dados

O arquivo contém 10.000 registros e as colunas `timestamp`, `id_maquina`, `temperatura_c`, `vibracao_mm_s`, `pressao_bar` e `falha_motor`. Na amostra incluída, há quatro máquinas e 500 registros positivos de falha (5% do total). As três colunas de sensores têm valores ausentes: 296 em temperatura, 299 em vibração e 300 em pressão. O timestamp e o identificador não apresentam ausências, assim como o rótulo.

Em dados industriais, falhas de comunicação podem produzir valores nulos, enquanto ruído, erro de calibração ou falha do sensor podem gerar leituras fisicamente implausíveis. A interpretação dos limites deve ser definida com conhecimento do equipamento e do processo; um valor raro não é necessariamente inválido.

### Limpeza implementada

No script `treinamento.py`, o timestamp é convertido para data/hora e os dados são ordenados por máquina e horário. O valor de temperatura igual a `999.9` e vibrações negativas são substituídos por ausentes. Para cada sensor, a interpolação linear é feita separadamente por máquina, com preenchimento também nas extremidades (`limit_direction="both"`). Registros sem `falha_motor` são removidos antes de formar os conjuntos de treino e teste.

A interpolação é adequada quando medições próximas representam razoavelmente a evolução temporal do processo. Em produção, deve-se avaliar a duração das lacunas e evitar interpolar longos períodos sem dados. O código não aplica Z-score nem regra geral de remoção de outliers e não gera no script de treinamento os histogramas, boxplots ou matriz de correlação pedidos no enunciado; esses itens podem ser documentados ou complementados no notebook de análise exploratória.

### Análise exploratória

Histogramas ajudam a observar a distribuição e assimetrias; boxplots facilitam a identificação de valores extremos; e uma matriz de correlação resume associações lineares entre variáveis. Correlação não demonstra causalidade e pode ser influenciada por classe desbalanceada, valores extremos, máquina ou período. Uma análise mais robusta deve comparar sensores por classe e por equipamento e validar os achados com especialistas de manutenção.

## Etapa 2: modelos preditivos e avaliação

### Preparação e treinamento

O alvo `y` é `falha_motor`; as variáveis previsoras `X` são temperatura, vibração e pressão. O script divide os registros aleatoriamente em 70% para treino e 30% para teste, com `random_state=42` e estratificação pelo alvo. A estratificação preserva aproximadamente a proporção entre classes em cada subconjunto.

São treinados dois classificadores supervisionados:

- **Regressão Logística:** modelo linear de classificação usado como referência. Neste projeto, as variáveis são padronizadas com `StandardScaler`, que transforma cada medida em uma escala comparável a partir da média e do desvio-padrão do conjunto de treino. O scaler é ajustado somente no treino e depois aplicado ao teste, evitando vazamento dessa etapa.
- **Random Forest:** conjunto de árvores de decisão que combina previsões de várias árvores e pode modelar relações não lineares. O classificador é treinado com as medidas originais, sem o scaler, pois árvores não dependem da escala da mesma forma que modelos lineares baseados em distância ou coeficientes.

Ambos usam `class_weight="balanced"`, que atribui maior peso à classe minoritária durante o ajuste. Isso é relevante porque apenas 5% dos rótulos da amostra indicam falha. O treinamento salva os dois classificadores e o scaler da Regressão Logística.

O K-Means é aplicado separadamente, sem usar os rótulos de falha para formar os grupos. As três variáveis são padronizadas e o algoritmo procura dois agrupamentos por proximidade aos centroides. A silhueta é calculada como indicador de coesão e separação dos grupos. K-Means não sabe que um grupo significa “normal” ou “crítico”; essa interpretação requer comparar os agrupamentos com os rótulos e com o conhecimento operacional. Os grupos também podem refletir diferenças entre máquinas ou condições de operação, e não falhas.

### Métricas e matriz de confusão

O script apresenta matriz de confusão e calcula acurácia, precisão, recall e F1-score para cada classificador no conjunto de teste.

| Medida | Interpretação para manutenção |
|---|---|
| Acurácia | Proporção total de previsões corretas. Pode parecer alta mesmo se o modelo ignorar a classe rara. |
| Precisão | Entre os alarmes de falha emitidos, quantos correspondem a falhas registradas. Baixa precisão pode gerar inspeções desnecessárias. |
| Recall (sensibilidade) | Entre as falhas registradas, quantas foram identificadas. Baixo recall significa mais falhas não sinalizadas. |
| F1-score | Média harmônica de precisão e recall, útil para resumir o equilíbrio entre ambos. |

Em uma aplicação industrial, falsos negativos podem causar parada não planejada ou danos; falsos positivos podem gerar custos de inspeção e manutenção antecipada. A escolha do limiar e da métrica prioritária deve considerar esses custos, a capacidade de resposta da equipe e dados de validação representativos. O código avalia o limiar padrão dos classificadores e não implementa uma otimização específica de recall. As matrizes salvas são `matriz_logistica.png` e `matriz_random_forest.png`; os valores numéricos são exibidos ao executar o treinamento.

### Artefatos gerados

`treinamento.py` salva `modelo_random_forest.pkl`, `modelo_regressao_logistica.pkl`, `scaler_regressao_logistica.pkl`, `modelo_kmeans.pkl` e `scaler_kmeans.pkl`, além das matrizes de confusão. Os arquivos `.pkl` devem ser usados com cautela: carregue apenas artefatos confiáveis e mantenha versões compatíveis de Python e scikit-learn entre treinamento e execução.

## Etapa 3: visão computacional e interface

### Fundamentos do processamento

Uma imagem digital pode ser tratada como uma matriz de pixels. A conversão para tons de cinza reduz a informação de cor e simplifica a análise de intensidade. O desfoque Gaussiano atenua variações pequenas e ruído antes da detecção. O algoritmo Canny identifica mudanças de intensidade que podem corresponder a bordas. A busca de contornos agrupa pixels de borda em formas que podem ser medidas ou comparadas.

O script `visao_computacional.py` aplica conversão para cinza, Gaussian Blur com janela 7×7 e Canny com limiares 70 e 180 nas imagens de exemplo. Ele localiza contornos externos, salva imagens intermediárias e compara o número de pixels de borda entre a peça de referência e a peça com defeito. Há uma regra experimental de variação percentual de 30% para imprimir uma indicação de possível anomalia.

Essa comparação é sensível a iluminação, escala, enquadramento, foco, orientação, fundo e textura. A contagem de bordas, isoladamente, não confirma defeito e o script de exemplo não alinha as imagens nem compara geometricamente a peça a um gabarito validado.

### Interface Streamlit

`app_visual.py` fornece uma interface para enviar imagens JPG, JPEG, PNG ou BMP e ajustar os limiares do Canny. Ela mostra imagem original, tons de cinza, bordas, contornos, regiões de maior concentração local de bordas e indicadores como quantidade de bordas, densidade, maior área de contorno, circularidade e tempo de processamento.

As áreas vermelhas representam concentração de bordas, não defeitos detectados. A interface recomenda revisão humana e não classifica automaticamente a peça como aprovada ou reprovada. A circularidade é uma medida geométrica do maior contorno e também depende do enquadramento e da segmentação.

## API REST

O arquivo `api.py` implementa um serviço Flask. A rota `GET /` informa que a API está ativa. A rota `POST /predict` recebe um objeto JSON com os campos numéricos `temperatura_c`, `vibracao_mm_s` e `pressao_bar`; o serviço valida a estrutura e os valores e retorna a classe prevista pelo Random Forest.

Exemplo de requisição:

```json
{
  "temperatura_c": 85.0,
  "vibracao_mm_s": 3.2,
  "pressao_bar": 41.0
}
```

Exemplo de resposta:

```json
{
  "falha_prevista": 0,
  "resultado": "Nenhuma falha prevista"
}
```

`falha_prevista` é a saída do classificador (0 ou 1), não uma probabilidade. Uma resposta `1` indica a classe positiva aprendida a partir dos rótulos do conjunto. A API não autentica clientes nem implementa transporte seguro, controle de acesso, monitoramento ou integração direta com CLPs; não deve ser exposta em uma rede industrial sem essas medidas e validação operacional.

O exemplo de chamada encontra-se em `testar_api.py`. O serviço deve estar rodando antes da execução desse cliente.

## Considerações técnicas, éticas e regulatórias

### Impacto sociolaboral

Automação pode reduzir tarefas repetitivas e tornar a inspeção mais consistente, mas também altera funções, competências e responsabilidades. A implantação deve envolver inspetores, operadores, manutenção e segurança desde a concepção. O objetivo recomendado é apoiar a decisão e reduzir exposição a tarefas perigosas, com capacitação para análise dos alertas, interpretação de limitações e manutenção dos sensores e câmeras. Mudanças de função e impactos no emprego devem ser avaliados de forma transparente, com diálogo e oportunidades de requalificação.

Decisões de manutenção e qualidade precisam manter responsáveis humanos identificáveis, especialmente quando um alerta é incerto ou contradiz o procedimento de segurança. É necessário registrar como o sistema contribuiu para a decisão e permitir contestação ou revisão de resultados. O protótipo não deve comandar máquinas nem substituir inspeções regulamentares.

### Cibersegurança industrial

Um modelo conectado à rede fabril amplia a superfície de ataque. Leituras falsas ou alteradas podem provocar alarmes indevidos ou ocultar condições de risco; um serviço indisponível pode interromper o fluxo de consulta; e arquivos de modelo adulterados podem executar código ao serem desserializados ou produzir resultados incorretos. A implantação deve seguir segmentação de rede e princípio do menor privilégio, autenticação e autorização, criptografia nas comunicações, validação de entradas, atualização controlada, registro de eventos, cópias de segurança e gestão de versões e integridade dos modelos. A API deste repositório é didática e não inclui essas proteções.

Os sensores e a aplicação devem operar em modo de apoio, com limites seguros e procedimentos locais de contingência. A indisponibilidade ou divergência do modelo não pode impedir proteções determinísticas dos equipamentos nem a parada segura prevista pela engenharia.

### Privacidade e LGPD

Imagens de câmeras podem identificar uma pessoa quando seu rosto aparece. Nesse caso, a imagem ou o dado biométrico usado para identificação pode constituir dado pessoal; dados biométricos vinculados a uma pessoa são dados pessoais sensíveis nos termos da Lei nº 13.709/2018 (LGPD). Antes da implantação, a organização deve definir finalidade e hipótese legal aplicável, necessidade e proporcionalidade da captação, transparência aos trabalhadores, controles de acesso, prazo de retenção, segurança e descarte. A avaliação deve envolver o encarregado e as áreas jurídica, de segurança da informação e de relações de trabalho; pode ser apropriado elaborar relatório de impacto à proteção de dados pessoais.

Sempre que viável, o enquadramento deve limitar-se às peças e evitar a captura de rostos e áreas não necessárias. Também se deve separar a finalidade de controle de qualidade de qualquer monitoramento de desempenho individual. O processamento local, mascaramento de rostos e retenção mínima podem reduzir exposição, mas não dispensam a análise legal e organizacional. Esta seção apresenta orientação geral para o projeto acadêmico e não substitui parecer jurídico.

## Plano de monitoramento MLOps por seis meses

O desempenho de um modelo pode se degradar quando mudam sensores, máquinas, matéria-prima, manutenção ou condições de operação. *Data drift* é uma mudança na distribuição das entradas; *concept drift* ocorre quando a relação entre as entradas e a falha muda. Um plano simples deve acompanhar dados e resultados com supervisão da equipe:

| Período | Ações sugeridas |
|---|---|
| Mês 1 | Registrar versão do modelo, versão do software e origem dos dados; estabelecer linha de base para distribuição dos sensores, dados ausentes, alertas e métricas. Revisar alertas com manutenção. |
| Mês 2 | Verificar semanalmente faixas e distribuições de temperatura, vibração e pressão por máquina; investigar mudanças de calibração e aumento de valores inválidos ou ausentes. |
| Mês 3 | Confrontar previsões com ordens de serviço e falhas confirmadas; calcular matriz de confusão, recall, precisão e F1 assim que os rótulos estiverem disponíveis. |
| Mês 4 | Revisar falsos negativos e positivos, diferenças entre máquinas e condições de operação; confirmar se os dados de avaliação continuam representativos. |
| Mês 5 | Avaliar candidato a retreinamento com dados recentes rotulados. Comparar com o modelo vigente em validação temporal e revisar segurança, desempenho e impacto operacional antes de qualquer troca. |
| Mês 6 | Documentar tendências, incidentes, limitações e decisão de manter ou substituir o modelo. Aprovar plano para o próximo ciclo e preservar possibilidade de retorno à versão anterior. |

Como gatilhos iniciais, a equipe pode definir limites de desvio de entrada, falhas de disponibilidade, aumento de alarmes e queda de recall. Esses limites devem ser calibrados com dados reais e custos de erro; um valor como recall de 90% só deve ser adotado como meta após validação com responsáveis técnicos. Retreinamento não deve ser implantado automaticamente sem avaliação e aprovação operacional. Todo evento relevante deve registrar data, máquina, versão do modelo e resultado da investigação, minimizando dados pessoais.

## Instalação e execução

Recomenda-se Python 3.10 ou versão compatível com os artefatos e bibliotecas usados para treiná-los. Em um ambiente virtual, instale as dependências:

```bash
python -m venv .venv
```

No Windows:

```powershell
.venv\Scripts\Activate.ps1
```

No Linux/macOS:

```bash
source .venv/bin/activate
```

Instalação das bibliotecas utilizadas:

```bash
pip install pandas numpy scikit-learn matplotlib opencv-python flask streamlit pillow requests jupyter
```

Execute os componentes a partir da pasta do projeto:

```bash
# Treina novamente os classificadores e o K-Means e atualiza os artefatos
python treinamento.py

# Executa o processamento OpenCV das imagens de exemplo
python visao_computacional.py

# Inicia a API em http://127.0.0.1:5000
python api.py

# Em outro terminal, envia a requisição de exemplo à API
python testar_api.py

# Inicia a interface visual Streamlit
streamlit run app_visual.py

# Abre o notebook para exploração
jupyter notebook tratamento.ipynb
```

Para uma implantação real, desative o modo de depuração do Flask, use um servidor WSGI adequado e configure controles de rede, autenticação, TLS, logs e gestão de segredos. A interface Streamlit e a API devem ser avaliadas e protegidas conforme o ambiente em que forem disponibilizadas.

## Estrutura dos arquivos

```text
.
├── api.py
├── app_visual.py
├── dados_sensores_metaltech.csv
├── tratamento.ipynb
├── treinamento.py
├── visao_computacional.py
├── testar_api.py
├── peca.jpg
├── peca_defeito.jpg
├── modelo_random_forest.pkl
├── modelo_regressao_logistica.pkl
├── modelo_kmeans.pkl
├── scaler_regressao_logistica.pkl
├── scaler_kmeans.pkl
└── imagens de resultados e matrizes de confusão
```

## Limitações e próximos passos

- O particionamento aleatório pode colocar registros próximos no tempo e da mesma máquina tanto em treino quanto em teste. Para estimar generalização futura, uma validação temporal ou por máquina pode ser mais apropriada.
- A interpolação e as regras de valores inválidos devem ser avaliadas com especialistas, especialmente em intervalos longos e no início/fim das séries.
- O conjunto é desbalanceado; resultados devem ser lidos junto à matriz de confusão e ao custo dos erros. O README não fixa métricas numéricas porque elas dependem da execução do treinamento.
- O agrupamento K-Means é exploratório e não substitui um classificador supervisionado nem identifica automaticamente estados operacionais.
- A inspeção visual é baseada em bordas e descritores geométricos simples. Para classificar defeitos com confiabilidade, será necessário um conjunto representativo e rotulado, iluminação e captura controladas, referência geométrica, validação por tipo de defeito e medição de falsos alarmes e defeitos não detectados.
- A API não implementa controles de acesso ou medidas de produção e devolve somente a classe prevista.
- Para operação real, são necessários testes em campo, monitoramento, validação independente, documentação de procedimentos, segurança funcional e aprovação dos responsáveis técnicos.

## Referências conceituais

- Enunciado do Projeto Integrador: *Fábrica Inteligente – MetalTech Soluções Industriais S.A.* (arquivo PDF fornecido para a atividade).
- Lei nº 13.709/2018: Lei Geral de Proteção de Dados Pessoais (LGPD), Brasil.
- Documentação das bibliotecas utilizadas: [Pandas](https://pandas.pydata.org/docs/), [scikit-learn](https://scikit-learn.org/stable/), [OpenCV](https://docs.opencv.org/), [Flask](https://flask.palletsprojects.com/) e [Streamlit](https://docs.streamlit.io/).
