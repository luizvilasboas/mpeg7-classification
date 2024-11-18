# mpeg7-classification

Este projeto realiza a classificação de imagens do conjunto de dados MPEG7 modificado. Ele envolve a segmentação de imagens, extração de características morfológicas, normalização dos dados e avaliação de classificadores.

## Estrutura do Projeto

```
project/
├── dataset/                # Conjunto de dados com subpastas para cada classe
├── metrics/                # Pasta para salvar os resultados (métricas)
├── main.py                 # Código principal do projeto
├── requirements.txt        # Arquivo de dependências
└── README.md               # Este arquivo
```

## Dependências

As bibliotecas necessárias estão listadas no arquivo `requirements.txt`. Para instalar as dependências, execute:

```
pip install -r requirements.txt
```

### Bibliotecas utilizadas:
- `numpy`
- `opencv-python`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `pandas`

## Conjunto de Dados

O conjunto de dados deve estar organizado na pasta `dataset` no seguinte formato:

```
dataset/
├── class1/
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── class2/
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── ...
```

Cada subpasta representa uma classe, e as imagens dentro dela pertencem a essa classe.

### Pré-processamento
1. **Segmentação**: As imagens são segmentadas para isolar as formas presentes.
2. **Extração de Características**: Calcula-se área, perímetro e circularidade das formas segmentadas.
3. **Normalização**: Os dados são normalizados para melhorar o desempenho dos classificadores.

## Como Executar

1. Clone o repositório usando o Git (instale o [Git LFS](https://git-lfs.com/) para clonar o dataset também).
2. Execute o script principal `main.py`:

```
python main.py
```

3. Após a execução, os resultados serão salvos na pasta `metrics/`:
   - Relatórios de classificação (`.json`) para cada classificador.
   - Matrizes de confusão (`.png`) para cada classificador.

## Resultados

Os resultados incluem:
- **Relatório de Classificação**: Contendo acurácia, precisão, recall e F1-score.
- **Matriz de Confusão**: Uma representação visual dos acertos e erros de classificação.

Os arquivos salvos em `metrics/` podem ser usados diretamente para análise e inclusão no relatório final.

## Personalizações

- **Classificadores**: Atualmente, o projeto utiliza k-NN e Random Forest. Outros classificadores podem ser adicionados ao código.
- **Extração de Características**: A lógica de extração pode ser ajustada para incluir mais características, caso necessário.