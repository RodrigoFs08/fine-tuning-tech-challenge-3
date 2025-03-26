# Tech Challenge Fase 3 - Fine-tuning de Foundation Model

## Link para o vídeo de demonstração
[Link do YouTube]()

## O Problema

No Tech Challenge desta fase, foi necessário executar o fine-tuning de um foundation model (Llama, BERT, MISTRAL etc.), utilizando o dataset "The AmazonTitles-1.3MM". O modelo treinado deveria:

- Receber perguntas com um contexto obtido por meio do arquivo json "trn.json" contido dentro do dataset.
- A partir do prompt formado pela pergunta do usuário sobre o título do produto, o modelo deveria gerar uma resposta baseada na pergunta do usuário, trazendo como resultado do aprendizado do fine-tuning os dados da sua descrição.

## Fluxo de Trabalho

### 1. Escolha do Dataset
**Descrição**: O dataset The AmazonTitles-1.3MM consiste em consultas textuais reais de usuários e títulos associados de produtos relevantes encontrados na Amazon e suas descrições, medidos por ações implícitas ou explícitas dos usuários.

### 2. Preparação do Dataset
- Download do dataset AmazonTitles-1.3MM e utilização do arquivo "trn.json"
- Utilização das colunas "title" e "content", que contêm título e descrição respectivamente
- Preparação dos prompts para o fine-tuning, garantindo que estejam organizados de maneira adequada para o treinamento do modelo escolhido
- Limpeza e pré-processamento dos dados conforme necessário para o modelo escolhido

### 3. Chamada do Foundation Model
- Importação do foundation model utilizado
- Teste apresentando o resultado atual do modelo antes do treinamento (para obter uma base de análise após o fine-tuning)
- Avaliação da diferença do resultado gerado após o fine-tuning

### 4. Execução do Fine-Tuning
- Execução do fine-tuning do foundation model selecionado utilizando o dataset preparado
- Documentação do processo de fine-tuning, incluindo os parâmetros utilizados e ajustes específicos realizados no modelo

### 5. Geração de Respostas
- Configuração do modelo treinado para receber perguntas dos usuários
- Geração de respostas baseadas nas perguntas do usuário e nos dados provenientes do fine-tuning, incluindo as fontes fornecidas

## Implementação

A implementação deste projeto foi realizada utilizando o notebook `tech_challenge_3.ipynb`, que contém todo o código necessário para:

1. Carregar e preparar os dados do dataset AmazonTitles-1.3MM
2. Configurar e inicializar o foundation model escolhido
3. Executar o processo de fine-tuning
4. Avaliar o desempenho do modelo antes e depois do fine-tuning
5. Demonstrar a geração de respostas baseadas em perguntas do usuário

## Como Executar

1. Clone este repositório
2. Certifique-se de ter todas as dependências instaladas (listadas no notebook)
3. Execute o notebook `tech_challenge_3.ipynb` em um ambiente Jupyter ou Google Colab
4. Siga as instruções no notebook para cada etapa do processo

## Resultados

Os resultados obtidos demonstram a eficácia do fine-tuning do modelo escolhido para a tarefa de geração de respostas baseadas em títulos de produtos da Amazon. O modelo consegue compreender o contexto da pergunta do usuário e fornecer informações relevantes extraídas das descrições dos produtos.

## Conclusão

Este projeto demonstra a aplicação prática de técnicas de fine-tuning em foundation models para tarefas específicas de processamento de linguagem natural, especificamente para a geração de respostas contextualizadas sobre produtos com base em seus títulos e descrições.
