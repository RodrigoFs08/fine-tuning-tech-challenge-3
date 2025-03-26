# Fine-Tuning de Foundation Model para Descrição de Produtos da Amazon
## Link para o vídeo de demonstração
[Link do YouTube](https://youtu.be/1Oh-KqStLgk)

## Sobre o Projeto

Este projeto implementa o fine-tuning de um foundation model (Llama-3.2-1B-Instruct) utilizando o dataset "AmazonTitles-1.3MM". O objetivo é treinar o modelo para receber perguntas sobre títulos de produtos da Amazon e gerar respostas precisas com suas descrições, baseando-se no conhecimento adquirido durante o processo de fine-tuning.

## Problema Abordado

O desafio consiste em criar um modelo capaz de:
- Receber perguntas com contexto obtido do arquivo "trn.json" contido no dataset
- Gerar respostas baseadas na pergunta do usuário sobre o título do produto
- Retornar como resultado a descrição do produto correspondente

## Dataset Utilizado

**The AmazonTitles-1.3MM**
- Consiste em consultas textuais reais de usuários e títulos associados de produtos relevantes encontrados na Amazon
- Contém títulos e descrições de produtos, medidos por ações implícitas ou explícitas dos usuários
- Para este projeto, utilizamos uma amostra do dataset (trn-amostra.jsonl)

### Estrutura do Dataset
- **uid**: ID único do item
- **title**: Título do produto
- **content**: Descrição do produto
- **target_ind** e **target_rel**: Índices e relevância para modelos de recuperação

## Preparação dos Dados

1. Carregamento do dataset a partir do arquivo JSONL
2. Filtragem de entradas com descrição vazia
3. Criação de uma estrutura de prompt/response para o fine-tuning
   - Prompt: "What is the description of product {título}?"
   - Response: Descrição do produto
4. Conversão para o formato Hugging Face Dataset
5. Formatação dos dados usando o template Alpaca para instrução

## Modelo Utilizado

- **Modelo Base**: unsloth/Llama-3.2-1B-Instruct-bnb-4bit
- **Quantização**: 4-bit para reduzir o uso de memória
- **Comprimento Máximo de Sequência**: 2048 tokens
- **Técnica de Fine-Tuning**: LoRA (Low-Rank Adaptation) para PEFT (Parameter-Efficient Fine-Tuning)

### Parâmetros LoRA
- **r**: 16 (rank da matriz de adaptação)
- **lora_alpha**: 16 (escala de adaptação)
- **lora_dropout**: 0 (sem dropout para estabilidade)
- **Módulos Alvo**: Projeções de atenção (q_proj, k_proj, v_proj, o_proj) e MLP (gate_proj, up_proj, down_proj)

## Processo de Fine-Tuning

O fine-tuning foi realizado utilizando a biblioteca TRL (Transformer Reinforcement Learning) com o SFTTrainer (Supervised Fine-Tuning Trainer).

### Parâmetros de Treinamento
- **Batch Size**: 2 por dispositivo
- **Gradient Accumulation Steps**: 4 (efetivamente um batch size de 8)
- **Learning Rate**: 2e-4
- **Warmup Steps**: 5
- **Max Steps**: 60 (limitado para testes rápidos)
- **Otimizador**: AdamW 8-bit (para eficiência de memória)
- **Precisão**: BF16 (bfloat16)

O fine-tuning foi realizado com PEFT (Parameter-Efficient Fine-Tuning) usando LoRA, o que permitiu treinar apenas uma pequena fração dos parâmetros do modelo (cerca de 1.13% dos parâmetros totais).

## Avaliação do Modelo

Após o fine-tuning, o modelo foi avaliado em exemplos de teste para verificar sua capacidade de gerar descrições precisas com base nos títulos dos produtos. Comparamos as respostas geradas antes e depois do fine-tuning.

### Exemplos de Teste
- "What is the description of product Girls Ballet Tutu?"
- "What is the description of product The Wall: Images and Offerings from the Vietnam Veterans Memorial?"

### Resultados
Os resultados mostraram que o modelo fine-tuned é capaz de gerar descrições mais precisas e relevantes para os títulos dos produtos em comparação com o modelo base.

## Tecnologias Utilizadas

- **Unsloth**: Para otimização do modelo Llama
- **PEFT**: Para fine-tuning eficiente em parâmetros
- **Transformers**: Para manipulação do modelo e tokenizer
- **TRL**: Para treinamento supervisionado
- **Datasets**: Para manipulação do dataset
- **Bitsandbytes**: Para quantização de 4-bit
- **PyTorch**: Como framework de deep learning

## Como Reproduzir

1. Instale as dependências necessárias:
```bash
pip install -q unsloth accelerate peft trl bitsandbytes transformers datasets
pip uninstall -y bitsandbytes
pip install bitsandbytes --prefer-binary --upgrade --no-cache-dir
```

2. Prepare o dataset:
   - Baixe o dataset AmazonTitles-1.3MM ou utilize a amostra fornecida (trn-amostra.jsonl)
   - Execute o código de preparação de dados conforme o notebook

3. Carregue o modelo e configure o LoRA:
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
```

4. Execute o fine-tuning:
```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=hf_dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        logging_steps=1,
        output_dir="outputs",
        optim="adamw_8bit",
        seed=3407,
        fp16=False,
        bf16=True,
    ),
)

trainer_stats = trainer.train()
```

5. Salve o modelo treinado:
```python
model.save_pretrained("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")
```

6. Avalie o modelo com exemplos de teste conforme demonstrado no notebook

## Conclusões e Próximos Passos

### Principais Aprendizados
- O uso de técnicas de PEFT como LoRA permite realizar fine-tuning eficiente em modelos grandes
- A preparação adequada dos dados é crucial para o sucesso do fine-tuning
- O modelo fine-tuned apresenta melhor desempenho na geração de descrições de produtos em comparação com o modelo base

### Próximos Passos
- Experimentar com diferentes parâmetros de fine-tuning para melhorar ainda mais o desempenho
- Utilizar o dataset completo para treinamento
- Implementar métricas quantitativas de avaliação (BLEU, ROUGE, etc.)

## Autor

RODRIGO FERREIRA SANTOS

## Licença

Este projeto está licenciado sob a [Licença MIT](LICENSE).
