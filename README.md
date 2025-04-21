## 📦 Execução do Projeto MLOps

### 🧱 Pré-requisitos
- Python 3.10+
- Docker (opcional, para execução containerizada)
- MLflow UI (`mlflow ui`)
- Ambiente virtual: `python -m venv venv`

---

### 🚀 Etapas para rodar localmente

```bash
# 1. Crie e ative o ambiente virtual
python -m venv venv
source venv/bin/activate

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Execute o pré-processamento\python src/preprocess.py

# 4. Treine e registre os modelos
python src/train.py

# 5. Gere dados atuais simulados
python src/generate_current_data.py

# 6. Execute monitoramento
python src/monitor.py

# 7. (Opcional) Execute re-treinamento automático
python src/drift_and_retrain.py

# 8. Inicie o MLflow UI
mlflow ui

# 9. Rode a API localmente
uvicorn app.main:app --reload
```

---

### 🐳 Execução com Docker

```bash
# 1. Construir a imagem Docker
docker build -t mlops-api .

# 2. Executar a API com volume montado para acesso aos artefatos do modelo
docker run -p 8000:8000 -v $(pwd):/app mlops-api
```

Acesse a documentação interativa da API em:
[http://localhost:8000/docs](http://localhost:8000/docs)

Isso garante que o container consiga carregar os artefatos do modelo diretamente da pasta `mlruns`, e funcione com total compatibilidade com o MLflow.

---

## 🔁 Monitoramento e Re-treinamento Automático

Para garantir a performance contínua do modelo, foi implementado um mecanismo de monitoramento com Evidently AI, que avalia periodicamente o desvio (drift) entre os dados de produção e os dados usados no treinamento original.

- O script `generate_current_data.py` simula dados atuais.
- O script `monitor.py` gera relatórios visuais em HTML.
- O script `drift_and_retrain.py`:
  - Detecta automaticamente a presença de `data drift`.
  - Se identificado, executa `train.py` para re-treinar o modelo.
  - O novo modelo é registrado como uma nova versão no MLflow Model Registry.

A API (`main.py`) carrega sempre a **última versão registrada** automaticamente. Assim, o sistema é capaz de se adaptar a mudanças no comportamento dos dados sem necessidade de intervenção manual.

Esse ciclo fecha o loop de MLOps com: **Treinamento → Deploy → Monitoramento → Re-treinamento → Redeploy**.
