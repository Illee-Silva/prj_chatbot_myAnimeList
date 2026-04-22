# Chatbot MyAnimeList (Flask + TF-IDF)

Projeto em Python para:
1. Coletar textos de páginas web (crawler).
2. Pré-processar com **NLTK** e **spaCy**.
3. Filtrar/ranquear documentos com **TF-IDF**.
4. Responder perguntas via API Flask.

## URLs padrão
- https://myanimelist.net/topanime.php?type=bypopularity
- https://myanimelist.net/anime/season
- https://myanimelist.net/topanime.php?type=upcoming

## Como rodar localmente

### 1) Criar ambiente virtual (Windows)
```bash
python -m venv .venv
```

### 2) Instalar dependências na venv
```bash
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3) Baixar modelo do spaCy (opcional)
```bash
.venv\Scripts\python.exe -m spacy download en_core_web_sm
```

### 4) Iniciar a API Flask
```bash
.venv\Scripts\python.exe app.py
```

A API sobe em `http://127.0.0.1:5000`.

## Testes no Postman

### Healthcheck
- Método: `GET`
- URL: `http://127.0.0.1:5000/health`

### Coletar e indexar documentos
- Método: `POST`
- URL: `http://127.0.0.1:5000/crawl`
- Body (raw JSON):
```json
{
  "urls": [
    "https://myanimelist.net/topanime.php?type=bypopularity",
    "https://myanimelist.net/anime/season",
    "https://myanimelist.net/topanime.php?type=upcoming"
  ]
}
```

### Fazer pergunta ao chatbot
- Método: `POST`
- URL: `http://127.0.0.1:5000/ask`
- Body (raw JSON):
```json
{
  "question": "Quais animes aparecem entre os mais populares e os da temporada atual?",
  "top_k": 3
}
```

## Estrutura de arquivos
- `app.py`: API Flask e endpoints.
- `crawler.py`: coleta e extração de texto das páginas.
- `preprocess.py`: limpeza textual com NLTK e spaCy.
- `indexer.py`: indexação TF-IDF e busca por similaridade.

## Observações
- Se o site limitar scraping em alguns momentos, tente novamente.
- O índice TF-IDF é recriado toda vez que `/crawl` é chamado.
- Se o modelo `en_core_web_sm` não estiver instalado, o projeto usa fallback com `spacy.blank("en")`.
