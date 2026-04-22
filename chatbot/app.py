"""API Flask do chatbot de pesquisa textual com TF-IDF."""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

from flask import Flask, jsonify, render_template, request

from crawler import WebCrawler
from indexer import TFIDFChatbot

app = Flask(__name__)

# URLs definidas no enunciado do projeto.
DEFAULT_URLS = [
    "https://myanimelist.net/topanime.php?type=bypopularity",
    "https://myanimelist.net/anime/season",
    "https://myanimelist.net/topanime.php?type=upcoming",
]

crawler = WebCrawler()
chatbot = TFIDFChatbot()

# Filtro por categorias para facilitar manutenção e expansão.
SAFETY_PATTERNS_BY_CATEGORY = {
    "odio_discriminacao": [
        r"\bracism[oa]?\b",
        r"\bracist[ao]?\b",
        r"\bracista\b",
        r"\bxenofob[iaoa]?\b",
        r"\bhomofob[iaoa]?\b",
        r"\btransfob[iaoa]?\b",
        r"\bmisogin[iaoa]?\b",
        r"\bdiscriminac(?:ao|oes)?\b",
        r"\bpreconceito\b",
        r"\bodio\b",
    ],
    "ofensas_e_profanidade": [
        r"\bpalavrao\b",
        r"\binsulto\b",
        r"\bofensa\b",
        r"\bseu\s+merda\b",
        r"\bvai\s+se\s+foder\b",
        r"\bvai\s+tomar\s+no\s+cu\b",
        r"\bfilho\s+da\s+puta\b",
        r"\barrombado\b",
        r"\bcorno\b",
        r"\bidiota\b",
        r"\bbabaca\b",
    ],
    "ameacas_e_assedio": [
        r"\bameaca(?:r|s)?\b",
        r"\bassedio\b",
        r"\bbullying\b",
        r"\bte\s+mato\b",
    ],
}


def normalize_for_safety(text: str) -> str:
    """Normaliza texto para detecção robusta de termos inadequados."""
    lowered = (text or "").lower()
    normalized = unicodedata.normalize("NFKD", lowered)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def get_blocked_category(text: str) -> Optional[str]:
    """Retorna categoria bloqueada quando algum padrão de segurança for detectado."""
    normalized = normalize_for_safety(text)
    for category, patterns in SAFETY_PATTERNS_BY_CATEGORY.items():
        for pattern in patterns:
            if re.search(pattern, normalized):
                return category
    return None


def is_blocked_topic(text: str) -> bool:
    """Indica se a mensagem deve ser bloqueada pela camada de segurança."""
    return get_blocked_category(text) is not None


def safety_message() -> str:
    """Mensagem padrao para conteúdo bloqueado."""
    return (
        "Nao posso ajudar com racismo, xenofobia ou linguagem inapropriada. "
        "Posso te ajudar com animes. Tente, por exemplo: "
        "'Quais sao os animes mais bem avaliados?', "
        "'Quais animes estao em lancamento na temporada atual?' "
        "ou 'Me fale sobre Berserk'."
    )


@app.get("/")
def home_page():
    """Interface web simples para conversar com o chatbot."""
    return render_template("chat.html")


@app.get("/health")
def health_check():
    """Endpoint simples para verificar se a API está ativa."""
    return jsonify({"status": "ok"})


@app.post("/crawl")
def crawl_and_index():
    """Coleta páginas web e recria índice TF-IDF."""
    payload = request.get_json(silent=True) or {}
    urls = payload.get("urls", DEFAULT_URLS)

    if not isinstance(urls, list) or not urls:
        return jsonify({"error": "Campo 'urls' deve ser uma lista com ao menos 1 URL."}), 400

    clean_urls = []
    for item in urls:
        if not isinstance(item, str):
            continue
        candidate = item.strip()
        if candidate.startswith("http://") or candidate.startswith("https://"):
            clean_urls.append(candidate)

    clean_urls = list(dict.fromkeys(clean_urls))[:10]
    if not clean_urls:
        return jsonify({"error": "Nenhuma URL valida foi informada."}), 400

    try:
        documents = crawler.crawl(urls=clean_urls)
    except Exception as exc:  # pragma: no cover - proteção para runtime inesperado.
        return jsonify({"error": f"Falha inesperada no crawl: {exc}"}), 500

    # Filtra documentos vazios e também entradas de erro de rede.
    valid_documents = [
        doc
        for doc in documents
        if doc.text and doc.text.strip() and doc.title != "Erro ao coletar"
    ]
    if not valid_documents:
        return jsonify({"error": "Nenhum texto válido foi coletado."}), 400

    chatbot.build_index(valid_documents)

    return jsonify(
        {
            "message": "Coleta e indexação concluídas.",
            "documents_indexed": len(valid_documents),
            "sources": [{"title": d.title, "url": d.url} for d in valid_documents],
        }
    )


@app.post("/ask")
def ask_question():
    """Recebe pergunta do usuário e retorna resposta + top resultados."""
    payload = request.get_json(silent=True) or {}
    question = payload.get("question", "").strip()

    try:
        top_k = int(payload.get("top_k", 3))
    except (TypeError, ValueError):
        top_k = 3
    top_k = max(1, min(top_k, 10))

    if not question:
        return jsonify({"error": "Campo 'question' é obrigatório."}), 400

    if len(question) > 400:
        return jsonify({"error": "Pergunta muito longa. Use no máximo 400 caracteres."}), 400

    blocked_category = get_blocked_category(question)
    if blocked_category:
        return jsonify(
            {
                "intent": "safety",
                "category": blocked_category,
                "answer": safety_message(),
                "results": [],
            }
        )

    try:
        answer, results, intent = chatbot.generate_response(question=question, top_k=top_k)
    except ValueError as exc:
        return jsonify({"error": str(exc), "hint": "Execute /crawl antes de /ask."}), 400
    except Exception as exc:  # pragma: no cover - proteção para runtime inesperado.
        return (
            jsonify(
                {
                    "intent": "error",
                    "answer": "Ocorreu um erro interno ao processar sua pergunta.",
                    "error": str(exc),
                    "results": [],
                }
            ),
            500,
        )

    return jsonify(
        {
            "intent": intent,
            "answer": answer,
            "results": [
                {
                    "score": round(r.score, 6),
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet,
                }
                for r in results
            ],
        }
    )


if __name__ == "__main__":
    # Execução local para testes no Postman.
    app.run(host="127.0.0.1", port=5000, debug=True)
