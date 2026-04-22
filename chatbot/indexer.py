"""Indexação e busca textual com TF-IDF e respostas conversacionais."""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from crawler import CrawledDocument
from preprocess import TextPreprocessor


@dataclass
class SearchResult:
    """Resultado de busca semântica simples por similaridade de cosseno."""

    score: float
    title: str
    url: str
    snippet: str


class TFIDFChatbot:
    """Chatbot que responde com base em documentos indexados."""

    def __init__(self) -> None:
        self.preprocessor = TextPreprocessor()

        # Vetor principal por palavras + bigramas.
        self.word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)

        # Vetor auxiliar por caracteres para reduzir impacto de variações PT/EN.
        self.char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)

        self.documents: List[CrawledDocument] = []
        self.normalized_docs: List[str] = []
        self.raw_docs_lower: List[str] = []
        self.word_tfidf_matrix = None
        self.char_tfidf_matrix = None
        self.jikan_base_url = "https://api.jikan.moe/v4/anime"
        self.known_anime_titles: List[str] = []

    def build_index(self, documents: List[CrawledDocument]) -> None:
        """Pré-processa documentos e cria matrizes TF-IDF."""
        self.documents = documents
        self.normalized_docs = [self.preprocessor.normalize_text(doc.text) for doc in documents]
        self.raw_docs_lower = [doc.text.lower() for doc in documents]
        self.known_anime_titles = self._collect_known_titles(documents)

        if not self.normalized_docs:
            raise ValueError("Nenhum documento foi informado para indexação.")

        self.word_tfidf_matrix = self.word_vectorizer.fit_transform(self.normalized_docs)
        self.char_tfidf_matrix = self.char_vectorizer.fit_transform(self.raw_docs_lower)

    def ask(self, question: str, top_k: int = 3) -> List[SearchResult]:
        """Retorna os documentos mais relevantes para a pergunta do usuário."""
        if self.word_tfidf_matrix is None or not self.documents:
            raise ValueError("Índice TF-IDF ainda não foi construído. Por Favor clique no botão de atualizar a base")

        expanded_question = self._expand_portuguese_query(question)
        normalized_question = self.preprocessor.normalize_text(expanded_question)

        question_word_vector = self.word_vectorizer.transform([normalized_question])
        question_char_vector = self.char_vectorizer.transform([question.lower()])

        word_sim = cosine_similarity(question_word_vector, self.word_tfidf_matrix).flatten()
        char_sim = cosine_similarity(question_char_vector, self.char_tfidf_matrix).flatten()

        # Combinação ponderada para maior robustez em perguntas PT-BR.
        similarities = (0.75 * word_sim) + (0.25 * char_sim)

        best_indexes = np.argsort(similarities)[::-1][: max(top_k, 1)]

        results: List[SearchResult] = []
        for idx in best_indexes:
            doc = self.documents[idx]
            score = float(similarities[idx])
            snippet = doc.text[:400].replace("\n", " ")
            results.append(
                SearchResult(score=score, title=doc.title, url=doc.url, snippet=snippet)
            )

        return results

    def generate_response(self, question: str, top_k: int = 3) -> Tuple[str, List[SearchResult], str]:
        """Gera resposta conversacional com tratamento de intenção."""
        normalized_q = question.strip().lower()
        intent = self._detect_intent(normalized_q)

        if intent == "greeting":
            return (
                "Oi! Eu posso te ajudar com animes do MyAnimeList. "
                "Você pode perguntar, por exemplo: 'Quais são os animes mais bem avaliados?', "
                "'Quais animes estão em lançamento na temporada atual?' "
                "ou 'Me fale sobre Berserk'.",
                [],
                intent,
            )

        if intent == "anime_detail":
            anime_name = self._extract_anime_name(question)
            if anime_name:
                resolved_name = self._resolve_anime_name(anime_name)
                target_name = resolved_name or anime_name

                api_detail = self._fetch_anime_details(target_name, anime_name)
                if api_detail:
                    return api_detail, [], intent

                detail = self._find_anime_mentions(anime_name=target_name, top_k=top_k)
                if detail:
                    return detail, [], intent
                return (
                    f"Nao consegui encontrar informacoes de '{anime_name}'. "
                    "Tente outro nome ou uma destas perguntas: "
                    "'Quais sao os animes mais bem avaliados?' ou "
                    "'Quais animes estao em lancamento na temporada atual?'.",
                    [],
                    intent,
                )

            return self._build_unknown_answer(), [], "unknown"

        results = self.ask(question=question, top_k=top_k)

        if intent == "top_rated":
            popular_doc = self._find_doc_by_url_keyword("bypopularity")
            if popular_doc:
                top_titles = popular_doc.anime_titles[:10]
                if top_titles:
                    formatted = self._format_top_list(top_titles)
                    answer = (
                        "Claro! Usando a lista de maior popularidade como referencia de destaque no MAL, "
                        "estes sao alguns animes em evidencia:\n"
                        f"{formatted}"
                    )
                    return answer, results, intent

        if intent == "season_current":
            season_doc = self._find_doc_by_url_keyword("/anime/season")
            if season_doc:
                season_titles = season_doc.anime_titles[:10]
                if season_titles:
                    formatted = self._format_top_list(season_titles)
                    answer = (
                        "Aqui estao alguns animes em lancamento/temporada atual no MyAnimeList:\n"
                        f"{formatted}"
                    )
                    return answer, results, intent

        if intent == "general":
            if not results or results[0].score < 0.03:
                return self._build_unknown_answer(), [], "unknown"

        answer = self.build_answer(question=question, results=results)
        return answer, results, intent

    def _detect_intent(self, q: str) -> str:
        """Detecta intenção da pergunta para respostas mais naturais."""
        if re.fullmatch(r"(oi|oi+|ola|olá|eae|eae+|hey|hello|salve|bom dia|boa tarde|boa noite)\W*", q):
            return "greeting"

        if re.search(r"\b(?:me\s+fale|me\s+fala|fale|fala)\s+sobre\b", q):
            return "anime_detail"

        if (
            "bem avaliados" in q
            or "melhores" in q
            or "top" in q
            or "mais populares" in q
        ):
            return "top_rated"

        if "temporada atual" in q or "lancamento" in q or "lançamento" in q or "season" in q:
            return "season_current"

        return "general"

    def _expand_portuguese_query(self, question: str) -> str:
        """Expande termos em PT-BR para aumentar match com documentos em inglês."""
        mapping = {
            "animes": "anime",
            "anime": "anime",
            "mais": "top",
            "bem avaliados": "top rated",
            "melhores": "best top",
            "populares": "popular popularity",
            "popularidade": "popularity",
            "temporada": "season",
            "atual": "current",
            "lancamento": "upcoming release",
            "lançamento": "upcoming release",
            "quais": "which",
        }

        expanded = question.lower()
        for pt_term, en_term in mapping.items():
            expanded = expanded.replace(pt_term, f"{pt_term} {en_term}")

        return expanded

    def _extract_anime_name(self, question: str) -> Optional[str]:
        """Extrai nome do anime em frases como 'me fale sobre X'."""
        match = re.search(
            r"(?:me\s+fale\s+sobre|me\s+fala\s+sobre|fale\s+sobre|fala\s+sobre|"
            r"me\s+conte\s+sobre|me\s+diga\s+sobre|sobre\s+o\s+anime)\s+(.+)$",
            question,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        return match.group(1).strip(" ?!.,")

    @staticmethod
    def _normalize_for_match(value: str) -> str:
        """Normaliza string para comparação tolerante a acentos e pontuação."""
        normalized = unicodedata.normalize("NFKD", value)
        normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        normalized = normalized.lower()
        normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _collect_known_titles(self, documents: List[CrawledDocument]) -> List[str]:
        """Consolida títulos extraídos pelo crawler para fuzzy search local."""
        titles: List[str] = []
        for doc in documents:
            titles.extend(doc.anime_titles)
        unique_titles = list(dict.fromkeys(titles))
        return unique_titles

    def _resolve_anime_name(self, anime_name: str) -> Optional[str]:
        """Resolve typo leve em nome de anime com fuzzy matching local."""
        if not self.known_anime_titles:
            return None

        normalized_index = {
            self._normalize_for_match(title): title
            for title in self.known_anime_titles
            if title and title.strip()
        }
        query_norm = self._normalize_for_match(anime_name)
        if not query_norm:
            return None

        if query_norm in normalized_index:
            return normalized_index[query_norm]

        candidates = difflib.get_close_matches(
            query_norm,
            list(normalized_index.keys()),
            n=1,
            cutoff=0.78,
        )
        if not candidates:
            return None

        return normalized_index[candidates[0]]

    def _find_doc_by_url_keyword(self, keyword: str) -> Optional[CrawledDocument]:
        """Busca um documento indexado por trecho da URL."""
        for doc in self.documents:
            if keyword in doc.url:
                return doc
        return None

    def _find_anime_mentions(self, anime_name: str, top_k: int = 3) -> Optional[str]:
        """Procura menções de um anime específico nos documentos coletados."""
        pattern = anime_name.lower()
        matches = []

        for doc in self.documents:
            in_titles = any(pattern in t.lower() for t in doc.anime_titles)
            in_text = pattern in doc.text.lower()
            if in_titles or in_text:
                matches.append(doc)

        if not matches:
            return None

        lines = [f"Encontrei informacoes sobre '{anime_name}' nestas fontes:"]
        for doc in matches[:top_k]:
            lines.append(f"- {doc.title} ({doc.url})")

        return "\n".join(lines)

    def _fetch_anime_details(self, anime_name: str, original_query: Optional[str] = None) -> Optional[str]:
        """Busca detalhes do anime na API pública Jikan (MyAnimeList)."""
        try:
            response = requests.get(
                self.jikan_base_url,
                params={"q": anime_name, "limit": 5, "sfw": "false"},
                timeout=12,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError):
            return None

        data = payload.get("data") or []
        if not data:
            return None

        anime = self._pick_best_jikan_match(original_query or anime_name, data)
        if not anime:
            anime = data[0]

        title = anime.get("title") or anime_name
        year = anime.get("year")
        season = anime.get("season")
        anime_type = anime.get("type")
        episodes = anime.get("episodes")
        score = anime.get("score")
        status = anime.get("status")
        synopsis = (anime.get("synopsis") or "Sem resumo disponível no momento.").replace("\n", " ").strip()
        mal_url = anime.get("url")

        launch_info = "não informado"
        if season and year:
            launch_info = f"{season.title()} de {year}"
        elif year:
            launch_info = str(year)

        score_info = f"{score}" if score is not None else "não informado"
        episodes_info = str(episodes) if episodes is not None else "não informado"
        type_info = anime_type or "não informado"
        status_info = status or "não informado"

        lines = [
            f"Claro! Aqui vai um resumo de {title}:",
            f"- Tipo: {type_info}",
            f"- Lançamento: {launch_info}",
            f"- Episódios: {episodes_info}",
            f"- Nota (MAL): {score_info}",
            f"- Status: {status_info}",
            f"- Resumo: {synopsis}",
        ]

        if original_query and self._normalize_for_match(original_query) != self._normalize_for_match(title):
            lines.insert(1, f"- Interpretei sua busca como: {title}")

        if mal_url:
            lines.append(f"- Página no MyAnimeList: {mal_url}")

        return "\n".join(lines)

    def _pick_best_jikan_match(self, query: str, candidates: List[dict]) -> Optional[dict]:
        """Seleciona melhor candidato da Jikan com similaridade textual."""
        query_norm = self._normalize_for_match(query)
        best_score = -1.0
        best_item: Optional[dict] = None

        for item in candidates:
            titles = [
                item.get("title") or "",
                item.get("title_english") or "",
                item.get("title_japanese") or "",
            ]
            titles.extend(item.get("titles", []))

            candidate_names: List[str] = []
            for title_value in titles:
                if isinstance(title_value, dict):
                    candidate_names.append(title_value.get("title") or "")
                else:
                    candidate_names.append(str(title_value))

            max_ratio = 0.0
            for name in candidate_names:
                name_norm = self._normalize_for_match(name)
                if not name_norm:
                    continue
                ratio = difflib.SequenceMatcher(None, query_norm, name_norm).ratio()
                if ratio > max_ratio:
                    max_ratio = ratio

            if max_ratio > best_score:
                best_score = max_ratio
                best_item = item

        if best_score < 0.45:
            return None

        return best_item

    @staticmethod
    def _build_unknown_answer() -> str:
        """Mensagem padrão quando a pergunta não é reconhecida com segurança."""
        return (
            "Nao entendi muito bem sua pergunta. Tente uma destas opcoes:\n"
            "- 'Quais sao os animes mais bem avaliados?'\n"
            "- 'Quais animes estao em lancamento na temporada atual?'\n"
            "- 'Me fale sobre Naruto'"
        )

    @staticmethod
    def _format_top_list(titles: List[str]) -> str:
        """Formata uma lista curta de títulos para saída de conversa."""
        lines = []
        for idx, title in enumerate(titles, start=1):
            lines.append(f"{idx}. {title}")
        return "\n".join(lines)

    @staticmethod
    def build_answer(question: str, results: List[SearchResult]) -> str:
        """Monta resposta textual amigável a partir dos resultados mais próximos."""
        if not results:
            return "Nao encontrei informacoes suficientes nos documentos coletados."

        best = results[0]
        lines = [
            f"Pergunta: {question}",
            "Resposta baseada nos textos coletados:",
            f"- Fonte principal: {best.title} ({best.url})",
            f"- Similaridade: {best.score:.4f}",
            f"- Trecho: {best.snippet}",
        ]
        return "\n".join(lines)
