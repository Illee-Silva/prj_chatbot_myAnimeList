"""Crawler simples para extrair textos de páginas web do MyAnimeList."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

import requests
from bs4 import BeautifulSoup


@dataclass
class CrawledDocument:
    """Representa um documento textual extraído de uma URL."""

    url: str
    title: str
    text: str
    anime_titles: List[str] = field(default_factory=list)


class WebCrawler:
    """Responsável por baixar páginas e extrair texto relevante."""

    def __init__(self) -> None:
        # User-Agent para evitar bloqueios básicos de scraping.
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        }

    def crawl(self, urls: List[str]) -> List[CrawledDocument]:
        """Extrai textos de uma lista de URLs e retorna documentos limpos."""
        documents: List[CrawledDocument] = []

        for url in urls:
            try:
                response = requests.get(url, headers=self.headers, timeout=20)
                response.raise_for_status()
                document = self._extract_from_html(url=url, html=response.text)
                if document.text.strip():
                    documents.append(document)
            except requests.RequestException as exc:
                # Erro de rede fica registrado como documento mínimo para diagnóstico.
                documents.append(
                    CrawledDocument(
                        url=url,
                        title="Erro ao coletar",
                        text=f"Falha no acesso da página: {exc}",
                        anime_titles=[],
                    )
                )

        return documents

    def _extract_from_html(self, url: str, html: str) -> CrawledDocument:
        """Extrai título, corpo textual e lista de animes da página HTML."""
        soup = BeautifulSoup(html, "html.parser")

        # Remove elementos que geram ruído textual.
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()

        page_title = soup.title.get_text(strip=True) if soup.title else "Sem título"
        anime_titles = self._extract_anime_titles(soup)

        # Estratégia principal: focar em nós com maior chance de conteúdo útil.
        content_blocks = []
        selectors = [
            "h1",
            "h2",
            "h3",
            "p",
            "li",
            "td",
            "div.ranking-header",
            "div.detail",
            "div.seasonal-anime",
        ]

        for selector in selectors:
            for element in soup.select(selector):
                text = element.get_text(" ", strip=True)
                if text and len(text) > 20:
                    content_blocks.append(text)

        # Inclui lista de animes em formato textual para fortalecer a busca TF-IDF.
        if anime_titles:
            content_blocks.append("Anime list: " + " | ".join(anime_titles))

        # Fallback: caso os seletores retornem pouco conteúdo, pega texto geral.
        if not content_blocks:
            raw_text = soup.get_text(" ", strip=True)
            content_blocks = [raw_text]

        # Remove duplicatas mantendo ordem.
        unique_blocks = list(dict.fromkeys(content_blocks))
        full_text = "\n".join(unique_blocks)

        return CrawledDocument(url=url, title=page_title, text=full_text, anime_titles=anime_titles)

    def _extract_anime_titles(self, soup: BeautifulSoup) -> List[str]:
        """Extrai nomes de animes a partir de links e cabeçalhos comuns do MAL."""
        titles: List[str] = []
        noise = {
            "seasonal anime",
            "add to my list",
            "schedule",
            "archive",
            "later",
            "winter 2026",
            "spring 2026",
            "summer 2026",
            "fall 2026",
            "...",
        }

        # O MAL expõe boa parte dos títulos em links para /anime/<id>/<slug>.
        for link in soup.select("a[href*='/anime/']"):
            href = (link.get("href") or "").strip()
            title = link.get_text(" ", strip=True)
            lowered = title.lower()

            # Aceita apenas links de obra com id numérico.
            if not re.search(r"/anime/\d+", href):
                continue

            if not (2 < len(title) <= 120):
                continue

            if lowered in noise:
                continue

            if len(title.split()) < 2 and len(title) < 6:
                continue

            titles.append(title)

        # Remove entradas repetidas preservando ordem.
        unique_titles = list(dict.fromkeys(titles))

        # Limita para reduzir ruído excessivo no índice.
        return unique_titles[:120]
