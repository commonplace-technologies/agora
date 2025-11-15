#!/usr/bin/env python3
"""
Agora LibraryThing Agentic Text Acquisition Script
==================================================

Multi-agent system where each author operates as an autonomous agent with:
- Automatic profile inference (era, public domain status)
- Intelligent strategy selection
- Autonomous decision-making
- Independent book acquisition

Source Agents - Specialists for each repository:
- GutenbergAgent: Expert in pre-1928 classics (90% confidence)
- InternetArchiveAgent: All eras, scanned books (60% confidence)
- WikisourceAgent: Ancient texts, foreign languages (85% confidence)

Shared Knowledge Base:
- Tracks successful/failed searches
- Records source performance
- Recommends best strategies
- Manages rate limiting

Usage:
    python scripts/acquire_from_librarything_agentic.py --input library.tsv
    python scripts/acquire_from_librarything_agentic.py --input library.tsv --max-concurrent 20
    python scripts/acquire_from_librarything_agentic.py --input library.tsv --author-filter "Marx,Whitman" --verbose
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import re
import sys
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from urllib.parse import quote_plus
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# GUTENBERG CATALOG CACHE
# ============================================================================

class GutenbergCatalog:
    """
    Local cache of the entire Project Gutenberg catalog.

    Downloads the CSV catalog once (~20 MB) and provides instant local searches.
    This eliminates rate limiting issues during the search phase.
    """

    CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
    CACHE_LIFETIME_DAYS = 7

    def __init__(self, cache_path: Path = Path("data/gutenberg_catalog.csv")):
        self.cache_path = cache_path
        self.catalog_by_author = {}  # author_normalized -> list of books
        self.catalog_by_id = {}  # book_id -> book info
        self._load_or_download()

    def _is_stale(self) -> bool:
        """Check if cached catalog is older than CACHE_LIFETIME_DAYS"""
        if not self.cache_path.exists():
            return True

        file_age_days = (time.time() - self.cache_path.stat().st_mtime) / 86400
        return file_age_days > self.CACHE_LIFETIME_DAYS

    def _load_or_download(self):
        """Download catalog if not cached or stale, then parse it"""
        if self._is_stale():
            logger.info("Downloading fresh Gutenberg catalog...")
            self._download_catalog()
        else:
            logger.info("Using cached Gutenberg catalog")

        self._parse_catalog()

    def _download_catalog(self):
        """Download the complete Gutenberg CSV catalog"""
        try:
            response = requests.get(self.CATALOG_URL, timeout=30)
            response.raise_for_status()

            self.cache_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.cache_path, 'wb') as f:
                f.write(response.content)

            size_mb = len(response.content) / 1024 / 1024
            logger.info(f"Downloaded Gutenberg catalog: {size_mb:.1f} MB")

        except Exception as e:
            logger.error(f"Failed to download Gutenberg catalog: {e}")
            raise

    def _parse_catalog(self):
        """Parse CSV into searchable structures"""
        try:
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    # Skip non-text entries
                    if row['Type'] != 'Text':
                        continue

                    # Skip if no author
                    if not row.get('Authors'):
                        continue

                    # Parse book info
                    book_id = row['Text#']
                    book_info = {
                        'id': book_id,
                        'title': row['Title'],
                        'author': row['Authors'],
                        'language': row.get('Language', 'en'),
                        'issued': row.get('Issued', ''),
                    }

                    # Index by book ID
                    self.catalog_by_id[book_id] = book_info

                    # Index by author (normalized)
                    author_keys = self._normalize_author(row['Authors'])
                    for author_key in author_keys:
                        if author_key not in self.catalog_by_author:
                            self.catalog_by_author[author_key] = []
                        self.catalog_by_author[author_key].append(book_info)

            logger.info(f"Loaded {len(self.catalog_by_id)} books from {len(self.catalog_by_author)} authors")

        except Exception as e:
            logger.error(f"Failed to parse Gutenberg catalog: {e}")
            raise

    def _normalize_author(self, author: str) -> List[str]:
        """
        Normalize author name for matching.
        Returns multiple variations to improve match rate.

        Example: "Marx, Karl, 1818-1883" -> ["marx", "karl marx", "marx karl"]
        """
        if not author:
            return []

        # Remove dates in parentheses or after comma-digits
        author = re.sub(r',?\s*\d{4}-?\d{0,4}', '', author)

        # Split on commas to handle "LastName, FirstName" format
        parts = [p.strip() for p in author.split(',') if p.strip()]

        normalized = []

        # Add just last name
        if parts:
            normalized.append(re.sub(r'[^\w\s]', '', parts[0]).lower().strip())

        # Add "FirstName LastName" format
        if len(parts) >= 2:
            first_last = f"{parts[1]} {parts[0]}"
            normalized.append(re.sub(r'[^\w\s]', '', first_last).lower().strip())

        # Add "LastName FirstName" format
        if len(parts) >= 2:
            last_first = f"{parts[0]} {parts[1]}"
            normalized.append(re.sub(r'[^\w\s]', '', last_first).lower().strip())

        # Deduplicate
        return list(set(n for n in normalized if n))

    def _normalize_title(self, title: str) -> str:
        """Normalize title for fuzzy matching"""
        # Remove subtitles (after colon or semicolon)
        title = re.split(r'[;:]', title)[0]

        # Remove common articles and normalize
        title = re.sub(r'^(the|a|an)\s+', '', title.lower())
        title = re.sub(r'[^\w\s]', '', title)

        return title.strip()

    def _title_similarity(self, title1: str, title2: str) -> float:
        """Calculate title similarity (0.0 to 1.0)"""
        norm1 = self._normalize_title(title1)
        norm2 = self._normalize_title(title2)

        # Exact match
        if norm1 == norm2:
            return 1.0

        # One contains the other
        if norm1 in norm2 or norm2 in norm1:
            return 0.9

        # Calculate word overlap
        words1 = set(norm1.split())
        words2 = set(norm2.split())

        if not words1 or not words2:
            return 0.0

        overlap = len(words1 & words2)
        total = len(words1 | words2)

        return overlap / total if total > 0 else 0.0

    def search_by_author_and_title(self, author: str, title: str, min_similarity: float = 0.7) -> Optional[Dict]:
        """
        Search for a book by author and title using local catalog.

        Returns the best match or None if no good match found.
        This is instant and requires no API calls!
        """
        # Normalize author name and get all variations
        author_keys = self._normalize_author(author)

        best_match = None
        best_score = 0.0

        # Search all author name variations
        for author_key in author_keys:
            if author_key in self.catalog_by_author:
                for book in self.catalog_by_author[author_key]:
                    similarity = self._title_similarity(book['title'], title)

                    if similarity > best_score:
                        best_score = similarity
                        best_match = book

        # Return best match if it meets minimum threshold
        if best_match and best_score >= min_similarity:
            return {
                'id': best_match['id'],
                'title': best_match['title'],
                'author': best_match['author'],
                'language': best_match['language'],
                'similarity_score': best_score
            }

        return None

    def get_all_books_by_author(self, author: str) -> List[Dict]:
        """Get all books by an author (for debugging/exploration)"""
        author_keys = self._normalize_author(author)

        all_books = []
        seen_ids = set()

        for author_key in author_keys:
            if author_key in self.catalog_by_author:
                for book in self.catalog_by_author[author_key]:
                    if book['id'] not in seen_ids:
                        all_books.append(book)
                        seen_ids.add(book['id'])

        return all_books


# ============================================================================
# DATA MODELS
# ============================================================================

class Era(Enum):
    """Historical eras for author classification"""
    ANCIENT = "ancient"          # Before 500 CE
    CLASSICAL = "classical"      # 500-1500 CE
    EARLY_MODERN = "early_modern" # 1500-1800 CE
    MODERN = "modern"            # 1800-1928 (public domain)
    CONTEMPORARY = "contemporary" # After 1928


@dataclass
class Book:
    """Represents a book from LibraryThing export"""
    title: str
    author: str
    author_normalized: str
    isbn: Optional[str] = None
    publication_date: Optional[str] = None
    source_id: Optional[str] = None  # Generic ID (gutenberg_id, archive_id, etc.)
    source_url: Optional[str] = None
    source_name: Optional[str] = None  # "gutenberg", "archive", "wikisource"
    downloaded: bool = False
    download_path: Optional[str] = None
    error: Optional[str] = None
    confidence: float = 0.0  # Agent's confidence in the match


@dataclass
class AuthorProfile:
    """Profile of an author inferred from their works"""
    name: str
    normalized_id: str
    era: Era
    birth_year: Optional[int] = None
    death_year: Optional[int] = None
    is_public_domain: bool = False
    languages: Set[str] = field(default_factory=lambda: {"english"})
    total_books: int = 0

    def __post_init__(self):
        """Infer public domain status and era"""
        if self.death_year:
            # Simple heuristic: public domain if died before 1954 (70 years ago)
            self.is_public_domain = self.death_year < 1954

            # Infer era from death year
            if self.death_year < 500:
                self.era = Era.ANCIENT
            elif self.death_year < 1500:
                self.era = Era.CLASSICAL
            elif self.death_year < 1800:
                self.era = Era.EARLY_MODERN
            elif self.death_year < 1928:
                self.era = Era.MODERN
            else:
                self.era = Era.CONTEMPORARY


@dataclass
class SearchStrategy:
    """Strategy for searching sources"""
    source_name: str
    priority: int  # 1=highest, 5=lowest
    confidence: float  # Expected success rate 0.0-1.0
    reason: str  # Why this strategy was chosen


@dataclass
class SearchResult:
    """Result from a source agent search"""
    success: bool
    book: Optional[Book] = None
    source_name: str = ""
    confidence: float = 0.0
    error: Optional[str] = None
    search_time_ms: float = 0.0


@dataclass
class AgentDecision:
    """Autonomous decision made by an agent"""
    agent_id: str
    decision_type: str  # "strategy_selection", "source_search", "download"
    reasoning: str
    confidence: float
    timestamp: float = field(default_factory=time.time)


# ============================================================================
# SHARED KNOWLEDGE BASE
# ============================================================================

class SharedKnowledgeBase:
    """
    Shared knowledge base for all agents to learn from each other.

    Tracks:
    - Successful searches per source
    - Failed searches to avoid repetition
    - Source performance metrics
    - Rate limiting coordination
    """

    def __init__(self):
        self.successful_searches: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)  # source -> [(author, title, id)]
        self.failed_searches: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)  # source -> {(author, title)}
        self.source_stats: Dict[str, Dict[str, any]] = defaultdict(lambda: {
            "total_searches": 0,
            "successful_searches": 0,
            "total_downloads": 0,
            "avg_confidence": 0.0,
            "last_request_time": 0.0
        })
        self.decisions: List[AgentDecision] = []
        self._lock = asyncio.Lock()

    async def record_search(self, source: str, author: str, title: str, success: bool, source_id: Optional[str] = None):
        """Record a search result"""
        async with self._lock:
            self.source_stats[source]["total_searches"] += 1
            if success and source_id:
                self.successful_searches[source].append((author, title, source_id))
                self.source_stats[source]["successful_searches"] += 1
            elif not success:
                self.failed_searches[source].add((author, title))

    async def record_download(self, source: str):
        """Record a successful download"""
        async with self._lock:
            self.source_stats[source]["total_downloads"] += 1

    async def record_decision(self, decision: AgentDecision):
        """Record an agent decision for analysis"""
        async with self._lock:
            self.decisions.append(decision)

    async def update_rate_limit(self, source: str):
        """Update last request time for rate limiting"""
        async with self._lock:
            self.source_stats[source]["last_request_time"] = time.time()

    async def should_rate_limit(self, source: str, min_delay: float = 1.0) -> bool:
        """Check if we should rate limit requests to a source"""
        async with self._lock:
            last_time = self.source_stats[source]["last_request_time"]
            return (time.time() - last_time) < min_delay

    async def get_source_performance(self, source: str) -> Dict:
        """Get performance metrics for a source"""
        async with self._lock:
            return self.source_stats[source].copy()

    async def get_best_source_for_profile(self, profile: AuthorProfile) -> str:
        """Recommend best source based on author profile and past performance"""
        async with self._lock:
            # Simple heuristic based on era and success rates
            if profile.era == Era.ANCIENT:
                return "wikisource"
            elif profile.era in [Era.CLASSICAL, Era.EARLY_MODERN]:
                return "gutenberg" if self.source_stats["gutenberg"]["successful_searches"] > 0 else "wikisource"
            elif profile.era == Era.MODERN and profile.is_public_domain:
                return "gutenberg"
            else:
                return "archive"  # Contemporary works might be in archive

    def get_summary(self) -> Dict:
        """Get summary of all knowledge"""
        total_searches = sum(stats["total_searches"] for stats in self.source_stats.values())
        successful = sum(stats["successful_searches"] for stats in self.source_stats.values())

        return {
            "total_searches": total_searches,
            "successful": successful,
            "failed": total_searches - successful,
            "sources": dict(self.source_stats),
            "total_decisions": len(self.decisions)
        }


# ============================================================================
# SOURCE AGENTS (Specialists)
# ============================================================================

class SourceAgent(ABC):
    """Base class for source-specific agents"""

    def __init__(self, knowledge_base: SharedKnowledgeBase):
        self.knowledge_base = knowledge_base
        self.session = self._create_session()
        self.source_name = self.__class__.__name__.replace("Agent", "").lower()

    def _create_session(self) -> requests.Session:
        """Create a requests session with retry logic"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; AgoraAgenticBot/1.0; +https://github.com/agora)',
            'Accept': 'application/json, text/html',
        })
        retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    @abstractmethod
    async def search(self, author: str, title: str, profile: AuthorProfile) -> SearchResult:
        """Search for a book in this source"""
        pass

    @abstractmethod
    async def download(self, book: Book, output_dir: Path) -> bool:
        """Download a book from this source"""
        pass

    @abstractmethod
    def get_confidence_for_profile(self, profile: AuthorProfile) -> float:
        """Get confidence score for this source given an author profile"""
        pass


class GutenbergAgent(SourceAgent):
    """Expert agent for Project Gutenberg with local catalog cache"""

    GUTENBERG_API = "https://gutendex.com/books"
    GUTENBERG_MIRROR = "https://www.gutenberg.org"

    def __init__(self, knowledge_base: SharedKnowledgeBase):
        super().__init__(knowledge_base)
        # Load entire Gutenberg catalog once (eliminates rate limiting during search!)
        logger.info("Initializing Gutenberg catalog cache...")
        self.catalog = GutenbergCatalog()
        logger.info("Gutenberg catalog ready for instant local searches")

    def get_confidence_for_profile(self, profile: AuthorProfile) -> float:
        """Gutenberg is best for pre-1928 English texts"""
        if not profile.is_public_domain:
            return 0.1
        if profile.era == Era.MODERN:
            return 0.9
        elif profile.era == Era.EARLY_MODERN:
            return 0.85
        elif profile.era == Era.CLASSICAL:
            return 0.7
        else:
            return 0.5

    async def search(self, author: str, title: str, profile: AuthorProfile) -> SearchResult:
        """
        Search Gutenberg using local catalog (instant, no rate limits!).

        This searches the cached CSV catalog locally instead of making API calls.
        Much faster and eliminates all rate limiting issues.
        """
        start_time = time.time()

        try:
            # Search local catalog (runs in thread pool to avoid blocking)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.catalog.search_by_author_and_title,
                author,
                title
            )

            elapsed = (time.time() - start_time) * 1000

            if result:
                # Build URLs for downloading the text
                book_id = result['id']

                # Try multiple URL formats (Gutenberg has inconsistent paths)
                download_urls = [
                    f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",  # UTF-8 version
                    f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",    # ASCII version
                    f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",  # Cache version
                ]

                book = Book(
                    title=title,
                    author=author,
                    author_normalized=profile.normalized_id,
                    source_id=book_id,
                    source_url=download_urls[0],  # Will try all URLs in download()
                    source_name="gutenberg",
                    confidence=result['similarity_score']
                )

                # Store all URL variations for download phase
                book.error = json.dumps({'download_urls': download_urls})  # Temporary storage

                await self.knowledge_base.record_search(
                    self.source_name, author, title, True, book_id
                )

                logger.debug(
                    f"Found '{title}' by {author} in catalog "
                    f"(ID: {book_id}, similarity: {result['similarity_score']:.2f})"
                )

                return SearchResult(
                    success=True,
                    book=book,
                    source_name="gutenberg",
                    confidence=result['similarity_score'],
                    search_time_ms=elapsed
                )
            else:
                await self.knowledge_base.record_search(self.source_name, author, title, False)
                logger.debug(f"Not found in catalog: '{title}' by {author}")
                return SearchResult(success=False, source_name="gutenberg", search_time_ms=elapsed)

        except Exception as e:
            logger.error(f"Gutenberg catalog search error: {e}")
            await self.knowledge_base.record_search(self.source_name, author, title, False)
            return SearchResult(success=False, source_name="gutenberg", error=str(e))

    async def download(self, book: Book, output_dir: Path) -> bool:
        """
        Download text file from Gutenberg.

        Tries multiple URL formats and adds rate limiting to avoid
        overwhelming the server during downloads.
        """
        try:
            # Extract download URLs from temporary storage
            download_urls = []
            if book.error:
                try:
                    data = json.loads(book.error)
                    download_urls = data.get('download_urls', [])
                except:
                    pass

            # Fallback to single URL if no variations stored
            if not download_urls:
                download_urls = [book.source_url]

            # Rate limit downloads (be nice to Gutenberg!)
            while await self.knowledge_base.should_rate_limit(self.source_name, 1.5):
                await asyncio.sleep(0.5)

            author_dir = output_dir / book.author_normalized
            author_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{book.author_normalized}_{book.title[:50]}.txt"
            filename = re.sub(r'[^\w\s-]', '', filename).strip().replace(' ', '_')
            filepath = author_dir / filename

            loop = asyncio.get_event_loop()

            # Try each URL format until one succeeds
            last_error = None
            for url in download_urls:
                try:
                    logger.debug(f"Trying download URL: {url}")

                    response = await loop.run_in_executor(
                        None,
                        lambda u=url: self.session.get(u, timeout=30)
                    )

                    if response.status_code == 200:
                        with open(filepath, 'wb') as f:
                            f.write(response.content)

                        book.downloaded = True
                        book.download_path = str(filepath)
                        book.error = None  # Clear temporary storage

                        await self.knowledge_base.record_download(self.source_name)
                        await self.knowledge_base.update_rate_limit(self.source_name)

                        size_kb = len(response.content) / 1024
                        logger.info(f"Downloaded: {filename} ({size_kb:.1f} KB)")
                        return True

                    logger.debug(f"URL returned {response.status_code}: {url}")

                except Exception as e:
                    logger.debug(f"Failed to download from {url}: {e}")
                    last_error = e
                    continue

            # All URLs failed
            book.error = f"All download URLs failed. Last error: {last_error}"
            logger.warning(f"Failed to download '{book.title}': {book.error}")
            return False

        except Exception as e:
            logger.error(f"Download error for '{book.title}': {e}")
            book.error = str(e)
            return False


class InternetArchiveAgent(SourceAgent):
    """Expert agent for Internet Archive"""

    ARCHIVE_API = "https://archive.org/advancedsearch.php"

    def get_confidence_for_profile(self, profile: AuthorProfile) -> float:
        """Archive has everything but lower quality"""
        return 0.6  # Moderate confidence for all eras

    async def search(self, author: str, title: str, profile: AuthorProfile) -> SearchResult:
        """Search Internet Archive"""
        # Placeholder - would implement similar to Gutenberg
        return SearchResult(success=False, source_name="archive", error="Not implemented")

    async def download(self, book: Book, output_dir: Path) -> bool:
        """Download from Archive"""
        return False


class WikisourceAgent(SourceAgent):
    """Expert agent for Wikisource"""

    WIKISOURCE_API = "https://en.wikisource.org/w/api.php"

    def get_confidence_for_profile(self, profile: AuthorProfile) -> float:
        """Wikisource is best for ancient/classical texts"""
        if profile.era == Era.ANCIENT:
            return 0.9
        elif profile.era == Era.CLASSICAL:
            return 0.85
        else:
            return 0.4

    async def search(self, author: str, title: str, profile: AuthorProfile) -> SearchResult:
        """Search Wikisource"""
        # Placeholder - would implement similar to Gutenberg
        return SearchResult(success=False, source_name="wikisource", error="Not implemented")

    async def download(self, book: Book, output_dir: Path) -> bool:
        """Download from Wikisource"""
        return False


# ============================================================================
# AUTHOR AGENT (Autonomous)
# ============================================================================

class AuthorAgent:
    """
    Autonomous agent representing an author.

    Capabilities:
    - Infers own profile from book data
    - Selects best search strategy
    - Searches multiple sources
    - Makes autonomous decisions
    - Learns from shared knowledge
    """

    def __init__(
        self,
        author_name: str,
        books: List[Book],
        knowledge_base: SharedKnowledgeBase,
        source_agents: Dict[str, SourceAgent]
    ):
        self.author_name = author_name
        self.books = books
        self.knowledge_base = knowledge_base
        self.source_agents = source_agents
        self.profile = self._infer_profile()
        self.agent_id = f"author_agent_{self.profile.normalized_id}"

        logger.info(f"Initialized {self.agent_id}: {len(books)} books, era={self.profile.era.value}, pd={self.profile.is_public_domain}")

    def _infer_profile(self) -> AuthorProfile:
        """Infer author profile from available data"""
        normalized_id = re.sub(r'[^\w\s-]', '', self.author_name).strip().replace(' ', '_').lower()

        # Try to infer from author name (this is a simplified heuristic)
        profile = AuthorProfile(
            name=self.author_name,
            normalized_id=normalized_id,
            era=Era.MODERN,  # Default
            total_books=len(self.books)
        )

        # Simple heuristics based on author name
        ancient_indicators = ["homer", "virgil", "plato", "aristotle"]
        classical_indicators = ["dante", "chaucer", "aquinas"]
        early_modern_indicators = ["shakespeare", "milton", "voltaire"]
        modern_indicators = ["marx", "whitman", "baudelaire", "dickens", "twain"]

        name_lower = self.author_name.lower()

        if any(ind in name_lower for ind in ancient_indicators):
            profile.era = Era.ANCIENT
            profile.is_public_domain = True
        elif any(ind in name_lower for ind in classical_indicators):
            profile.era = Era.CLASSICAL
            profile.is_public_domain = True
        elif any(ind in name_lower for ind in early_modern_indicators):
            profile.era = Era.EARLY_MODERN
            profile.is_public_domain = True
        elif any(ind in name_lower for ind in modern_indicators):
            profile.era = Era.MODERN
            profile.is_public_domain = True

        return profile

    async def select_strategy(self) -> List[SearchStrategy]:
        """
        Autonomously select search strategy based on profile and shared knowledge.

        Returns strategies ordered by priority (highest first).
        """
        strategies = []

        # Get confidence from each source agent
        for source_name, agent in self.source_agents.items():
            confidence = agent.get_confidence_for_profile(self.profile)

            # Get performance stats from knowledge base
            perf = await self.knowledge_base.get_source_performance(source_name)
            success_rate = perf["successful_searches"] / max(perf["total_searches"], 1)

            # Combine agent confidence with observed performance
            adjusted_confidence = (confidence * 0.7) + (success_rate * 0.3)

            strategies.append(SearchStrategy(
                source_name=source_name,
                priority=0,  # Will be set after sorting
                confidence=adjusted_confidence,
                reason=f"Era={self.profile.era.value}, agent_conf={confidence:.2f}, observed_success={success_rate:.2f}"
            ))

        # Sort by confidence and assign priorities
        strategies.sort(key=lambda s: s.confidence, reverse=True)
        for i, strategy in enumerate(strategies, 1):
            strategy.priority = i

        # Record decision
        decision = AgentDecision(
            agent_id=self.agent_id,
            decision_type="strategy_selection",
            reasoning=f"Selected {len(strategies)} strategies based on profile and knowledge",
            confidence=strategies[0].confidence if strategies else 0.0
        )
        await self.knowledge_base.record_decision(decision)

        return strategies

    async def acquire_books(self, output_dir: Path, max_retries: int = 2) -> Dict:
        """
        Autonomously acquire all books for this author.

        Returns summary statistics.
        """
        logger.info(f"[{self.agent_id}] Starting acquisition for {len(self.books)} books")

        strategies = await self.select_strategy()
        logger.info(f"[{self.agent_id}] Selected strategies: {[(s.source_name, s.confidence) for s in strategies]}")

        stats = {
            "total": len(self.books),
            "found": 0,
            "downloaded": 0,
            "failed": 0
        }

        for book in self.books:
            found = False

            # Try each strategy in order of priority
            for strategy in strategies:
                if found:
                    break

                agent = self.source_agents[strategy.source_name]

                logger.debug(f"[{self.agent_id}] Searching {strategy.source_name} for '{book.title}'")
                result = await agent.search(book.author, book.title, self.profile)

                if result.success and result.book:
                    # Download the book
                    success = await agent.download(result.book, output_dir)
                    if success:
                        stats["found"] += 1
                        stats["downloaded"] += 1
                        found = True
                        logger.info(f"[{self.agent_id}] ✓ Found and downloaded '{book.title}' from {strategy.source_name}")
                    else:
                        logger.warning(f"[{self.agent_id}] Found but failed to download '{book.title}' from {strategy.source_name}")

            if not found:
                stats["failed"] += 1
                logger.warning(f"[{self.agent_id}] ✗ Failed to find '{book.title}' in any source")

        logger.info(f"[{self.agent_id}] Completed: {stats['downloaded']}/{stats['total']} books downloaded")
        return stats


# ============================================================================
# AGENTIC ORCHESTRATOR
# ============================================================================

class AgenticOrchestrator:
    """
    Orchestrates multiple author agents running in parallel.

    Manages:
    - Concurrent execution with limits
    - Shared knowledge base
    - Performance monitoring
    - Results aggregation
    """

    def __init__(self, max_concurrent: int = 5):
        self.max_concurrent = max_concurrent
        self.knowledge_base = SharedKnowledgeBase()
        self.source_agents = {
            "gutenberg": GutenbergAgent(self.knowledge_base),
            "archive": InternetArchiveAgent(self.knowledge_base),
            "wikisource": WikisourceAgent(self.knowledge_base)
        }

    async def orchestrate(
        self,
        books_by_author: Dict[str, List[Book]],
        output_dir: Path
    ) -> Dict:
        """
        Orchestrate parallel acquisition across all authors.

        Args:
            books_by_author: Dictionary mapping author names to their books
            output_dir: Directory to save downloaded books

        Returns:
            Summary statistics and insights
        """
        start_time = time.time()

        # Create author agents
        author_agents = []
        for author_name, books in books_by_author.items():
            agent = AuthorAgent(author_name, books, self.knowledge_base, self.source_agents)
            author_agents.append(agent)

        logger.info(f"Created {len(author_agents)} author agents")
        logger.info(f"Max concurrent agents: {self.max_concurrent}")

        # Run agents with concurrency limit
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def run_agent(agent: AuthorAgent):
            async with semaphore:
                return await agent.acquire_books(output_dir)

        # Execute all agents in parallel (with concurrency limit)
        results = await asyncio.gather(*[run_agent(agent) for agent in author_agents])

        elapsed = time.time() - start_time

        # Aggregate results
        summary = {
            "total_agents": len(author_agents),
            "total_books": sum(r["total"] for r in results),
            "books_found": sum(r["found"] for r in results),
            "books_downloaded": sum(r["downloaded"] for r in results),
            "books_failed": sum(r["failed"] for r in results),
            "execution_time_seconds": elapsed,
            "knowledge_base": self.knowledge_base.get_summary()
        }

        return summary


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def normalize_author(author: str) -> str:
    """Normalize author name for directory naming"""
    return re.sub(r'[^\w\s-]', '', author).strip().replace(' ', '_').lower()


def parse_librarything_export(filepath: Path) -> Dict[str, List[Book]]:
    """Parse LibraryThing TSV export"""
    books_by_author = defaultdict(list)

    with open(filepath, 'r', encoding='utf-8') as f:
        # LibraryThing exports are TSV with headers
        reader = csv.DictReader(f, delimiter='\t')

        for row in reader:
            title = row.get('TITLE', '').strip()
            author = row.get('AUTHOR (LAST, FIRST)', '') or row.get('AUTHOR', '')
            author = author.strip()

            if not title or not author:
                continue

            book = Book(
                title=title,
                author=author,
                author_normalized=normalize_author(author),
                isbn=row.get('ISBN', None)
            )

            books_by_author[author].append(book)

    return dict(books_by_author)


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Agentic book acquisition from LibraryThing')
    parser.add_argument('--input', required=True, help='Path to LibraryThing export file (TSV/JSON)')
    parser.add_argument('--output-dir', default='data/raw', help='Output directory for downloads')
    parser.add_argument('--max-concurrent', type=int, default=5, help='Max concurrent author agents')
    parser.add_argument('--author-filter', help='Comma-separated list of authors to process')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading books from {input_path}")
    books_by_author = parse_librarything_export(input_path)

    # Apply author filter if specified
    if args.author_filter:
        filter_authors = set(a.strip() for a in args.author_filter.split(','))
        books_by_author = {
            author: books for author, books in books_by_author.items()
            if any(fa.lower() in author.lower() for fa in filter_authors)
        }

    logger.info(f"Processing {len(books_by_author)} authors, {sum(len(b) for b in books_by_author.values())} total books")

    # Run orchestrator
    orchestrator = AgenticOrchestrator(max_concurrent=args.max_concurrent)
    summary = await orchestrator.orchestrate(books_by_author, output_dir)

    # Print results
    print("\n" + "="*60)
    print("AGENTIC ACQUISITION SUMMARY")
    print("="*60)
    print(f"Authors processed: {summary['total_agents']}")
    print(f"Total books: {summary['total_books']}")
    print(f"Books found: {summary['books_found']}")
    print(f"Books downloaded: {summary['books_downloaded']}")
    print(f"Books failed: {summary['books_failed']}")
    print(f"Success rate: {summary['books_downloaded']/max(summary['total_books'],1)*100:.1f}%")
    print(f"Execution time: {summary['execution_time_seconds']:.3f}s")
    print()
    print("KNOWLEDGE BASE INSIGHTS")
    print("="*60)
    kb = summary['knowledge_base']
    print(f"Total searches: {kb['total_searches']}")
    print(f"Successful: {kb['successful']}")
    print(f"Failed: {kb['failed']}")
    print()
    print("Source Performance:")
    for source, stats in kb['sources'].items():
        if stats['total_searches'] > 0:
            success_rate = stats['successful_searches'] / stats['total_searches'] * 100
            print(f"  {source}: {stats['successful_searches']}/{stats['total_searches']} searches ({success_rate:.1f}%), {stats['total_downloads']} downloads")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
