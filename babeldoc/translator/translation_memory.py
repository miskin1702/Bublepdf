import json
import logging
import random
import re
import threading
import time
from pathlib import Path

import peewee
from peewee import AutoField
from peewee import CharField
from peewee import IntegerField
from peewee import Model
from peewee import SqliteDatabase
from peewee import TextField

from babeldoc.const import CACHE_FOLDER

logger = logging.getLogger(__name__)

memory_db = SqliteDatabase(None)
_init_lock = threading.Lock()
_cleanup_lock = threading.Lock()
_memory_token_pattern = re.compile(r"\b[^\W\d_][\w-]{2,}\b", flags=re.UNICODE)

CLEAN_PROBABILITY = 0.001
MAX_MEMORY_ROWS = 100_000


class _TranslationMemoryRecord(Model):
    id = AutoField()
    translate_engine = CharField(max_length=20)
    lang_in = CharField(max_length=16)
    lang_out = CharField(max_length=16)
    source_text = TextField()
    translated_text = TextField()
    source_terms_json = TextField(default="[]")
    created_at = IntegerField(default=lambda: int(time.time()))

    class Meta:
        database = memory_db


class TranslationMemory:
    def __init__(
        self,
        translate_engine: str,
        lang_in: str,
        lang_out: str,
        db_path: str | Path | None = None,
        max_rows: int = MAX_MEMORY_ROWS,
        cleanup_probability: float = CLEAN_PROBABILITY,
    ):
        self.translate_engine = translate_engine
        self.lang_in = lang_in
        self.lang_out = lang_out
        self.max_rows = max(1_000, int(max_rows))
        self.cleanup_probability = min(max(cleanup_probability, 0.0), 1.0)
        self._init_db(db_path)

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = (text or "").strip().lower()
        return re.sub(r"\s+", " ", text)

    @classmethod
    def _lemmatize_token(cls, token: str) -> str:
        # Lightweight rule-based lemmatization to avoid extra dependencies.
        if len(token) <= 4:
            return token
        if token.endswith("ies") and len(token) > 5:
            return token[:-3] + "y"
        if token.endswith("ing") and len(token) > 6:
            return token[:-3]
        if token.endswith("ed") and len(token) > 5:
            return token[:-2]
        if token.endswith("es") and len(token) > 5:
            return token[:-2]
        if token.endswith("s") and len(token) > 4:
            return token[:-1]
        return token

    @classmethod
    def _extract_terms(cls, text: str) -> list[str]:
        normalized = cls._normalize_text(text)
        tokens = []
        for match in _memory_token_pattern.finditer(normalized):
            term = match.group(0)
            if len(term) >= 3:
                tokens.append(term)

        base_tokens = [cls._lemmatize_token(tok) for tok in tokens]
        terms = set(base_tokens)

        # Add n-gram features to improve phrase-level similarity.
        for n in (2, 3):
            for i in range(len(base_tokens) - n + 1):
                terms.add(" ".join(base_tokens[i : i + n]))

        return sorted(terms)[:400]

    def _init_db(self, db_path: str | Path | None) -> None:
        with _init_lock:
            if db_path is None:
                CACHE_FOLDER.mkdir(parents=True, exist_ok=True)
                db_path = CACHE_FOLDER / "translation_memory.v1.db"
            db_path = Path(db_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)

            if memory_db.is_closed() and memory_db.database is None:
                memory_db.init(
                    db_path,
                    pragmas={
                        "journal_mode": "wal",
                        "busy_timeout": 1000,
                    },
                )
                memory_db.create_tables([_TranslationMemoryRecord], safe=True)
            elif memory_db.database != str(db_path):
                logger.warning(
                    "Translation memory db already initialized with %s, ignoring new path %s.",
                    memory_db.database,
                    db_path,
                )

    def store(self, source_text: str, translated_text: str) -> None:
        source_text = (source_text or "").strip()
        translated_text = (translated_text or "").strip()
        if not source_text or not translated_text:
            return
        if source_text == translated_text:
            return

        terms = self._extract_terms(source_text)
        try:
            _TranslationMemoryRecord.create(
                translate_engine=self.translate_engine,
                lang_in=self.lang_in,
                lang_out=self.lang_out,
                source_text=source_text,
                translated_text=translated_text,
                source_terms_json=json.dumps(terms, ensure_ascii=False),
            )
            if random.random() < self.cleanup_probability:  # noqa: S311
                self._cleanup()
        except peewee.OperationalError as e:
            if "database is locked" in str(e).lower():
                logger.debug("Translation memory database is locked, skip store.")
                return
            raise

    def _cleanup(self) -> None:
        if not _cleanup_lock.acquire(blocking=False):
            return
        try:
            max_id = (
                _TranslationMemoryRecord.select(peewee.fn.MAX(_TranslationMemoryRecord.id))
                .where(
                    (_TranslationMemoryRecord.translate_engine == self.translate_engine)
                    & (_TranslationMemoryRecord.lang_in == self.lang_in)
                    & (_TranslationMemoryRecord.lang_out == self.lang_out)
                )
                .scalar()
            )
            if not max_id or max_id <= self.max_rows:
                return
            threshold = max_id - self.max_rows
            _TranslationMemoryRecord.delete().where(
                (_TranslationMemoryRecord.translate_engine == self.translate_engine)
                & (_TranslationMemoryRecord.lang_in == self.lang_in)
                & (_TranslationMemoryRecord.lang_out == self.lang_out)
                & (_TranslationMemoryRecord.id <= threshold)
            ).execute()
        finally:
            _cleanup_lock.release()

    def get_similar_hints(
        self,
        source_text: str,
        max_candidates: int = 3,
        lookup_rows: int = 2000,
        min_shared_terms: int = 1,
    ) -> list[dict]:
        source_text = (source_text or "").strip()
        if not source_text:
            return []
        query_terms = set(self._extract_terms(source_text))
        if not query_terms:
            return []

        hints = []
        try:
            rows = (
                _TranslationMemoryRecord.select()
                .where(
                    (_TranslationMemoryRecord.translate_engine == self.translate_engine)
                    & (_TranslationMemoryRecord.lang_in == self.lang_in)
                    & (_TranslationMemoryRecord.lang_out == self.lang_out)
                )
                .order_by(_TranslationMemoryRecord.id.desc())
                .limit(lookup_rows)
            )
            for row in rows:
                if row.source_text == source_text:
                    continue
                try:
                    row_terms = set(json.loads(row.source_terms_json or "[]"))
                except Exception:
                    row_terms = set(self._extract_terms(row.source_text))
                shared_terms = sorted(query_terms.intersection(row_terms))
                if len(shared_terms) < min_shared_terms:
                    continue
                score = sum(2 if " " in term else 1 for term in shared_terms)
                hints.append(
                    {
                        "source_text": row.source_text,
                        "translated_text": row.translated_text,
                        "shared_terms": shared_terms[:8],
                        "score": score,
                    }
                )
        except peewee.OperationalError as e:
            if "database is locked" in str(e).lower():
                logger.debug("Translation memory database is locked, skip lookup.")
                return []
            raise

        hints.sort(key=lambda x: x["score"], reverse=True)
        return hints[:max_candidates]
