import babeldoc.translator.translation_memory as tm
from babeldoc.format.pdf.document_il.midend.il_translator import LLMTranslateTracker
from babeldoc.translator.translation_memory import TranslationMemory
from babeldoc.translator.translation_memory import _TranslationMemoryRecord


def _cleanup_engine_rows(engine: str, lang_in: str, lang_out: str) -> None:
    _TranslationMemoryRecord.delete().where(
        (_TranslationMemoryRecord.translate_engine == engine)
        & (_TranslationMemoryRecord.lang_in == lang_in)
        & (_TranslationMemoryRecord.lang_out == lang_out)
    ).execute()


def test_store_and_get_similar_hints():
    engine = "tmtest"
    lang_in = "en"
    lang_out = "tr"
    memory = TranslationMemory(engine, lang_in, lang_out)
    _cleanup_engine_rows(engine, lang_in, lang_out)
    try:
        memory.store(
            "Transformer model improves translation quality in scientific documents.",
            "Transformer modeli bilimsel belgelerde ceviri kalitesini artirir.",
        )
        memory.store(
            "Neural translation model keeps terminology consistent.",
            "Sinirsel ceviri modeli terminolojiyi tutarli tutar.",
        )
        memory.store(
            "Image preprocessing can speed up OCR pipelines.",
            "Gorsel on isleme OCR boru hatlarini hizlandirabilir.",
        )

        hints = memory.get_similar_hints(
            "This transformer translation model should keep terminology consistent.",
            max_candidates=2,
            lookup_rows=100,
            min_shared_terms=2,
        )

        assert hints
        assert len(hints) <= 2
        assert "source_text" in hints[0]
        assert "translated_text" in hints[0]
        assert "shared_terms" in hints[0]
    finally:
        _cleanup_engine_rows(engine, lang_in, lang_out)


def test_store_skips_empty_and_same_text():
    engine = "tmtest_empty"
    lang_in = "en"
    lang_out = "tr"
    memory = TranslationMemory(engine, lang_in, lang_out)
    _cleanup_engine_rows(engine, lang_in, lang_out)
    try:
        memory.store("", "bos olmamali")
        memory.store("same text", "same text")

        row_count = (
            _TranslationMemoryRecord.select()
            .where(
                (_TranslationMemoryRecord.translate_engine == engine)
                & (_TranslationMemoryRecord.lang_in == lang_in)
                & (_TranslationMemoryRecord.lang_out == lang_out)
            )
            .count()
        )
        assert row_count == 0
    finally:
        _cleanup_engine_rows(engine, lang_in, lang_out)


def test_ngram_and_lemmatization_matching():
    engine = "tmtest_ngram"
    lang_in = "en"
    lang_out = "tr"
    memory = TranslationMemory(engine, lang_in, lang_out)
    _cleanup_engine_rows(engine, lang_in, lang_out)
    try:
        memory.store(
            "Neural networks are running robust experiments for translation memory.",
            "Sinir aglari ceviri hafizasi icin saglam deneyler yurutuyor.",
        )
        hints = memory.get_similar_hints(
            "A neural network run should improve translation memory quality.",
            max_candidates=3,
            lookup_rows=100,
            min_shared_terms=2,
        )
        assert hints
        assert any("neural network" in " ".join(h["shared_terms"]) for h in hints)
    finally:
        _cleanup_engine_rows(engine, lang_in, lang_out)


def test_cleanup_policy(monkeypatch):
    engine = "tmtest_cleanup"
    lang_in = "en"
    lang_out = "tr"
    memory = TranslationMemory(engine, lang_in, lang_out)
    _cleanup_engine_rows(engine, lang_in, lang_out)
    monkeypatch.setattr(tm, "CLEAN_PROBABILITY", 1.0)
    memory.cleanup_probability = 1.0
    memory.max_rows = 5
    try:
        for i in range(30):
            memory.store(f"source term set {i}", f"hedef terim {i}")

        row_count = (
            _TranslationMemoryRecord.select()
            .where(
                (_TranslationMemoryRecord.translate_engine == engine)
                & (_TranslationMemoryRecord.lang_in == lang_in)
                & (_TranslationMemoryRecord.lang_out == lang_out)
            )
            .count()
        )
        assert row_count <= 5
    finally:
        _cleanup_engine_rows(engine, lang_in, lang_out)


def test_llm_tracker_keeps_memory_hints():
    tracker = LLMTranslateTracker()
    hints = [
        {
            "source_text": "A",
            "translated_text": "B",
            "shared_terms": ["term"],
        }
    ]
    tracker.set_translation_memory_hints(hints)
    data = tracker.to_dict()
    assert "translation_memory_hints" in data
    assert data["translation_memory_hints"] == hints
