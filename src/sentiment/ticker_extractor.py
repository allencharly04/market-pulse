"""
Ticker extraction for Agent 29.

Identifies stock tickers and crypto symbols mentioned in news headlines
using three matching strategies:

1. Cashtag matching ($AAPL, $NVDA, $BTC) — highest precision
2. Name alias matching (Nvidia → NVDA) — high recall via curated registry
3. Crypto bare-symbol matching (Bitcoin → BTC, ETH → ETH)

Returns a list of matched (ticker, asset_class) tuples per headline.
Empty list if no matches found.

Filters: ambiguous single-letter tickers and overloaded acronyms (AI, GO, ON)
are listed in tickers.yaml and only matched via explicit cashtag.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

REGISTRY_PATH = Path("config/tickers.yaml")


class TickerExtractor:
    """Hybrid cashtag + alias + symbol ticker extractor."""

    name = "ticker_extractor"

    def __init__(self, registry_path: Path = REGISTRY_PATH):
        with open(registry_path) as f:
            data = yaml.safe_load(f)

        self.stocks: dict[str, list[str]] = data.get("stocks", {})
        self.crypto: dict[str, list[str]] = data.get("crypto", {})
        self.ambiguous = set(data.get("ambiguous", []))
        self.crypto_ambiguous = set(data.get("crypto_ambiguous", []))

        # Build search structures
        self._build_alias_index()
        self._build_regex_patterns()

        n_stock = len(self.stocks)
        n_crypto = len(self.crypto)
        n_alias = sum(len(v) for v in {**self.stocks, **self.crypto}.values())
        logger.info(
            f"[{self.name}] loaded {n_stock} stocks + {n_crypto} crypto, "
            f"{n_alias} aliases total"
        )

    def _build_alias_index(self):
        """Map lowercase alias → (ticker, asset_class)."""
        self.alias_to_ticker: dict[str, tuple[str, str]] = {}
        for ticker, aliases in self.stocks.items():
            for alias in aliases:
                self.alias_to_ticker[alias.lower()] = (ticker, "stock")
        for ticker, aliases in self.crypto.items():
            for alias in aliases:
                self.alias_to_ticker[alias.lower()] = (ticker, "crypto")

    def _build_regex_patterns(self):
        """Compile regex patterns for fast matching."""
        # Cashtag: $TICKER (1-5 uppercase letters, optionally with .X suffix for class shares)
        self.cashtag_re = re.compile(r"\$([A-Z]{1,5}(?:\.[A-Z])?)\b")

        # Aliases (longer first to avoid partial matches like "Apple" inside "Apple Inc")
        sorted_aliases = sorted(self.alias_to_ticker.keys(), key=len, reverse=True)
        # Escape special chars + word boundaries
        alias_pattern = r"\b(" + "|".join(re.escape(a) for a in sorted_aliases) + r")\b"
        self.alias_re = re.compile(alias_pattern, re.IGNORECASE)

        # Bare crypto symbols (BTC, ETH, etc) — only the keys, only uppercase, word-bounded
        # Skip ambiguous ones from crypto_ambiguous
        crypto_keys = [
            t for t in self.crypto.keys()
            if t not in self.crypto_ambiguous and len(t) >= 3
        ]
        if crypto_keys:
            crypto_sym_pattern = r"\b(" + "|".join(re.escape(t) for t in crypto_keys) + r")\b"
            self.crypto_sym_re = re.compile(crypto_sym_pattern)
        else:
            self.crypto_sym_re = None

    def extract(self, text: str) -> list[tuple[str, str]]:
        """
        Extract tickers from a single text.

        Returns list of (ticker, asset_class) tuples, deduplicated.
        asset_class is "stock" or "crypto".
        """
        if not text or not isinstance(text, str):
            return []

        found: dict[str, str] = {}  # ticker → asset_class (preserves dedup)

        # 1. Cashtag matches: highest priority, override anything
        for m in self.cashtag_re.findall(text):
            t = m.upper()
            if t in self.stocks:
                found[t] = "stock"
            elif t in self.crypto:
                found[t] = "crypto"
            else:
                # Unknown cashtag — assume stock (most common case)
                # Skip ambiguous list
                if t not in self.ambiguous:
                    found[t] = "stock_unknown"

        # 2. Alias matches: company names → tickers
        for m in self.alias_re.findall(text):
            ticker, asset_class = self.alias_to_ticker[m.lower()]
            if ticker not in found:
                found[ticker] = asset_class

        # 3. Bare crypto symbol matches (BTC, ETH appearing standalone)
        if self.crypto_sym_re is not None:
            for m in self.crypto_sym_re.findall(text):
                if m not in found:
                    found[m] = "crypto"

        return [(t, c) for t, c in found.items()]

    def extract_batch(self, texts: list[str]) -> list[list[tuple[str, str]]]:
        """Vectorized over a list of texts."""
        return [self.extract(t) for t in texts]

    def tag_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "title",
        ticker_col: str = "tickers",
        primary_col: str = "primary_ticker",
    ) -> pd.DataFrame:
        """
        Add ticker columns to a DataFrame.

        - {ticker_col}: comma-separated string of all matched tickers
        - {primary_col}: the first matched ticker (for single-ticker analysis)
        - {ticker_col}_count: number of matches
        - {ticker_col}_classes: comma-separated asset classes
        """
        if df.empty or text_col not in df.columns:
            return df

        results = self.extract_batch(df[text_col].fillna("").tolist())

        df[ticker_col] = [
            ",".join(t for t, _ in r) if r else None for r in results
        ]
        df[primary_col] = [r[0][0] if r else None for r in results]
        df[f"{ticker_col}_count"] = [len(r) for r in results]
        df[f"{ticker_col}_classes"] = [
            ",".join(c for _, c in r) if r else None for r in results
        ]
        return df


if __name__ == "__main__":
    ext = TickerExtractor()

    # Test cases — each should resolve to specific tickers
    test_cases = [
        ("Apple's record iPhone sales beat expectations", ["AAPL"]),
        ("Nvidia $NVDA stock surges on AI chip demand", ["NVDA"]),
        ("Tesla and Ford both report earnings tomorrow", ["TSLA", "F"]),
        ("Bitcoin breaks above $100,000 as Ethereum rallies", ["BTC", "ETH"]),
        ("$AAPL up 3%, $MSFT up 2% — tech rally continues", ["AAPL", "MSFT"]),
        ("Coinbase posts $400M loss as crypto winter deepens", ["COIN"]),
        ("BTC, ETH, SOL all hit new highs", ["BTC", "ETH", "SOL"]),
        ("Toyota cuts profit forecast as Iran war weighs", ["TM"]),
        ("Cloudflare sinks 18% on layoffs, $NET tumbles after hours", ["NET"]),
        ("Markets close mixed", []),  # nothing
        ("I have to write a report", []),  # the "I" should be filtered
        ("Pope Leo rejects Trump attack", []),  # no tickers
        ("Berkshire Hathaway annual meeting", ["BRK.B"]),
        ("Nvidia and AMD both report; chips sector hot", ["NVDA", "AMD"]),
    ]

    print(f"\n--- Ticker extraction tests ({len(test_cases)} cases) ---\n")
    correct = 0
    for text, expected in test_cases:
        result = ext.extract(text)
        result_tickers = [t for t, _ in result]
        match = set(result_tickers) == set(expected)
        marker = "✓" if match else "✗"
        if match:
            correct += 1
        print(f"  {marker} {text[:75]}")
        print(f"    Expected: {expected}")
        print(f"    Got:      {result_tickers}")
        if not match:
            print(f"    ⚠️  MISMATCH")

    print(f"\n  Score: {correct}/{len(test_cases)}")

    # Test on actual data from the database
    print("\n--- Tagging real headlines from database ---")
    import sqlite3
    conn = sqlite3.connect("data/agent29.db")
    df = pd.read_sql_query(
        "SELECT title, source, finbert_label FROM news ORDER BY id DESC LIMIT 20",
        conn,
    )
    conn.close()

    df = ext.tag_dataframe(df)

    print(f"\n  Tagged {len(df)} recent headlines:")
    print(f"  {df['tickers_count'].sum()} ticker mentions across {(df['primary_ticker'].notna()).sum()} headlines")
    print()

    # Show top examples
    tagged = df[df["primary_ticker"].notna()].copy()
    if not tagged.empty:
        print("  Examples (most-recent tagged headlines):")
        for _, row in tagged.head(10).iterrows():
            print(f"    [{row['primary_ticker']:6s}] [{row['finbert_label']:>8s}] {row['title'][:75]}")

    # Frequency of tickers in last 20 headlines
    if df["tickers"].notna().any():
        all_tickers = []
        for t_str in df["tickers"].dropna():
            all_tickers.extend(t_str.split(","))
        from collections import Counter
        top = Counter(all_tickers).most_common(10)
        if top:
            print("\n  Top tickers in recent headlines:")
            for t, n in top:
                print(f"    {t:6s} {n}")
