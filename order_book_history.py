"""
Order Book History Collector
----------------------------
Методы для скачивания исторических снимков стакана через Binance REST API.

Может использоваться двумя способами:
  1. Как набор методов для вставки в CLOBDataSource (clob.py)
  2. Как самостоятельный скрипт: python order_book_history.py

Binance REST API endpoint:
  GET https://api.binance.com/api/v3/depth
  Параметры: symbol, limit (5 / 10 / 20 / 50 / 100 / 500 / 1000 / 5000)

Веса запросов (Request Weight):
  limit <= 100   →  weight 1
  limit <= 500   →  weight 5
  limit <= 1000  →  weight 10
  limit <= 5000  →  weight 50

Лимит без API-ключа: 1200 weight/min (т.е. 1200 запросов/мин при limit <= 100)
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp
import pandas as pd

logger = logging.getLogger(__name__)

BINANCE_DEPTH_URL = "https://api.binance.com/api/v3/depth"

# Вес запроса в зависимости от глубины стакана
DEPTH_WEIGHT = {5: 1, 10: 1, 20: 1, 50: 1, 100: 1, 500: 5, 1000: 10, 5000: 50}
VALID_DEPTHS = sorted(DEPTH_WEIGHT.keys())


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _nearest_valid_depth(depth: int) -> int:
    """Округляет глубину до ближайшего допустимого значения Binance API."""
    for d in VALID_DEPTHS:
        if depth <= d:
            return d
    return VALID_DEPTHS[-1]


def _snapshots_to_dataframe(snapshots: List[Dict]) -> pd.DataFrame:
    """
    Конвертирует список снимков стакана в плоский DataFrame.

    Каждая строка — один уровень стакана в один момент времени.
    Колонки: ts, side, price, amount, level
    """
    rows = []
    for snap in snapshots:
        ts = snap["ts"]
        for level, (price, amount) in enumerate(snap["bids"]):
            rows.append({"ts": ts, "side": "bid", "price": price, "amount": amount, "level": level})
        for level, (price, amount) in enumerate(snap["asks"]):
            rows.append({"ts": ts, "side": "ask", "price": price, "amount": amount, "level": level})

    if not rows:
        return pd.DataFrame(columns=["ts", "side", "price", "amount", "level"])

    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df["price"] = df["price"].astype(float)
    df["amount"] = df["amount"].astype(float)
    return df


def _snapshots_to_obi_series(snapshots: List[Dict], levels: int = 5) -> pd.DataFrame:
    """
    Вычисляет Order Book Imbalance (OBI) для каждого снимка.

    OBI = (bid_vol - ask_vol) / (bid_vol + ask_vol)  ∈ [-1, 1]
    Положительное значение — давление покупателей, отрицательное — продавцов.

    Параметры
    ---------
    snapshots : список снимков из collect_order_book_history
    levels    : сколько уровней учитывать (от лучшей цены)

    Возвращает DataFrame с колонками: ts, obi, bid_vol, ask_vol, mid_price, spread
    """
    rows = []
    for snap in snapshots:
        bids = snap["bids"][:levels]
        asks = snap["asks"][:levels]
        if not bids or not asks:
            continue

        bid_vol = sum(float(b[1]) for b in bids)
        ask_vol = sum(float(a[1]) for a in asks)
        total = bid_vol + ask_vol

        obi = (bid_vol - ask_vol) / total if total > 0 else 0.0
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid

        rows.append({
            "ts": snap["ts"],
            "obi": obi,
            "bid_vol": bid_vol,
            "ask_vol": ask_vol,
            "mid_price": mid_price,
            "spread": spread,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    return df


# ---------------------------------------------------------------------------
# Основная функция сбора — вставляется в CLOBDataSource как метод
# ---------------------------------------------------------------------------

async def collect_order_book_history(
    trading_pair: str,
    duration_seconds: int = 3600,
    interval_seconds: float = 1.0,
    depth: int = 20,
    save_path: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[Dict]:
    """
    Собирает серию снимков стакана с Binance через регулярные опросы REST API.

    Поскольку Binance не предоставляет исторические снимки стакана напрямую,
    эта функция делает регулярные запросы к /api/v3/depth и накапливает
    хронологическую серию снимков.

    Параметры
    ---------
    trading_pair      : пара в формате Hummingbot ('BTC-USDT') или Binance ('BTCUSDT')
    duration_seconds  : сколько секунд собирать данные (по умолчанию 1 час)
    interval_seconds  : интервал между снимками в секундах (по умолчанию 1 сек)
    depth             : глубина стакана — допустимые значения: 5,10,20,50,100,500,1000,5000
    save_path         : путь для сохранения результата в Parquet (None = не сохранять)
    api_key           : Binance API ключ (необязателен, но снимает строгие лимиты)

    Возвращает
    ----------
    Список словарей вида:
        {"ts": float, "bids": [[price, amount], ...], "asks": [[price, amount], ...]}

    Пример использования
    --------------------
    # В CLOBDataSource:
    snapshots = await self.collect_order_book_history(
        trading_pair="BTC-USDT",
        duration_seconds=600,
        interval_seconds=1.0,
        depth=20,
        save_path="data/btc_ob_history.parquet",
    )
    obi_df = self._snapshots_to_obi_series(snapshots, levels=5)

    # Standalone:
    import asyncio
    snapshots = asyncio.run(collect_order_book_history("BTC-USDT", duration_seconds=60))
    """
    symbol = trading_pair.replace("-", "")
    depth = _nearest_valid_depth(depth)
    weight_per_request = DEPTH_WEIGHT[depth]

    # Лимит weight: 1200/мин без ключа, 6000/мин с ключом
    weight_limit_per_min = 6000 if api_key else 1200
    # Минимальный безопасный интервал с небольшим запасом 20%
    min_safe_interval = (weight_per_request / weight_limit_per_min) * 60 * 1.2

    if interval_seconds < min_safe_interval:
        logger.warning(
            f"interval_seconds={interval_seconds:.3f}s может превысить лимиты Binance "
            f"(weight={weight_per_request}, min безопасный интервал={min_safe_interval:.3f}s). "
            f"Автоматически увеличиваем до {min_safe_interval:.3f}s."
        )
        interval_seconds = min_safe_interval

    headers = {"X-MBX-APIKEY": api_key} if api_key else {}
    params = {"symbol": symbol, "limit": depth}

    snapshots: List[Dict] = []
    total_requests = int(duration_seconds / interval_seconds)
    start_ts = time.time()

    logger.info(
        f"Сбор стакана {trading_pair} | глубина={depth} | "
        f"интервал={interval_seconds:.2f}s | длительность={duration_seconds}s | "
        f"ожидаемых снимков≈{total_requests}"
    )

    async with aiohttp.ClientSession(headers=headers) as session:
        for i in range(total_requests):
            tick_start = time.monotonic()
            try:
                async with session.get(BINANCE_DEPTH_URL, params=params, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 429:
                        retry_after = int(resp.headers.get("Retry-After", 60))
                        logger.warning(f"Rate limit (429), ждём {retry_after}s")
                        await asyncio.sleep(retry_after)
                        continue
                    if resp.status == 418:
                        logger.error("IP забанен Binance (418). Остановка сбора.")
                        break
                    resp.raise_for_status()
                    data = await resp.json()

                snapshot = {
                    "ts": time.time(),
                    "bids": [[float(p), float(q)] for p, q in data["bids"]],
                    "asks": [[float(p), float(q)] for p, q in data["asks"]],
                }
                snapshots.append(snapshot)

                if (i + 1) % 60 == 0:
                    elapsed = time.time() - start_ts
                    logger.info(f"  [{i+1}/{total_requests}] собрано снимков: {len(snapshots)}, прошло: {elapsed:.0f}s")

            except asyncio.TimeoutError:
                logger.warning(f"Снимок {i+1}: таймаут запроса, пропускаем")
            except Exception as e:
                logger.error(f"Снимок {i+1}: ошибка — {type(e).__name__}: {e}")

            # Точный sleep с учётом времени запроса
            elapsed_tick = time.monotonic() - tick_start
            sleep_time = max(0.0, interval_seconds - elapsed_tick)
            await asyncio.sleep(sleep_time)

    logger.info(f"Сбор завершён. Итого снимков: {len(snapshots)}")

    if save_path and snapshots:
        _save_snapshots(snapshots, save_path, trading_pair)

    return snapshots


# ---------------------------------------------------------------------------
# Загрузка одного снимка (для разовых запросов)
# ---------------------------------------------------------------------------

async def get_binance_order_book_snapshot(
    trading_pair: str,
    depth: int = 20,
    api_key: Optional[str] = None,
) -> Dict:
    """
    Делает один запрос к Binance /api/v3/depth и возвращает снимок стакана.

    Параметры
    ---------
    trading_pair : пара ('BTC-USDT' или 'BTCUSDT')
    depth        : глубина (5,10,20,50,100,500,1000,5000)
    api_key      : опциональный Binance API ключ

    Возвращает
    ----------
    {"ts": float, "bids": [[price, amount], ...], "asks": [[price, amount], ...]}
    """
    symbol = trading_pair.replace("-", "")
    depth = _nearest_valid_depth(depth)
    headers = {"X-MBX-APIKEY": api_key} if api_key else {}
    params = {"symbol": symbol, "limit": depth}

    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(BINANCE_DEPTH_URL, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            resp.raise_for_status()
            data = await resp.json()

    return {
        "ts": time.time(),
        "bids": [[float(p), float(q)] for p, q in data["bids"]],
        "asks": [[float(p), float(q)] for p, q in data["asks"]],
    }


# ---------------------------------------------------------------------------
# Сохранение и загрузка
# ---------------------------------------------------------------------------

def _save_snapshots(snapshots: List[Dict], save_path: str, trading_pair: str) -> None:
    """Сохраняет снимки в Parquet (плоский формат) и рядом кладёт OBI-серию."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = _snapshots_to_dataframe(snapshots)
    df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
    logger.info(f"Снимки сохранены: {path} ({len(df)} строк)")

    obi_path = path.with_name(path.stem + "_obi.parquet")
    obi_df = _snapshots_to_obi_series(snapshots)
    obi_df.to_parquet(obi_path, engine="pyarrow", compression="snappy", index=False)
    logger.info(f"OBI-серия сохранена: {obi_path} ({len(obi_df)} строк)")


def load_order_book_history(
    file_path: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Загружает ранее сохранённые снимки стакана из Parquet-файлов.

    Параметры
    ---------
    file_path  : путь к основному файлу (например, 'data/btc_ob_history.parquet')
    start_time : фильтр начала периода (UTC datetime, опционально)
    end_time   : фильтр конца периода (UTC datetime, опционально)

    Возвращает
    ----------
    (snapshots_df, obi_df) — плоский DataFrame снимков и DataFrame с OBI-метриками
    """
    path = Path(file_path)
    obi_path = path.with_name(path.stem + "_obi.parquet")

    snapshots_df = pd.read_parquet(path)
    obi_df = pd.read_parquet(obi_path) if obi_path.exists() else pd.DataFrame()

    if start_time:
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        snapshots_df = snapshots_df[snapshots_df["ts"] >= start_time]
        if not obi_df.empty:
            obi_df = obi_df[obi_df["ts"] >= start_time]

    if end_time:
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)
        snapshots_df = snapshots_df[snapshots_df["ts"] <= end_time]
        if not obi_df.empty:
            obi_df = obi_df[obi_df["ts"] <= end_time]

    return snapshots_df, obi_df


# ---------------------------------------------------------------------------
# Standalone запуск
# ---------------------------------------------------------------------------

async def _main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Параметры сбора — меняйте под свои нужды
    TRADING_PAIR = "BTC-USDT"
    DURATION_SECONDS = 300   # 5 минут для демонстрации
    INTERVAL_SECONDS = 1.0   # снимок каждую секунду
    DEPTH = 20
    SAVE_PATH = f"data/order_book/{TRADING_PAIR.replace('-', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    API_KEY = None  # вставьте ключ или оставьте None

    snapshots = await collect_order_book_history(
        trading_pair=TRADING_PAIR,
        duration_seconds=DURATION_SECONDS,
        interval_seconds=INTERVAL_SECONDS,
        depth=DEPTH,
        save_path=SAVE_PATH,
        api_key=API_KEY,
    )

    if snapshots:
        obi_df = _snapshots_to_obi_series(snapshots, levels=5)
        print("\n=== Первые 5 строк OBI ===")
        print(obi_df.head())
        print(f"\nСредний OBI: {obi_df['obi'].mean():.4f}")
        print(f"Средний спред: {obi_df['spread'].mean():.2f} USDT")
        print(f"\nФайлы сохранены в: {SAVE_PATH}")


if __name__ == "__main__":
    asyncio.run(_main())
