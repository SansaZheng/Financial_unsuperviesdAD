"""Convert the Xu & Livshits (USENIX Security 2019) pump-and-dump release into OHLCVA K-lines with labels.

Input (downloaded verbatim into data/raw/xu_livshits_pumpdump/, pinned upstream commit):
  cryptocompare.RData   CryptoCompare ``histohour`` candles per coin and exchange:
                        time, open, high, low, close, volumefrom (base volume), volumeto (quote amount)
  rawpumpolymp.json     PumpOlymp pump signals (coin, exchange, Telegram channel, signal time, ...)
  reg_cryptopia.RData   authors' coin-hour Cryptopia panel with a ``pumped`` flag (regression sample)

Output (local, not committed; the raw files are the canonical copy):
  klines_1h.csv.gz  coin, exchange, quote, timestamp(UTC), open, high, low, close, volume, amount,
                    pump (1 in the hour containing a PumpOlymp signal for that coin on that exchange)
  events.csv        one row per signal with the matching status against the candles
  summary.json      counts and coverage

R objects are parsed as data with the ``rdata`` package (no R code is executed). Matrix-valued
data-frame columns are split into numbered columns. CryptoCompare quotes these candles in BTC.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT/'data'/'raw'/'xu_livshits_pumpdump'
COLUMNS = ['coin', 'exchange', 'quote', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pump']


def read_rdata(path):
    import rdata

    def data_frame(obj, attrs):
        columns = {}
        for name, value in obj.items():
            array = np.asarray(value)
            if array.ndim == 2:
                for i in range(array.shape[1]):
                    columns[f'{name}.{i}'] = array[:, i]
            else:
                columns[name] = value
        return pd.DataFrame(columns)

    constructors = {**rdata.conversion.DEFAULT_CLASS_MAP, 'data.frame': data_frame}
    return rdata.conversion.convert(rdata.parser.parse_file(path), constructors)


def klines(raw):
    frames = []
    for coin, by_exchange in read_rdata(raw/'cryptocompare.RData')['hourprice'].items():
        for exchange, candles in by_exchange.items():
            d = pd.DataFrame(candles)
            if d.empty:
                continue
            frames.append(pd.DataFrame({
                'coin': str(coin), 'exchange': str(exchange), 'quote': 'BTC',
                'timestamp': pd.to_datetime(d['time'].astype('int64'), unit='s', utc=True),
                'open': d['open'], 'high': d['high'], 'low': d['low'], 'close': d['close'],
                'volume': d['volumefrom'], 'amount': d['volumeto']}))
    k = pd.concat(frames, ignore_index=True)
    k = k.drop_duplicates(['coin', 'exchange', 'timestamp'], keep='last')
    return k.sort_values(['coin', 'exchange', 'timestamp'], kind='stable').reset_index(drop=True)


def events(raw):
    e = pd.DataFrame(json.loads((raw/'rawpumpolymp.json').read_text())).drop(columns='marketHistoryEntries')
    e = e.rename(columns={'currency': 'coin', 'signalTime': 'signal_time', 'channelTitle': 'channel',
                          'priceBeforePump': 'price_before_pump', 'max': 'max_price'})
    e['signal_time'] = pd.to_datetime(e['signal_time'], utc=True, format='ISO8601')
    e['hour'] = e['signal_time'].dt.floor('h')
    return e[['id', 'coin', 'exchange', 'channel', 'signal_time', 'hour', 'duration',
              'price_before_pump', 'max_price', 'volume']].sort_values('signal_time').reset_index(drop=True)


def build(raw, output):
    output.mkdir(parents=True, exist_ok=True)
    k, e = klines(raw), events(raw)
    series = set(zip(k.coin, k.exchange))
    index = pd.MultiIndex.from_frame(k[['coin', 'exchange', 'timestamp']])
    keys = pd.MultiIndex.from_frame(e[['coin', 'exchange', 'hour']].rename(columns={'hour': 'timestamp'}))
    e['has_series'] = [(c, x) in series for c, x in zip(e.coin, e.exchange)]
    e['in_candles'] = keys.isin(index)
    k['pump'] = index.isin(keys[e.in_candles.to_numpy()]).astype(np.int8)
    k[COLUMNS].to_csv(output/'klines_1h.csv.gz', index=False, float_format='%.10g')
    e.to_csv(output/'events.csv', index=False)
    panel = read_rdata(raw/'reg_cryptopia.RData')['reg_cryptopia']
    numeric = k[['open', 'high', 'low', 'close', 'volume', 'amount']].to_numpy(dtype=float)
    summary = {
        'candles': int(len(k)), 'series': len(series), 'coins': int(k.coin.nunique()),
        'exchanges': k.exchange.value_counts().to_dict(),
        'time_range_utc': [str(k.timestamp.min()), str(k.timestamp.max())],
        'nonfinite_values': int((~np.isfinite(numeric)).sum()),
        'ohlc_violations': int(((k.high < k[['open', 'close', 'low']].max(axis=1)) | (k.low > k[['open', 'close']].min(axis=1))).sum()),
        'zero_volume_share': float((k.volume == 0).mean()),
        'events': int(len(e)), 'events_by_exchange': e.exchange.value_counts().to_dict(),
        'events_with_series': int(e.has_series.sum()), 'events_in_candles': int(e.in_candles.sum()),
        'events_in_candles_by_exchange': e[e.in_candles].exchange.value_counts().to_dict(),
        'pump_hours': int(k.pump.sum()),
        'event_time_range_utc': [str(e.signal_time.min()), str(e.signal_time.max())],
        'reg_cryptopia_panel': {'rows': int(len(panel)), 'columns': int(panel.shape[1]), 'pumped_rows': int(panel['pumped'].sum()),
                                'time_range_utc': [str(pd.to_datetime(panel['datetime'].min(), unit='s', utc=True)),
                                                   str(pd.to_datetime(panel['datetime'].max(), unit='s', utc=True))]}}
    (output/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--raw', type=Path, default=RAW)
    parser.add_argument('--output', type=Path, default=ROOT/'.local_data'/'ohlcva'/'xu_livshits')
    args = parser.parse_args()
    print(json.dumps(build(args.raw, args.output), indent=2, ensure_ascii=False))
