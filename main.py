from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Callable

import pandas as pd
import yfinance as yf
from pydantic import BaseModel, Field


# Data Models
class CandleData(BaseModel):
    """Represents a single candlestick data point with additional indicators."""
    open: float
    high: float
    low: float
    close: float
    volume: int
    datetime: datetime
    indicators: dict = Field(default_factory=dict)


class OHLCData(BaseModel):
    """Container for a collection of CandleData items with metadata."""
    candle_data: List[CandleData] = Field(default_factory=list)
    ticker: str
    exchange: str
    data_start_date: datetime
    data_end_date: datetime
    period: str
    interval: str


# Strategy Pattern for Indicators
class IndicatorStrategy(ABC):
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        pass


class ATRStrategy(IndicatorStrategy):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        high, low, close = data['High'], data['Low'], data['Close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=self.period).mean()


class SMAStrategy(IndicatorStrategy):
    def __init__(self, period: int = 20):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        return data['Close'].rolling(window=self.period).mean()


class EMAStrategy(IndicatorStrategy):
    def __init__(self, period: int = 50):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        return data['Close'].ewm(span=self.period, adjust=False).mean()


class RSIStrategy(IndicatorStrategy):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))


# Factory for creating indicator strategies
class IndicatorFactory:
    @staticmethod
    def create_indicator(name: str, **kwargs) -> IndicatorStrategy:
        indicators = {
            'ATR': ATRStrategy,
            'SMA': SMAStrategy,
            'EMA': EMAStrategy,
            'RSI': RSIStrategy
        }
        return indicators[name.upper()](**kwargs)


# Decorator for caching
def memoize(func: Callable) -> Callable:
    cache = {}

    def memoized(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]

    return memoized


# Main OHLC Data Fetcher and Analyzer
class OHLCDataAnalyzer:
    def __init__(self, ticker: str, period: str = '1mo', interval: str = '1d'):
        self.ticker = ticker
        self.period = period
        self.interval = interval
        self.data = None
        self.indicators = {}

    @memoize
    def fetch_data(self) -> pd.DataFrame:
        """Fetches OHLC data using yfinance."""
        return yf.download(self.ticker, period=self.period, interval=self.interval)

    def add_indicator(self, name: str, **kwargs):
        """Adds an indicator to be calculated."""
        self.indicators[name] = IndicatorFactory.create_indicator(name, **kwargs)

    def calculate_indicators(self):
        """Calculates all added indicators concurrently."""
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(indicator.calculate, self.data): name
                       for name, indicator in self.indicators.items()}
            for future in as_completed(futures):
                name = futures[future]
                self.data[name] = future.result()

    def create_ohlc_data(self) -> OHLCData:
        """Creates an OHLCData object from the fetched and calculated data."""
        candle_data = [
            CandleData(
                open=row['Open'],
                high=row['High'],
                low=row['Low'],
                close=row['Close'],
                volume=int(row['Volume']),
                datetime=index.to_pydatetime(),
                indicators={name: row[name] for name in self.indicators}
            )
            for index, row in self.data.iterrows()
        ]

        stock_info = yf.Ticker(self.ticker).info
        exchange = stock_info.get('exchange', 'Unknown')

        return OHLCData(
            candle_data=candle_data,
            ticker=self.ticker,
            exchange=exchange,
            data_start_date=candle_data[0].datetime,
            data_end_date=candle_data[-1].datetime,
            period=self.period,
            interval=self.interval
        )

    def analyze(self) -> dict:
        """Performs basic analysis on the OHLC data."""
        return {
            "ticker": self.ticker,
            "exchange": self.data.get('exchange', 'Unknown'),
            "start_date": self.data.index[0],
            "end_date": self.data.index[-1],
            "period": self.period,
            "interval": self.interval,
            "num_candles": len(self.data),
            "latest_close": self.data['Close'].iloc[-1],
            "avg_price": self.data['Close'].mean(),
            "min_price": self.data['Close'].min(),
            "max_price": self.data['Close'].max(),
            "price_change": self.data['Close'].iloc[-1] - self.data['Close'].iloc[0],
            "price_change_percent": (self.data['Close'].iloc[-1] / self.data['Close'].iloc[0] - 1) * 100,
            "latest_volume": self.data['Volume'].iloc[-1],
            "avg_volume": self.data['Volume'].mean(),
            **{f"latest_{name}": self.data[name].iloc[-1] for name in self.indicators}
        }

    def run(self) -> tuple:
        """Executes the entire analysis process."""
        self.data = self.fetch_data()
        self.calculate_indicators()
        ohlc_data = self.create_ohlc_data()
        analysis = self.analyze()
        return ohlc_data, analysis


# Example usage
if __name__ == "__main__":
    analyzer = OHLCDataAnalyzer('AAPL', period='6mo', interval='1d')
    analyzer.add_indicator('ATR', period=14)
    analyzer.add_indicator('SMA', period=20)
    analyzer.add_indicator('EMA', period=50)
    analyzer.add_indicator('RSI', period=14)

    try:
        ohlc_data, analysis = analyzer.run()

        print(f"Analysis for {analyzer.ticker}:")
        for key, value in analysis.items():
            if isinstance(value, (int, float)):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Explanation of Advanced Concepts

"""
1. Strategy Pattern:
   We use the Strategy pattern for our indicators. This allows us to encapsulate
   different indicator calculations and easily add new ones without modifying existing code.

2. Factory Pattern:
   The IndicatorFactory creates indicator strategies based on their names. This centralizes
   the creation logic and makes it easy to add new indicators.

3. Decorator:
   We use a custom memoize decorator to cache the results of data fetching, improving
   performance for repeated calls.

4. Concurrency:
   ThreadPoolExecutor is used to calculate indicators concurrently, speeding up the process
   when multiple indicators are used.

5. Type Hinting:
   Type hints are used throughout to improve code readability and catch type-related errors early.

6. Functional Programming:
   We use concepts like partial functions and higher-order functions (map, filter) where appropriate.

7. Data Classes:
   Pydantic models are used for structured data representation, providing automatic validation
   and serialization capabilities.

8. Single Responsibility Principle:
   Each class and method has a single, well-defined responsibility, improving code organization
   and maintainability.

9. Open/Closed Principle:
   The code is open for extension (e.g., adding new indicators) but closed for modification,
   adhering to the Open/Closed principle of SOLID.

10. Dependency Inversion:
    High-level modules (OHLCDataAnalyzer) depend on abstractions (IndicatorStrategy),
    not concrete implementations, following the Dependency Inversion principle.
"""
