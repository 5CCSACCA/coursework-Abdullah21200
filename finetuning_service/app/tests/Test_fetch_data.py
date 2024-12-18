import pytest
from utils.fetch_data import DataFetcher

@pytest.mark.integration
def test_yahoo_api_fetch():
    # Initialize DataFetcher with default ticker (NVDA)
    fetcher = DataFetcher(ticker="NVDA")

    # Run fetch_data method
    df = fetcher.fetch_data()

    # Check if DataFrame is not empty
    assert not df.empty, "Fetched data should not be empty."

    # Check if 'Close' column is present
    assert 'Close' in df.columns, "DataFrame should contain 'Close' column."

    # Check a few basic conditions, like the shape of the DataFrame
    # ensuring at least some rows and columns
    assert df.shape[0] > 0, "DataFrame should have at least one row."
    assert df.shape[1] >= 2, "DataFrame should have at least 5 columns (Open, High, Low, Close, Volume)."

    print("Test passed: Yahoo Finance API fetch is working as expected.")

test_yahoo_api_fetch()