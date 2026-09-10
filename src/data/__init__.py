"""Dataset loaders shared across reproduced models."""

from .amazon_books import AmazonBooks, download_amazon_books
from .amazon_c4 import AmazonC4Item, AmazonC4Query, BM25Index
from .avito import AvitoSERP
from .ml1m import MovieLens1M, RerankSample, download_movielens_1m
from .steam import SteamReviews, download_steam

__all__ = [
    "AmazonBooks",
    "download_amazon_books",
    "AmazonC4Item",
    "AmazonC4Query",
    "BM25Index",
    "AvitoSERP",
    "MovieLens1M",
    "RerankSample",
    "download_movielens_1m",
    "SteamReviews",
    "download_steam",
]
