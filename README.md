# Applied Data Science

A hands-on reference project covering applied data science techniques with Python.

## Topics

| Notebook | Description |
|---|---|
| `01_scraping.ipynb` | Web scraping with BeautifulSoup and Scrapy |

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
```

Then open any notebook with Jupyter:

```bash
uv run jupyter notebook
```

## Dependencies

- `beautifulsoup4` — HTML parsing
- `scrapy` — full-featured web crawling
- `pandas` — data manipulation
- `spacy` — NLP
- `nest-asyncio` — allows Scrapy to run inside Jupyter's event loop
