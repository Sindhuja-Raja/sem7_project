import unittest

from retrieve import filter_papers_by_metadata
from utils import Paper


class RetrievalFilterTests(unittest.TestCase):
    def test_filters_by_year_and_publisher(self):
        papers = [
            Paper(title="A", year=2022, publisher="Springer"),
            Paper(title="B", year=2024, publisher="IEEE"),
            Paper(title="C", year=2023, publisher="Springer"),
        ]

        filtered = filter_papers_by_metadata(papers, year_from=2023, year_to=2024, publishers=["springer"])

        self.assertEqual([p.title for p in filtered], ["C"])

    def test_returns_all_when_no_filters(self):
        papers = [Paper(title="A", year=2022, publisher="Springer")]
        self.assertEqual(filter_papers_by_metadata(papers, year_from=None, year_to=None, publishers=[]), papers)


if __name__ == "__main__":
    unittest.main()
