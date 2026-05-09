# reviews_parser.py

from dataclasses import dataclass


@dataclass
class Config:
    FILE_PATH: str


class ReviewParser:
    def __init__(self, config: Config):
        self.config = config
        self.reviews = []

    def load_reviews(self):
        with open(self.config.FILE_PATH, "r", encoding="utf-8") as file:
            content = file.read()

        # Split reviews using blank lines
        raw_reviews = content.split("\n\n")

        # Clean and store reviews
        self.reviews = [
            review.strip()
            for review in raw_reviews
            if review.strip()
        ]

    def get_reviews(self):
        return self.reviews

    def display_reviews(self):
        for index, review in enumerate(self.reviews, start=1):
            print(f"{index}. {review}\n")