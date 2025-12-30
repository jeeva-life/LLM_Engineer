from pydantic import BaseModel
from datasets import Dataset, DatasetDict, load_dataset # a single split (train / test / val), # DatasetDict → a collection of splits, # load_dataset → download a dataset from the Hub
from typing import Optional, Self

PREFIX = "Price is $"
QUESTION = "What does this cost to the nearest dollar?"

class Item(BaseModel):
    """
    An item is a datapoint of a product with a price.

    An Item represents one product data point:

        product metadata

        prompt text

        model-ready formatted input/output

        This is the unit of data used everywhere else (batching, API calls, datasets).
            """

    title: str
    category: str
    price: float
    full: Optional[str] = None
    weight: Optional[float] = None
    summary: Optional[str] = None
    prompt: Optional[str] = None
    id: Optional[int] = None

    def make_prompt(self, text: str):
        self.prompt = f"{QUESTION}\n\n{text}\n\n{PREFIX}{round(self.price)}.00"

    def test_prompt(self) -> str:
        return self.prompt.split(PREFIX)[0] + PREFIX

    def __repr__(self) -> str:
        return f"<{self.title} = ${self.price:.2f}>"

    @staticmethod
    def push_to_hub(dataset_namae: str, train: list[Self], val: list[Self], test: list[Self]):
        """Push Item lists to HuggingFace Hub"""
        DatasetDict(
            {
                "train": Dataset.from_list([item.model_dump() for item in train]), # model_dump() → converts Item → plain dict
                "validation": Dataset.from_list([item.model_dump() for item in val]), # HF datasets require dictionaries
                "test": Dataset.from_list([item.model_dump() for item in test]),
            }
        ).push_to_hub(dataset_name = dataset_name, private = True)

    @staticmethod
    def from_hub(cls, dataset_name: str) -> tuple[list[Self], list[Self], list[Self]]:
        """Load Item lists from HuggingFace Hub and reconstruct items"""
        ds = load_dataset(dataset_name)
        return (
            [cls.model_validate(item) for item in ds["train"]], # Converts raw dict → validated Item, Restores type safety and methods
            [cls.model_validate(item) for item in ds["validation"]],
            [cls.model_validate(item) for item in ds["test"]],
        )

"""
This file defines what a “product example” looks like, how to turn it into an LLM prompt, how to store it, and how to load/save it as a machine-learning dataset.
"""