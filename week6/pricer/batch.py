import os
from groq import Groq
from dotenv import load_dotenv
from pathlib import Path
import json
import pickle
from tqdm.notebook import tqdm

load_dotenv(override=True) # .env values take priority
groq = Groq(api_key = os.getenv("GROQ_API_KEY"))

MODEL = "openai/gpt-oss-20b"
BATCHES_FOLDER = "batches"
OUTPUT_FOLDER = "output"
state = Path("batches.pkl")

SYSTEM_PROMPT = """Create a concise description of a product. Respond only in this format. Do not
include part numbers.
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description of the product
Details: 1 sentence of features """

class Batch:
    BATCH_SIZE = 1_000

    batches = []

    def __init__(self, items, start, end, lite):
        self.items = items
        self.start = start
        self.end = end
        self.filename = f"{start}_{end}.jsonl"
        self.file_id = None
        self.batch_id = None
        self.output_file_id = None
        self.done = False
        folder = Path("lite") if lite else Path("full")
        self.batches = folder / BATCHES_FOLDER #This is path joining, not divisionpathlib.Path overloads / to mean “join paths”
        # Equivalent to: self.batches = Path(folder, BATCHES_FOLDER)
        self.output = folder / OUTPUT_FOLDER
        self.batches.mkdir(parents=True, exist_ok = True)
        self.output.mkdir(parents=True, exist_ok = True)
    
    def make_jsonl(self, item):
        body = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item.full},
            ],
            "reasoning_effort": "low",
        }
        line = {
            "custom_id": str(item.id), # must be a string
            "method": "POST", # Batch API needs to know how to call the endpoint
            "url": "/v1/chat/completions",
            "body": body,
            }
        return json.dumps(line)
        """Converts Python dict → JSON string

            One JSON object = one line in the file

            Perfect for .jsonl files

            ex: {"custom_id":"123","method":"POST","url":"/v1/chat/completions","body":{"model":"gpt-4.1","messages":[...],"reasoning_effort":"low"}}

            Converts one product item into one JSONL line that the OpenAI Batch API can execute. """

    def make_file(self):
        batch_file = self.batches / self.filename
        with batch_file.open("w") as f:
            for item in self.items[self.start : self.end]:
                f.write(self.make_jsonl(item) + "\n")

    def send_file(self):
        batch_file = self.batches / self.filename
        with batch_file.open("rb") as f:
            response = groq.files.create(file=f, purpose = "batch")
            """What this does

                Sends the file to the Groq/OpenAI-style API

                purpose="batch" tells the API:

                “This file will be used for batch processing.”

                The API:

                Stores the file

                Validates JSONL format

                Assigns it a file ID
                
                “Upload my batch JSONL file to the API and remember its ID.”"""
        self.file_id = response.id
        

    def submit_batch(self):
        response = groq.batches.create( # This calls the Batch API on Groq (OpenAI-compatible).
            completion_window = "24h", # “You have up to 24 hours to process this batch.”
            endpoint = "/v1/chat/completions",
            input_file_id = self.file_id,
        )

        self.batch_id = response.id

    def is_ready(self):
        # this method is about checking whether the batch job is finished.
        response = groq.batches.retrieve(self.batch_id)
        status = response.status
        if status == "completed":
            self.output_file_id = response.output_file_id # This is the ID of the results file, You save it for later downloading
        return status == "completed"

    def fetch_output(self):
        output_file = str(self.output / self.filename)
        response = groq.files.content(self.output_file_id)
        response.write_to_file(output_file)

    def apply_output(self):
        output_file = str(self.output / self.filename)
        with open(output_file, "r") as f:
            for line in f:
                json_line = json.loads(line)
                id = int(json_line["custom_id"])
                summary = json_line["response"]["body"]["choices"][0]["message"]["content"]
                self.items[id].summary = summary
        self.done = True

    @classmethod
    def create(cls, items, lite):
        for start in range(0, len(items), cls.BATCH_SIZE): 
            """ range(start, stop, step) loops from 0 → len(items) in steps of BATCH_SIZE.

                    cls.BATCH_SIZE = 1,000 (as defined in the class).

                    So each iteration defines the starting index of a batch."""
            end = min(start + cls.BATCH_SIZE, len(items)) # Prevents the last batch from going past the list length.
            batch = Batch(items, start, end, lite)
            cls.batches.append(batch) #cls.batches is a class-level list, shared across all Batch instances., Keeps track of all batches created so far.
        print(f"Created {len(cls.batches)} batches")

    
    @classmethod
    def run(cls):
        for batch in tqdm(cls.batches): # tqdm → shows a progress bar in Jupyter notebooks or terminals.
            batch.make_file()
            batch.send_file()
            batch.submit_batch()
        print(f"Submitted {len(cls.batches)} batches")


    @classmethod
    def fetch(cls):
        """This method is about checking batch completion, downloading results, and applying them"""
        for batch in tqdm(cls.batches):
            if not batch.done: # Skips batches that were already processed to avoid redundancy.
                if batch.is_ready(): # If not ready, this batch is skipped for now.
                    batch.fetch_output()
                    batch.apply_output() # Parses the downloaded output and updates each item.
                    """Typically, it will:

                        Map each response back to custom_id

                        Save the AI-generated content to your data structure"""
        finished = [batch for batch in cls.batches if batch.done]
        print(f"Finished {len(finished)} of {len(cls.batches)} batches")

    
    @classmethod
    def save(cls):
        # Purpose: save all batch objects to disk for later use.
        items = cls.batches[0].items # the full list of items in the first batch.
        for batch in cls.batches:
            batch.items = None
            """Prevents pickle from saving the items list in every batch object.

                Reduces file size significantly.

                After pickling, items will be restored.
            """
        with state.open("wb") as f: # state → Path("batches.pkl") (your save file).
            pickle.dump(cls.batches, f) # Serializes the list of batch objects (without items) into batches.pkl.
        for batch in cls.batches:
            batch.items = items # Re-attaches the original items list to each batch object, Ensures the program continues running normally after saving.
        print(f"Saved {len(cls.batches)} batches")

    
    @classmethod
    def load(cls):
        with state.open("rb") as f:
            cls.batches = pickle.load(f)
        for batch in cls.batches:
            batch.items = items
        print(f"Loaded {len(cls.batches)} batches")