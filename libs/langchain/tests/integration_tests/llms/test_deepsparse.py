"""Test DeepSparse wrapper."""
import time
import asyncio

from huggingface_hub import snapshot_download

from langchain.llms import DeepSparse

MODEL_ID = "mgoin/TinyStories-1M-deepsparse"
MODEL_PATH = snapshot_download(repo_id=MODEL_ID)


def test_deepsparse_call() -> None:
    """Test valid call to DeepSparse."""
    config = {"prompt_sequence_length": 1}
    llm = DeepSparse(model=MODEL_PATH, config=config)

    output = llm("Once upon a time")
    assert isinstance(output, str)
    assert len(output) > 1
    assert output == "ids_to_names"

def test_deepsparse_async() -> None:
    config = {"prompt_sequence_length": 1}
    llm = DeepSparse(model=MODEL_PATH, config=config)

    def generate_serially():
        for _ in range(10):
            resp = llm.generate(["Hello, how are you?"])
            print(resp.generations[0][0].text)


    async def async_generate(llm):
        resp = await llm.agenerate(["Hello, how are you?"])
        print(resp.generations[0][0].text)


    async def generate_concurrently():
        tasks = [async_generate(llm) for _ in range(10)]
        await asyncio.gather(*tasks)


    async def internal_fn():
        s = time.perf_counter()
        await generate_concurrently()
        elapsed = time.perf_counter() - s
        print("\033[1m" + f"Concurrent executed in {elapsed:0.2f} seconds." + "\033[0m")

        s = time.perf_counter()
        generate_serially()
        elapsed = time.perf_counter() - s
        print("\033[1m" + f"Serial executed in {elapsed:0.2f} seconds." + "\033[0m")

    # Run the internal_fn function
    asyncio.run(internal_fn())