# flake8: noqa
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from langchain.pydantic_v1 import root_validator

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.schema.output import GenerationChunk


class DeepSparse(LLM):
    """Neural Magic DeepSparse LLM interface.

    To use, you should have the ``deepsparse`` or ``deepsparse-nightly``
    python package installed. See https://github.com/neuralmagic/deepsparse

    This interface let's you deploy optimized LLMs straight from the
    [SparseZoo](https://sparsezoo.neuralmagic.com/?useCase=text_generation)
    Example:
        .. code-block:: python
            from langchain.llms import DeepSparse
            llm = DeepSparse(model="zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base_quant-none")
    """  # noqa: E501

    pipeline: Any  #: :meta private:

    model: str
    """The path to a model file or directory or the name of a SparseZoo model stub."""

    config: Optional[Dict[str, Any]] = None
    """Key word arguments passed to the pipeline."""

    streaming: bool = False
    """Whether to stream the results, token by token."""

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "config": self.config,
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "deepsparse"

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that ``deepsparse`` package is installed."""
        try:
            from deepsparse import Pipeline
        except ImportError:
            raise ImportError(
                "Could not import `deepsparse` package. "
                "Please install it with `pip install deepsparse`"
            )

        config = values["config"] or {}

        values["pipeline"] = Pipeline.create(
            task="text_generation",
            model_path=values["model"],
            **config,
        )
        return values

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Generate text from a prompt.
        Args:
            prompt: The prompt to generate text from.
            stop: A list of strings to stop generation when encountered.
        Returns:
            The generated text.
        Example:
            .. code-block:: python
                from langchain.llms import DeepSparse
                llm = DeepSparse(model="zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base_quant-none")
                llm("Tell me a joke.")
        """
        if self.streaming:
            combined_output = ""
            for chunk in self._stream(
                prompt=prompt, stop=stop, run_manager=run_manager, **kwargs
            ):
                combined_output += chunk.text
            text = combined_output
        else:
            text = self.pipeline(sequences=prompt).sequences[0]

        if stop is not None:
            text = enforce_stop_tokens(text, stop)

        return text

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Generate text from a prompt.
        Args:
            prompt: The prompt to generate text from.
            stop: A list of strings to stop generation when encountered.
        Returns:
            The generated text.
        Example:
            .. code-block:: python
                from langchain.llms import DeepSparse
                llm = DeepSparse(model="zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base_quant-none")
                llm("Tell me a joke.")
        """
        if self.streaming:
            combined_output = ""
            async for chunk in self._astream(
                prompt=prompt, stop=stop, run_manager=run_manager, **kwargs
            ):
                combined_output += chunk.text
            text = combined_output
        else:
            text = self.pipeline(sequences=prompt).sequences[0]

        if stop is not None:
            text = enforce_stop_tokens(text, stop)

        return text

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Yields results objects as they are generated in real time.

        It also calls the callback manager's on_llm_new_token event with
        similar parameters to the OpenAI LLM class method of the same name.
        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.
        Returns:
            A generator representing the stream of tokens being generated.
        Yields:
            A dictionary like object containing a string token.
        Example:
            .. code-block:: python
                from langchain.llms import DeepSparse
                llm = DeepSparse(
                    model="zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base_quant-none",
                    streaming=True
                )
                for chunk in llm.stream("Tell me a joke",
                        stop=["'","\n"]):
                    print(chunk, end='', flush=True)
        """
        from transformers import TextIteratorStreamer
        from threading import Thread

        streamer = TextIteratorStreamer(self.pipeline.tokenizer)

        generation_kwargs = dict(sequences=prompt, streamer=streamer)
        thread = Thread(target=self.pipeline, kwargs=generation_kwargs)
        thread.start()

        for token in streamer:
            chunk = GenerationChunk(text=token)
            yield chunk

            if run_manager:
                run_manager.on_llm_new_token(token=chunk.text)

        thread.join()

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        """Yields results objects as they are generated in real time.

        It also calls the callback manager's on_llm_new_token event with
        similar parameters to the OpenAI LLM class method of the same name.
        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.
        Returns:
            A generator representing the stream of tokens being generated.
        Yields:
            A dictionary like object containing a string token.
        Example:
            .. code-block:: python
                from langchain.llms import DeepSparse
                llm = DeepSparse(
                    model="zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base_quant-none",
                    streaming=True
                )
                for chunk in llm.stream("Tell me a joke",
                        stop=["'","\n"]):
                    print(chunk, end='', flush=True)
        """
        from transformers import TextIteratorStreamer
        from threading import Thread

        streamer = TextIteratorStreamer(self.pipeline.tokenizer)

        generation_kwargs = dict(sequences=prompt, streamer=streamer)
        thread = Thread(target=self.pipeline, kwargs=generation_kwargs)
        thread.start()

        for token in streamer:
            chunk = GenerationChunk(text=token)
            yield chunk

            if run_manager:
                await run_manager.on_llm_new_token(token=chunk.text)

        thread.join()
