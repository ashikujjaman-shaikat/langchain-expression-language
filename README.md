# LangChain Expression Language (LCEL) – What I Learned

Notes from the notebooks in [LCEL/](LCEL). All examples run locally with Ollama (`llama3.2`, `gemma2:2b`, `mistral:7b`, `nomic-embed-text`).

## Setup

```bash
pip install -qU langchain langchain-ollama langchain-community pypdf faiss-cpu grandalf
ollama pull llama3.2 && ollama pull gemma2:2b && ollama pull mistral:7b && ollama pull nomic-embed-text
```

## Core Idea

LCEL composes small units called **Runnables** into pipelines with the `|` operator. Every Runnable shares the same interface, so any chain automatically gets sync, async, batch and streaming support.

```python
chain = prompt | llm | StrOutputParser()
chain.invoke({"topic": "bears"})
```

## Notebook Summary

| Notebook | Topic | Key takeaways |
|---|---|---|
| [Runnable_Intro_Part1](LCEL/Runnable_Intro_Part1.ipynb) | Runnable interface | `invoke`, `batch`, `stream`, `ainvoke`; `RunnableLambda`, `RunnableGenerator` for streaming-aware custom steps; `config` with `run_name`, `tags`, `metadata` |
| [LCEL_Demo](LCEL/LCEL_Demo.ipynb) | Basics | Plain functions become Runnables via `RunnableLambda`; `\|` and `.pipe()` are equivalent; a `dict` becomes a `RunnableParallel`; plain functions in a chain are auto-wrapped |
| [LCEL_Chain_Runnables](LCEL/LCEL_Chain_Runnables.ipynb) | Chaining | `prompt \| llm \| parser`; feed one chain into another with `{"joke": chain} \| analysis_prompt`; insert custom functions mid-chain |
| [LCEL_Runnables_Parallel](LCEL/LCEL_Runnables_Parallel.ipynb) | Parallelism | `RunnableParallel` runs branches concurrently (faster than sequential); `itemgetter` extracts keys from dict input; multiple models can be used in one run |
| [LCEL_Pass_Arguments](LCEL/LCEL_Pass_Arguments.ipynb) | Passing data | `RunnablePassthrough` forwards input unchanged; combine with lambdas to modify values; used in simple RAG: `{"context": retriever, "question": RunnablePassthrough()}` |
| [LCEL_Default_Invocation_Arguments](LCEL/LCEL_Default_Invocation_Arguments.ipynb) | `.bind()` | Attach fixed arguments to a model, e.g. `stop=[...]` or `tools=[...]` (function calling) |
| [LCEL_Stream_Runnables](LCEL/LCEL_Stream_Runnables.ipynb) | Streaming | `stream` / `astream` token by token; parsers keep streaming; `JsonOutputParser` streams partial JSON; `astream_events(version="v2")` exposes intermediate events; tools must propagate `callbacks` to emit nested events |
| [LCEL_Inspect_Runnables](LCEL/LCEL_Inspect_Runnables.ipynb) | Introspection | `chain.get_graph()`, `graph.print_ascii()` (needs `grandalf`), `chain.get_prompts()`; FAISS retriever + Ollama embeddings |
| [LCEL_Route_Between_SubChains](LCEL/LCEL_Route_Between_SubChains.ipynb) | Routing | Classify the query, then route: custom function returning a chain, `RunnableBranch` (condition/chain pairs + default), or embedding similarity to pick a prompt |
| [LCEL_Runnables_Fallback](LCEL/LCEL_Runnables_Fallback.ipynb) | Reliability | `.with_fallbacks([...])` on a model or a whole chain (each with its own prompt) |
| [LCEL_Self_Constructing_Chain](LCEL/LCEL_Self_Constructing_Chain.ipynb) | Dynamic chains | A `RunnableLambda` can return a Runnable at runtime; `RunnablePassthrough().assign(...)` adds keys step by step; rephrase the question only when `chat_history` exists |

## Cheat Sheet

| Need | Use |
|---|---|
| Chain steps | `a \| b \| c` or `a.pipe(b)` |
| Wrap a function | `RunnableLambda(fn)` |
| Streaming custom transform | `RunnableGenerator(fn)` |
| Run branches concurrently | `RunnableParallel(a=..., b=...)` or a `dict` |
| Forward input unchanged | `RunnablePassthrough()` |
| Add keys to a dict flowing through | `RunnablePassthrough.assign(key=runnable)` |
| Pick a key from input | `operator.itemgetter("key")` |
| Fixed model arguments | `model.bind(stop=[...], tools=[...])` |
| Conditional routing | `RunnableBranch(...)` or `RunnableLambda` returning a chain |
| Backup on failure | `runnable.with_fallbacks([backup])` |
| Inspect structure | `get_graph()`, `print_ascii()`, `get_prompts()` |
| Tracing metadata | `config={"run_name", "tags", "metadata"}` |

## Key Lessons

- Every component takes an input and returns an output, so components are interchangeable and composable.
- A `dict` of runnables inside a chain runs them in parallel and builds the input for the next step.
- Streaming works end to end only if every step supports it; use `RunnableGenerator` for custom streaming parsers.
- Fallbacks can swap the whole chain, so the backup can use a different prompt as well as a different model.
- Routing and self-constructing chains let the pipeline decide its path at runtime.
- Check input keys carefully: prompt placeholders must match the dict keys produced by the previous step.
- Async cells (`await`, `async for`) work directly in Jupyter.

## Next Steps

- Replace the fake retrievers (first 3 PDF chunks, hard-coded text) with a real vector store retriever.
- Add a memory/history component to the self-constructing chain.
- Trace runs with LangSmith using the `run_name`, `tags` and `metadata` config.
