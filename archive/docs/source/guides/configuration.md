# Configure NeMo-Run

Nemo-Run supports two different configuration systems:
1. Python-based configuration: This system is supported by Fiddle.
1. Raw scripts and commands: These can also be used for configuration.

In the future, we may add a YAML/Hydra-based system and aim to achieve interoperability between Python and YAML if requested.

## Python Meets YAML
Let’s break down the process of configuring a Llama3 pre-training run using Nemo 2.0 with the Python-based configuration system. For brevity, we’ll use the default settings.

### Configure in Python
First, let's discuss the Pythonic configuration system in Nemo-Run. The pretraining recipe for Llama3 appears as follows:

```python
from nemo.collections import llm
from nemo.collections.llm import llama3_8b, default_log, default_resume, adam
from nemo.collections.llm.gpt.data.mock import MockDataModule

partial = run.Partial(
     llm.pretrain,
     model=llama3_8b.model(),
     trainer=llama3_8b.trainer(
         tensor_parallelism=1,
         pipeline_parallelism=1,
         pipeline_parallelism_type=None,
         virtual_pipeline_parallelism=None,
         context_parallelism=2,
         sequence_parallelism=False,
         num_nodes=1,
         num_gpus_per_node=8,
     ),
     data=Config(MockDataModule, seq_length=8192, global_batch_size=512, micro_batch_size=1),
     log=default_log(ckpt_dir=ckpt_dir, name=name),
     optim=adam.distributed_fused_adam_with_cosine_annealing(max_lr=3e-4),
     resume=default_resume(),
 )
```

The `partial` object is an instance of `run.Partial`. In turn,`run.Partial` serves a configuration object that ties together the function `llm.pretrain` with the provided args, creating a `functools.partial` object when built. Args like `llama3_8b.model` are python functions in NeMo that return `run.Config` objects for the underlying class:

```python
def model() -> run.Config[pl.LightningModule]:
    return run.Config(LlamaModel, config=run.Config(Llama3Config8B))
```

Alternatively, you could also use `run.autoconvert` as shown:
```python
@run.autoconvert
def automodel() -> pl.LightningModule:
    return LlamaModel(config=Llama3Config8B())
```

`run.autoconvert` is a decorator that helps convert regular python functions to their `run.Config` or `run.Partial` counterparts. This means that `model() == automodel()`. `run.autoconvert` uses fiddle's autoconfig under the hood and conversion is done by parsing the AST of the underlying function.

A `run.Config` instance is similar to `run.Partial`. However, `run.Partial` returns a `functools.partial` object whereas `run.Config` directly calls the configured entity. Functionally, this means that `run.Config` provides a more direct execution path.

```python
partial = run.Partial(
    LlamaModel,
    config=run.Config(
        Llama3Config8B,
        seq_length=16384
    )
)
config = run.Config(
    LlamaModel,
    config=run.Config(
        Llama3Config8B,
        seq_length=16384
    )
)
fdl.build(partial)() == fdl.build(config)
```
Building is equivalent to instantiating the underlying Python object in case of `run.Config` or building a `functools.partial` with the specified args in case of `run.Partial`.

Currently, there are certain restrictions on control flow and complex code when using `run.autoconvert`. However, you can work around this limitation by defining a function that directly returns a `run.Config` directly. This function can then be used like any regular Python function. For example:

```python
def llama3_8b_model_conf(seq_len: int) -> run.Config[LlamaModel]
    return run.Config(
        LlamaModel,
        config=run.Config(
            Llama3Config8B,
            seq_length=seq_len
        )
    )

llama3_8b_model_conf(seq_len=4096)
```

**As shown above, if you want to incorporate complex control flow, the preferred approach is to define a function that directly returns a run.Config. You can then use this function just like any regular Python function.**

This paradigm can be a bit too opinionated when it comes to defining configurations. If you’re accustomed to YAML-based configurations, transitioning to this paradigm might feel a bit tricky.  Let’s explore how we can draw parallels between the two to build a better understanding.

### Equate to YAML
Earlier we defined the llama3 8b model as follows:

```python
config = run.Config(
    LlamaModel,
    config=run.Config(
        Llama3Config8B,
        seq_length=16384
    )
)
```

In our context, this is equivalent to:
```yaml
 _target_: nemo.collections.llm.gpt.model.llama.LlamaModel
 config:
     _target_: nemo.collections.llm.gpt.model.llama.Llama3Config8B
     seq_length: 16384
```
> Note: we've used the [Hydra instantiation](https://hydra.cc/docs/advanced/instantiate_objects/overview/) syntax here.

Python operations are performed on the config rather than directly on the class. For example:

```python
config.config.seq_length *= 2
```
translates to
```yaml
 _target_: nemo.collections.llm.gpt.model.llama.LlamaModel
 config:
     _target_: nemo.collections.llm.gpt.model.llama.Llama3Config8B
     seq_length: 32768
```

We also provide `.broadcast` and `.walk` helper methods as part of `run.Config` and `run.Partial`. They can also be equated to yaml via the following example:

```python
config = run.Config(
    SomeObject,
    a=5,
    b=run.Config(
        a=10
    )
)

config.broadcast(a=20)
config.walk(a=lambda cfg: cfg.a * 2)
```

`broadcast` will give the following YAML:
```yaml
_target_: SomeObject
a: 20
b:
    _target_: SomeObject
    a: 20
```

Afterwards, `walk` will provide the following:

```yaml
_target_: SomeObject
a: 40
b:
    _target_: SomeObject
    a: 40
```

A `run.Partial` can also be understood in this context. For example, if config were a `run.Partial` instance, it would relate to:

```yaml
 _target_: nemo.collections.llm.gpt.model.llama.LlamaModel
 _partial_: true
 config:
     _target_: nemo.collections.llm.gpt.model.llama.Llama3Config8B
     seq_length: 16384
```

We hope this provides a clearer, more intuitive understanding of the Pythonic config system and how it corresponds to a YAML-based config system.

Of course, you are entitled to choose either option. Our goal is to make the interoperability as seamless and robust as possible, and we aim to achieve this in future versions. In the meantime, please report any issues to us via GitHub.

## Raw Scripts
As an alternative, you can also configure pre-training using NeMo-Run with raw scripts and commands. This is quite straightforward, as shown in the examples below:

```python
script = run.Script("./scripts/run_pretraining.sh")
inline_script = run.Script(
        inline="""
env
export DATA_PATH="/some/tmp/path"
bash ./scripts/run_pretraining.sh
"""
    )
```

You can take a configured instance and then run it on any supported environments via executors.
See [execution](./execution.md) to read more about how to define executors.
