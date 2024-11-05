Critical AI Visit
=================

+ Course: Critical AI (Dartmouth College, F24)
+ Instructor: Jed Dobson
+ Course Repository: [https://github.com/jeddobson/ENGL54.41-24F][repo]
+ Visit Date: 11/7/2024

[repo]: https://github.com/jeddobson/ENGL54.41-24F


Getting Started
---------------

To get set up for our session, copy/paste the following code into a Jupyter
Notebook. Note: this assumes you're working with a GPU runtime on Google Colab 

**Imports:**

```python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline, GenerationConfig
import pandas as pd


# Set device to the GPU, if available. Then turn off gradient accumulation
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)
```

**Model:**

```python
checkpoint = "allenai/OLMo-1B-hf"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
model.eval()
```

**Perplexity function:**

```py
def per_token_perplexity(logits, labels):
    """Calculate the perplexity of each token in a sequence.
    
    Reference: https://stackoverflow.com/a/77433933
    
    Parameters
    ----------
    logits : torch.Tensor
        Sequence logits
    labels : torch.Tensor
        Sequence token IDs

    Returns
    -------
    perplexities : torch.Tensor
        Every token's perplexity
    """
    # Shift the logits and labels by one position so we start from the
    # transition of the first token to the second token
    logits = logits[..., :-1, :].contiguous()
    labels = labels[..., 1:].contiguous()

    # Sqeeze out batch dimensions
    logits, labels = logits.squeeze(), labels.squeeze()

    # Calculate the cross entropy loss and exponentiate it for token-by-token
    # perplexity
    loss = F.cross_entropy(logits, labels, reduction="none")
    perplexities = torch.exp(loss)

    return perplexities
```

**Attention function:**

```python
def threshold_attentions(attentions, tokens, thresh=0.5):
    """Threshold attentions to find notable token pairs.

    Parameters
    ----------
    attentions : torch.Tensor
        Attention scores for a layer with shape (num_heads, num_tok, num_tok)
    tokens : list
        Tokens
    thresh : float
        Threshold below which tokens will be ignored

    Returns
    -------
    pairs : pd.DataFrame
        Source-target token pairs and their associated attention score
    """
    # Compute average attention across the heads. Note that we must ignore 0s
    # or the average will be skewed
    attentions = attentions.clone()
    attentions[attentions == 0] = torch.nan
    attentions = torch.nanmean(attentions, dim=0)

    # Ensure our tokens and attention scores are the same length
    if len(tokens) != len(attentions):
        raise ValueError("Token length does not match attention length")

    # Threshold the attentions
    filtered = attentions >= thresh

    # Get the indices of `True` values, which will represent the intersection
    # of a source token and its target. Find the corresponding attention score
    source, target = filtered.nonzero(as_tuple=True)
    scores = attentions[source, target].detach().cpu().numpy()

    # Align the indices with their tokens for source and target, then wrap them
    # up with the scores in a DataFrame
    source = [tokens[idx] for idx in source]
    target = [tokens[idx] for idx in target]
    pairs = pd.DataFrame({"source": source, "target": target, "score": scores})

    return pairs
```
