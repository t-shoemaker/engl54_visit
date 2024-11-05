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


device = "cuda:0" if torch.cuda.is_available() else "cpu"
```

**Model:**

```python
checkpoint = "allenai/OLMo-1B-hf"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)
model.to(device)
```

**Template code:**

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
