# Model families

Under the [unified contract](unified-contract.md), models differ in how they consume the input
window and produce the horizon. Knowing the family explains a model's input requirements.

## Encoder-only

The encoder maps the input window directly to the forecast; there is no autoregressive decoder.
`label_len = 0`. Examples: `patchtst`, `itransformer`, `card`, `timexer`.

## Encoder-decoder (seq2seq)

A decoder consumes a `label_len` slice of the encoder history plus placeholder future steps, often
with calendar **time marks** (`x_mark`, `y_mark`). `label_len > 0`. Examples: `autoformer`,
`informer`, `fedformer`, the vanilla `transformer`.

These families are why dataloaders take a `mode` and `label_len` — see
[Build dataloaders](../how-to/dataloaders.md).

## Frequency / decomposition models

Some models operate in the frequency domain (FFT-based) or decompose trend/seasonality —
`fedformer`, `etsformer`, `autoformer`, `pathformer`. These have specific numerical characteristics
worth noting when profiling.

## Continuous-time / ODE

`contiformer` treats attention as a continuous-time ODE, integrating pairwise interactions across
time steps — expressive but memory-intensive at long context.

## Pretrained / zero-shot foundation models

`chronos2`, `lag_llama` bring pretrained weights and can forecast with little or no task training.

!!! info "Capabilities as data"
    A model's family and input requirements are being formalized as declared **capabilities** (see
    the integration design). Each model's page will surface them so you never have to guess.
