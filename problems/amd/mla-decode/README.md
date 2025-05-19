# Description

You will implement a custom mla decode kernel optimized for MI300, a few things simplified here:

1. Q, K, V data type as bfloat16
  
2. decode only with pre-allocated non-paged latent kv cache

3. no need to update kv cache

The shapes of all outer and inner dimensions of tensors are from DeepSeek-R1, and split number of heads to fit in one GPU. 
To be explicit, you will be given a tuple to tensors:

```yml
  input [bs, sq, dim]
  attn_output [bs, n_heads, sq, v_head_dim]
``` 

  where 

  0. bs::128 # batch size
  1. sk::[1024, 6144] # as kv length
  2. sq::1 # as only consider decoding
  3. dim::7168 # hidden size of deepseek v3
  4. v_head_dim::128 # head size
  5. n_heads::128 # num of q heads

  The ranking criteria is the geometric mean of the benchmark results.

  For the grand price, your kernel will be evaluated against the speed of light analysis
  and the solution closest to the speed of light will be awarded the grand price.
 
  The speed of light analysis is::
  | bs | sk | sq | dtype |  roofline time(us) |
  |---|---|---|---|---|
  | 128 | 1024 | 1 | bf16 | 106.65 |
  | 128 | 6144 | 1 | bf16 | 280.87 | 
