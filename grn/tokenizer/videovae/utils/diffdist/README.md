# diffdist, for differentiable communication

borrowed from https://github.com/ag14774/diffdist and fix code:
```
    # tmp = dist.reduce(tensor_list[i], i, op, group, async_op=True)
    # to
    # tmp = dist.reduce(tensor_list[i].contiguous(), i, op, group, async_op=True)

```
