import torch
import torch.nn.functional as F
def padding_tensor(tensor, lens, max_len, mode='constant', value=0):
    """
    (all_valid_boxes, ...)
    Slice tensor by lengths in the first dim, and padding them to
    max_len. Then concat them as shape (batch, max_len, ...).
    """
    cat_list, idx, dims = [], 0, len(tensor.shape)
    for l in lens:
        assert l <= max_len, "Padding: max_len should not smaller than any length!"
        padding_fmt = tuple([0 for i in range(2 * dims - 2)] + [0, max_len - l])
        cat_list.append(F.pad(tensor[idx:idx+l,...], padding_fmt, mode, value)[None,...])
        idx += l
    return torch.cat(cat_list)

