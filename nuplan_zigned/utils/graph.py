from typing import List, Optional, Tuple, Union

import torch
from torch_geometric.utils import degree
from torch_geometric.utils import coalesce
import scipy


def add_edges(
        from_edge_index: torch.Tensor,
        to_edge_index: torch.Tensor,
        from_edge_attr: Optional[torch.Tensor] = None,
        to_edge_attr: Optional[torch.Tensor] = None,
        replace: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    from_edge_index = from_edge_index.to(device=to_edge_index.device, dtype=to_edge_index.dtype)
    mask = ((to_edge_index[0].unsqueeze(-1) == from_edge_index[0].unsqueeze(0)) &
            (to_edge_index[1].unsqueeze(-1) == from_edge_index[1].unsqueeze(0)))
    if replace:
        to_mask = mask.any(dim=1)
        if from_edge_attr is not None and to_edge_attr is not None:
            from_edge_attr = from_edge_attr.to(device=to_edge_attr.device, dtype=to_edge_attr.dtype)
            to_edge_attr = torch.cat([to_edge_attr[~to_mask], from_edge_attr], dim=0)
        to_edge_index = torch.cat([to_edge_index[:, ~to_mask], from_edge_index], dim=1)
    else:
        from_mask = mask.any(dim=0)
        if from_edge_attr is not None and to_edge_attr is not None:
            from_edge_attr = from_edge_attr.to(device=to_edge_attr.device, dtype=to_edge_attr.dtype)
            to_edge_attr = torch.cat([to_edge_attr, from_edge_attr[~from_mask]], dim=0)
        to_edge_index = torch.cat([to_edge_index, from_edge_index[:, ~from_mask]], dim=1)
    return to_edge_index, to_edge_attr


def merge_edges(
        edge_indices: List[torch.Tensor],
        edge_attrs: Optional[List[torch.Tensor]] = None,
        reduce: str = 'add') -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    edge_index = torch.cat(edge_indices, dim=1)
    if edge_attrs is not None:
        edge_attr = torch.cat(edge_attrs, dim=0)
    else:
        edge_attr = None
    return coalesce(edge_index=edge_index, edge_attr=edge_attr, reduce=reduce)


def complete_graph(
        num_nodes: Union[int, Tuple[int, int]],
        ptr: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        loop: bool = False,
        device: Optional[Union[torch.device, str]] = None) -> torch.Tensor:
    if ptr is None:
        if isinstance(num_nodes, int):
            num_src, num_dst = num_nodes, num_nodes
        else:
            num_src, num_dst = num_nodes
        edge_index = torch.cartesian_prod(torch.arange(num_src, dtype=torch.long, device=device),
                                          torch.arange(num_dst, dtype=torch.long, device=device)).t()
    else:
        if isinstance(ptr, torch.Tensor):
            ptr_src, ptr_dst = ptr, ptr
            num_src_batch = num_dst_batch = ptr[1:] - ptr[:-1]
        else:
            ptr_src, ptr_dst = ptr
            num_src_batch = ptr_src[1:] - ptr_src[:-1]
            num_dst_batch = ptr_dst[1:] - ptr_dst[:-1]
        edge_index = torch.cat(
            [torch.cartesian_prod(torch.arange(num_src, dtype=torch.long, device=device),
                                  torch.arange(num_dst, dtype=torch.long, device=device)) + p
             for num_src, num_dst, p in zip(num_src_batch, num_dst_batch, torch.stack([ptr_src, ptr_dst], dim=1))],
            dim=0)
        edge_index = edge_index.t()
    if isinstance(num_nodes, int) and not loop:
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    return edge_index.contiguous()


def bipartite_dense_to_sparse(adj: torch.Tensor) -> torch.Tensor:
    index = adj.nonzero(as_tuple=True)
    if len(index) == 3:
        batch_src = index[0] * adj.size(1)
        batch_dst = index[0] * adj.size(2)
        index = (batch_src + index[1], batch_dst + index[2])
    return torch.stack(index, dim=0)


def unbatch(
        src: torch.Tensor,
        batch: torch.Tensor,
        dim: int = 0) -> List[torch.Tensor]:
    sizes = degree(batch, dtype=torch.long).tolist()
    return src.split(sizes, dim)


def radius(x, y, r, batch_x=None, batch_y=None, max_num_neighbors=32):
    r"""Finds for each element in :obj:`y` all points in :obj:`x` within
    distance :obj:`r`.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        y (Tensor): Node feature matrix
            :math:`\mathbf{Y} \in \mathbb{R}^{M \times F}`.
        r (float): The radius.
        batch_x (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        batch_y (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^M`, which assigns each
            node to a specific example. (default: :obj:`None`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            return for each element in :obj:`y`. (default: :obj:`32`)

    :rtype: :class:`LongTensor`

    .. testsetup::

        import torch
        from torch_cluster import radius

    .. testcode::


        >>> x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        >>> batch_x = torch.tensor([0, 0, 0, 0])
        >>> y = torch.Tensor([[-1, 0], [1, 0]])
        >>> batch_y = torch.tensor([0, 0])
        >>> assign_index = radius(x, y, 1.5, batch_x, batch_y)
    """

    if batch_x is None:
        batch_x = x.new_zeros(x.size(0), dtype=torch.long)

    if batch_y is None:
        batch_y = y.new_zeros(y.size(0), dtype=torch.long)

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y

    x = torch.cat([x, 2 * r * batch_x.view(-1, 1).to(torch.float64)], dim=-1)
    y = torch.cat([y, 2 * r * batch_y.view(-1, 1).to(torch.float64)], dim=-1)

    # tree = scipy.spatial.cKDTree(x.detach().numpy())
    # _, col = tree.query(
    #     y.detach().numpy(), k=max_num_neighbors, distance_upper_bound=r + 1e-8)
    # col = [torch.from_numpy(c).to(torch.long) for c in col]
    # row = [torch.full_like(c, i) for i, c in enumerate(col)]
    # row, col = torch.cat(row, dim=0), torch.cat(col, dim=0)
    # mask = col < int(tree.n)
    # return torch.stack([row[mask], col[mask]], dim=0)

    # Calculate the pairwise Euclidean distance between x and y
    # dist = torch.cdist(x, y, p=2).float()  # p=2 for Euclidean distance  RuntimeError: Exporting the operator cdist to ONNX opset version 11 is not supported.
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    diff = x - y
    dist = torch.norm(diff, dim=2, p=2)

    # Get the indices of the nearest neighbors within the radius r
    mask = dist <= r
    row_indices, col_indices = mask.nonzero(as_tuple=True)

    # # TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect.
    # # We can't record the data flow of Python values, so this value will be treated as a constant in the future.
    # # This means that the trace might not generalize to other inputs!
    # # Clip the results to respect max_num_neighbors
    # if row_indices.size(0) > max_num_neighbors * x.size(0):
    #     # Sort distances for each query and select the closest `max_num_neighbors`
    #     _, sorted_indices = torch.sort(dist[row_indices, col_indices])
    #     selected_indices = sorted_indices[:max_num_neighbors]
    #     row_indices, col_indices = row_indices[selected_indices], col_indices[selected_indices]

    return torch.stack([col_indices, row_indices], dim=0)


def radius_graph(x: torch.Tensor, r: float,
                 batch: Optional[torch.Tensor] = None, loop: bool = False,
                 max_num_neighbors: int = 32, flow: str = 'source_to_target',
                 num_workers: int = 1) -> torch.Tensor:
    r"""Computes graph edges to all points within a given distance.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        r (float): The radius.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. :obj:`batch` needs to be sorted.
            (default: :obj:`None`)
        loop (bool, optional): If :obj:`True`, the graph will contain
            self-loops. (default: :obj:`False`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            return for each element.
            If the number of actual neighbors is greater than
            :obj:`max_num_neighbors`, returned neighbors are picked randomly.
            (default: :obj:`32`)
        flow (string, optional): The flow direction when used in combination
            with message passing (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
        num_workers (int): Number of workers to use for computation. Has no
            effect in case :obj:`batch` is not :obj:`None`, or the input lies
            on the GPU. (default: :obj:`1`)

    :rtype: :class:`LongTensor`

    .. code-block:: python

        import torch
        from torch_cluster import radius_graph

        x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        batch = torch.tensor([0, 0, 0, 0])
        edge_index = radius_graph(x, r=1.5, batch=batch, loop=False)
    """

    assert flow in ['source_to_target', 'target_to_source']
    edge_index = radius(x, x, r, batch, batch,
                        max_num_neighbors if loop else max_num_neighbors + 1,
                        )
    if flow == 'source_to_target':
        row, col = edge_index[1], edge_index[0]
    else:
        row, col = edge_index[0], edge_index[1]

    if not loop:
        mask = row != col
        row, col = row[mask], col[mask]

    return torch.stack([row, col], dim=0)
