import math
from enum import IntEnum
from functools import partial, reduce

import flax.struct as struct
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

_nil = -1


class Relation(IntEnum):
    NONE = 0
    LEFT_TO_RIGHT = 1
    RIGHT_TO_LEFT = 2
    BIDIRECTIONAL = 3


@struct.dataclass
class AncestralGraph:
    num_nodes: int = struct.field(pytree_node=False)

    amat: jax.Array
    value: jax.Array

    fmask: jax.Array
    bmask: jax.Array

    # Whether the (forward or backward) process has stopped
    fstopped: jax.Array
    bstopped: jax.Array

    pairs: jax.Array
    num_pairs: int = struct.field(pytree_node=False)

    relations: jax.Array

    has_dir_path: jax.Array
    has_bidir_path: jax.Array

    has_confounding: bool = struct.field(pytree_node=False)

    # Maximum trajectory length
    L: int = struct.field(pytree_node=False)

    batch_ids: jax.Array


def get_indicators(state: AncestralGraph):
    ln, rn = state.pairs[:, 0], state.pairs[:, 1]
    num_relations = len(state.relations)

    Id = jnp.eye(num_relations)

    @partial(jax.vmap, in_axes=(0,), out_axes=0)
    def get_indicator(amat: jax.Array):
        amat_lr = amat[ln, rn, None]
        amat_rl = amat[rn, ln, None]
        indicators = jnp.select(
            condlist=[
                (amat_lr == 0) & (amat_rl == 0),
                (amat_lr == 1) & (amat_rl == 0),
                (amat_lr == 0) & (amat_rl == 1),
                (amat_lr == 1) & (amat_rl == 1),
            ],
            choicelist=[Id[0], Id[1], Id[2], Id[3]],
            default=0,
        )
        return indicators

    return get_indicator(state.amat)


def create_state(bs: int, num_nodes: int, has_confounding: bool = True):
    # Create state with batch size bs

    num_pairs = math.comb(num_nodes, 2)

    fmask = jnp.ones((bs, 4 * num_pairs))
    fmask = fmask.at[:, 3 * num_pairs :].set(has_confounding)

    state = AncestralGraph(
        num_nodes=num_nodes,
        amat=jnp.zeros((bs, num_nodes, num_nodes)),
        value=_nil * jnp.ones((bs, num_nodes, num_nodes)),
        fstopped=jnp.zeros((bs,), dtype=jnp.bool),
        bstopped=jnp.ones((bs,), dtype=jnp.bool),
        fmask=fmask,
        bmask=jnp.zeros((bs, num_pairs)),
        pairs=jnp.stack(jnp.triu_indices(num_nodes, k=1), dtype=jnp.int32).T,
        num_pairs=num_pairs,
        relations=jnp.arange(4),
        has_dir_path=jnp.zeros((bs, num_nodes, num_nodes)),
        has_bidir_path=jnp.zeros((bs, num_nodes, num_nodes)),
        L=num_pairs,
        batch_ids=jnp.arange(bs),
        has_confounding=has_confounding,
    )
    return state


def from_amat(amat: jax.Array):
    bs, num_nodes, _ = amat.shape
    state = create_state(bs, num_nodes)

    ln, rn = jnp.triu_indices(num_nodes, k=1)
    value = jnp.select(
        condlist=[
            (amat[:, ln, rn] == 1) & (amat[:, rn, ln] == 0),
            (amat[:, ln, rn] == 0) & (amat[:, rn, ln] == 1),
            (amat[:, ln, rn] == 1) & (amat[:, rn, ln] == 1),
        ],
        choicelist=[1, 2, 3],
        default=0,
    )

    return state.replace(amat=amat, value=value)


def damat(amat: jax.Array):
    # Decompose the adjacency matrix
    amat_t = jnp.transpose(amat, (0, 2, 1))
    dir_amat = jnp.where((amat == amat_t) & (amat == 1), 0, amat)
    bidir_amat = jnp.where((amat == amat_t) & (amat == 1), amat, 0)
    return dir_amat, bidir_amat


def umask(
    state: AncestralGraph, ln: jax.Array, rn: jax.Array, rel: jax.Array
) -> AncestralGraph:
    bids = state.batch_ids
    # We verify whether there is a bidirected or directed path between each node
    dir_amat, bidir_amat = damat(state.amat)

    # If there is an (bi)directed edge from a to b, then there is a (bi)path from a to b
    dir_path = jnp.maximum(state.has_dir_path, dir_amat)
    bidir_path = jnp.maximum(state.has_bidir_path, bidir_amat)

    # If there is a directed path from a to ln, and from rn to b, then from a to b
    def mask_rel(x: jax.Array, r: Relation):
        return jnp.where(rel.reshape(-1, 1) == r, x, 0)

    def maxn(*arrays):
        return reduce(jnp.maximum, arrays)

    # Add identity so node i counts as reachable from itself (enables path stitching
    # at endpoints: e.g. 0→1 then 1→2 gives 0→2 because dir_path_id[1,1]=1)
    eye = jnp.eye(state.num_nodes)[None, :, :]
    dir_path_id = jnp.maximum(dir_path, eye)

    dir_path = maxn(
        dir_path,
        jnp.einsum(
            "bi,bj->bij",
            mask_rel(dir_path_id[bids, :, ln], Relation.LEFT_TO_RIGHT),
            mask_rel(dir_path_id[bids, rn, :], Relation.LEFT_TO_RIGHT),
        ),
        jnp.einsum(
            "bi,bj->bij",
            mask_rel(dir_path_id[bids, :, rn], Relation.RIGHT_TO_LEFT),
            mask_rel(dir_path_id[bids, ln, :], Relation.RIGHT_TO_LEFT),
        ),
    )

    # bidir paths have no self-loops — no identity added to bidir_path_id
    bidir_path = maxn(
        bidir_path,
        jnp.einsum(
            "bi,bj->bij",
            mask_rel(bidir_path[bids, :, ln], Relation.LEFT_TO_RIGHT),
            mask_rel(dir_path_id[bids, rn, :], Relation.LEFT_TO_RIGHT),
        ),
        jnp.einsum(
            "bi,bj->bij",
            mask_rel(bidir_path[bids, :, rn], Relation.RIGHT_TO_LEFT),
            mask_rel(dir_path_id[bids, ln, :], Relation.RIGHT_TO_LEFT),
        ),
        jnp.einsum(
            "bi,bj->bij",
            mask_rel(dir_path_id[bids, :, ln], Relation.LEFT_TO_RIGHT),
            mask_rel(bidir_path[bids, rn, :], Relation.LEFT_TO_RIGHT),
        ),
        jnp.einsum(
            "bi,bj->bij",
            mask_rel(dir_path_id[bids, :, rn], Relation.RIGHT_TO_LEFT),
            mask_rel(bidir_path[bids, ln, :], Relation.RIGHT_TO_LEFT),
        ),
        jnp.einsum(
            "bi,bj->bij",
            mask_rel(dir_path_id[bids, :, ln], Relation.BIDIRECTIONAL),
            mask_rel(dir_path_id[bids, rn, :], Relation.BIDIRECTIONAL),
        ),
        jnp.einsum(
            "bi,bj->bij",
            mask_rel(dir_path_id[bids, :, rn], Relation.BIDIRECTIONAL),
            mask_rel(dir_path_id[bids, ln, :], Relation.BIDIRECTIONAL),
        ),
    )

    # Recompute after transitive extension so almost_cycle sees full reachability
    dir_path_id = jnp.maximum(dir_path, eye)

    # We mask the action corresponding to adding the node i -> j
    # if there is a directed path from j to i
    pairs = state.pairs  # (num_pairs, 2)
    pi, pj = pairs[:, 0], pairs[:, 1]  # (num_pairs,)

    # True where the pair has been assigned any relation (including NONE, value=0)
    # value is -1 for unassigned pairs
    is_assigned = state.value[:, pi, pj] != _nil  # (bs, num_pairs)

    dir_ij = dir_path[:, pi, pj].astype(bool)
    dir_ji = dir_path[:, pj, pi].astype(bool)
    bidir_ij = bidir_path[:, pi, pj].astype(bool)
    bidir_ji = bidir_path[:, pj, pi].astype(bool)

    # Block pi -> pj if there is already a directed path from pj to pi (directed cycle)
    block_ltr = dir_ji | is_assigned | bidir_ji
    # Block pj -> pi if there is already a directed path from pi to pj
    block_rtl = dir_ij | is_assigned | bidir_ij
    # Block pi <-> pj if any directed path or almost-directed cycle risk exists
    block_bidir = dir_ij | dir_ji | is_assigned
    # Block all relations once a pair has been assigned any relation (including NONE)
    block_none = is_assigned

    # fmask layout: [NONE_0..P-1 | LTR_0..P-1 | RTL_0..P-1 | BIDIR_0..P-1]
    # where action = pair_idx + rel * num_pairs
    num_pairs = state.num_pairs
    fmask = jnp.concatenate(
        [
            jnp.where(block_none, 0, state.fmask[:, :num_pairs]),
            jnp.where(block_ltr, 0, state.fmask[:, num_pairs : 2 * num_pairs]),
            jnp.where(
                block_rtl, 0, state.fmask[:, 2 * num_pairs : 3 * num_pairs]
            ),
            jnp.where(
                block_bidir & state.has_confounding,
                0,
                state.fmask[:, 3 * num_pairs : 4 * num_pairs],
            ),
        ],
        axis=1,
    )

    # Backward mask: any assigned pair (including NONE) can be selected backward
    bmask = is_assigned.astype(jnp.float32)

    return state.replace(
        has_dir_path=dir_path,
        has_bidir_path=bidir_path,
        fmask=fmask,
        bmask=bmask,
    )


@jax.jit
def has_cycles(amat: jax.Array):
    num_nodes, _ = amat.shape
    # damat was designed for batched inputs, so we add a batch dimension
    dir_amat, bidir_amat = map(lambda x: x.squeeze(0), damat(amat[None]))

    # Compute the matrix exponential (which indicates the existence of paths)
    dir_exp = jsp.linalg.expm(dir_amat)
    th = jnp.trace(dir_exp) - num_nodes + jnp.sum(dir_exp * bidir_amat)
    return jnp.abs(th) > 1e-8


def is_mag(state: AncestralGraph) -> jax.Array:
    return ~jax.vmap(has_cycles, in_axes=(0,), out_axes=0)(state.amat)


def fapply(state: AncestralGraph, actions: jax.Array) -> AncestralGraph:
    # Pick the node pairs
    bids = state.batch_ids

    ln, rn = state.pairs[(actions % state.num_pairs)].T  # (B,) # (B,)
    rel = state.relations[actions // state.num_pairs]  # (B,)

    # Update the amat
    amat = state.amat
    ltr = rel == Relation.LEFT_TO_RIGHT
    rtl = rel == Relation.RIGHT_TO_LEFT
    bidir = rel == Relation.BIDIRECTIONAL
    amat = (
        amat.at[bids, ln, rn]
        .set(jnp.where(ltr | bidir, 1, amat[bids, ln, rn]))
        .at[bids, rn, ln]
        .set(jnp.where(rtl | bidir, 1, amat[bids, rn, ln]))
    )

    value = state.value.at[bids, ln, rn].set(rel).at[bids, rn, ln].set(rel)
    state = state.replace(amat=amat, value=value)

    # Update the forward and backward masks
    return umask(state, ln, rn, rel)


def enumerate_ancestral_graphs(
    num_nodes: int, has_confounding: bool = True
) -> list[jax.Array]:
    # Canonical DFS: iterate node pairs in a fixed order and, for each pair,
    # try the 4 relations (NONE, LTR, RTL, BIDIR). Each ancestral graph is
    # reached via exactly one sequence, so no visited set is needed.
    #
    # We carry an incremental directed transitive closure R (R[a,b] iff a
    # reaches b via a directed path of length >= 1). Adding i -> j updates
    # R to R | ((R[:,i] | e_i) outer (R[j,:] | e_j)) in O(n^2). Ancestrality
    # reduces to two local checks:
    #   - no directed cycle: before adding i -> j, require not R[j, i]
    #   - no almost-directed cycle: for every bidir (u, v), require
    #     not R'[u, v] and not R'[v, u] after the update.
    num_pairs = num_nodes * (num_nodes - 1) // 2
    if num_pairs == 0:
        return [jnp.zeros((num_nodes, num_nodes))]

    pairs = np.array(np.triu_indices(num_nodes, k=1)).T  # (num_pairs, 2)

    amat = np.zeros((num_nodes, num_nodes), dtype=np.int8)
    bidir: list[tuple[int, int]] = []
    results: list[np.ndarray] = []

    def bidirs_ok(R_new: np.ndarray) -> bool:
        for u, v in bidir:
            if R_new[u, v] or R_new[v, u]:
                return False
        return True

    def add_directed(R: np.ndarray, src: int, dst: int) -> np.ndarray:
        col = R[:, src].copy()
        col[src] = True
        row = R[dst, :].copy()
        row[dst] = True
        return R | np.outer(col, row)

    def backtrack(idx: int, R: np.ndarray) -> None:
        if idx == num_pairs:
            results.append(amat.copy())
            return

        i, j = int(pairs[idx, 0]), int(pairs[idx, 1])

        # NONE: leave the pair unassigned-as-edge
        backtrack(idx + 1, R)

        # LTR: i -> j (blocked if j already reaches i, i.e. a cycle would form)
        if not R[j, i]:
            R_new = add_directed(R, i, j)
            if bidirs_ok(R_new):
                amat[i, j] = 1
                backtrack(idx + 1, R_new)
                amat[i, j] = 0

        # RTL: j -> i
        if not R[i, j]:
            R_new = add_directed(R, j, i)
            if bidirs_ok(R_new):
                amat[j, i] = 1
                backtrack(idx + 1, R_new)
                amat[j, i] = 0

        # BIDIR: R is unchanged; fail if any directed path already exists
        if not R[i, j] and not R[j, i] and has_confounding:
            amat[i, j] = 1
            amat[j, i] = 1
            bidir.append((i, j))
            backtrack(idx + 1, R)
            bidir.pop()
            amat[i, j] = 0
            amat[j, i] = 0

    backtrack(0, np.zeros((num_nodes, num_nodes), dtype=bool))

    if not results:
        return []
    stacked = jnp.asarray(np.stack(results).astype(np.float32))
    return list(stacked)
