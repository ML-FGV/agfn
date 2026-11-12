import itertools
import math

import jax.numpy as jnp
import numpy as np
import pytest

from agfx.gym import AncestralGraph, Relation, create_state, damat, fapply, umask

# For 3 nodes, jnp.triu_indices(3, k=1) gives pairs [(0,1), (0,2), (1,2)]
P01, P02, P12 = 0, 1, 2


@pytest.fixture
def state3():
    return create_state(bs=1, num_nodes=3)


def add_edge(state: AncestralGraph, ln: int, rn: int, rel: Relation) -> AncestralGraph:
    """Apply a single edge assignment via fapply (keeps value and amat in sync)."""
    num_pairs = state.num_pairs
    pairs = np.array(state.pairs)
    pair_idx = next(
        i for i, (a, b) in enumerate(pairs) if (a == ln and b == rn) or (a == rn and b == ln)
    )
    if rel == Relation.RIGHT_TO_LEFT:
        # pairs are stored as (smaller, larger); RTL means rn->ln where rn>ln
        # ensure pair ordering matches: pair stored as (ln,rn) with ln<rn
        pass
    action = pair_idx + int(rel) * num_pairs
    return fapply(state, jnp.array([action]))


def fmask_for(state: AncestralGraph, rel: Relation) -> jnp.ndarray:
    P = state.num_pairs
    r = int(rel)
    return state.fmask[0, r * P : (r + 1) * P]


# ---------------------------------------------------------------------------
# Original masking/path tests (updated to use fapply-based add_edge)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rel", [Relation.LEFT_TO_RIGHT, Relation.RIGHT_TO_LEFT, Relation.BIDIRECTIONAL])
def test_assigned_pair_fully_masked(state3, rel):
    s = add_edge(state3, ln=0, rn=1, rel=rel)
    for r in Relation:
        assert fmask_for(s, r)[P01] == 0, f"fmask should be blocked for rel={r} after assigning pair (0,1)"


def test_none_pair_fully_masked(state3):
    s = add_edge(state3, ln=0, rn=1, rel=Relation.NONE)
    for r in Relation:
        assert fmask_for(s, r)[P01] == 0, "NONE-assigned pair should be fully masked"


def test_unrelated_pair_unaffected(state3):
    s = add_edge(state3, ln=0, rn=1, rel=Relation.LEFT_TO_RIGHT)
    for r in Relation:
        assert fmask_for(s, r)[P12] == 1, f"fmask should be open for rel={r} on unrelated pair (1,2)"


def test_bmask_set_for_assigned_pair(state3):
    s = add_edge(state3, ln=0, rn=1, rel=Relation.LEFT_TO_RIGHT)
    assert bool(s.bmask[0, P01]) is True
    assert bool(s.bmask[0, P02]) is False
    assert bool(s.bmask[0, P12]) is False


def test_bmask_set_for_none_pair(state3):
    """Bug 2 regression: NONE-assigned pair must appear in bmask even though amat stays 0."""
    s = add_edge(state3, ln=0, rn=1, rel=Relation.NONE)
    assert bool(s.bmask[0, P01]) is True, "NONE-assigned pair must be in bmask"
    assert bool(s.bmask[0, P02]) is False
    assert bool(s.bmask[0, P12]) is False


def test_has_dir_path_updated_after_ltr(state3):
    s = add_edge(state3, ln=0, rn=1, rel=Relation.LEFT_TO_RIGHT)
    assert bool(s.has_dir_path[0, 0, 1]) is True
    assert bool(s.has_dir_path[0, 1, 0]) is False


def test_has_dir_path_updated_after_rtl(state3):
    s = add_edge(state3, ln=0, rn=1, rel=Relation.RIGHT_TO_LEFT)
    assert bool(s.has_dir_path[0, 1, 0]) is True
    assert bool(s.has_dir_path[0, 0, 1]) is False


def test_has_bidir_path_updated_after_bidir(state3):
    s = add_edge(state3, ln=0, rn=1, rel=Relation.BIDIRECTIONAL)
    assert bool(s.has_bidir_path[0, 0, 1]) is True
    assert bool(s.has_bidir_path[0, 1, 0]) is True
    assert bool(s.has_dir_path[0, 0, 1]) is False
    assert bool(s.has_dir_path[0, 1, 0]) is False


def test_initial_state_fully_open(state3):
    assert (state3.fmask == 1).all()
    assert (state3.bmask == 0).all()


def test_two_edges_bmask(state3):
    s = add_edge(state3, ln=0, rn=1, rel=Relation.LEFT_TO_RIGHT)
    s = add_edge(s, ln=1, rn=2, rel=Relation.LEFT_TO_RIGHT)
    assert bool(s.bmask[0, P01]) is True
    assert bool(s.bmask[0, P12]) is True
    assert bool(s.bmask[0, P02]) is False


# ---------------------------------------------------------------------------
# Bug 3 regression: L should equal num_pairs (not num_pairs + 1)
# ---------------------------------------------------------------------------


def test_trajectory_length_equals_num_pairs():
    """Bug 3 regression: L=num_pairs prevents an extra step that would sample from empty fmask."""
    for n in range(2, 6):
        state = create_state(bs=1, num_nodes=n)
        assert state.L == state.num_pairs, f"n={n}: expected L={state.num_pairs}, got {state.L}"


# ---------------------------------------------------------------------------
# Bug 1 regression: relation decoding via // not %
# ---------------------------------------------------------------------------


def test_relation_decoding_ltr(state3):
    """Bug 1 regression: action=num_pairs*1 + pair_idx should set LTR edge, not NONE."""
    num_pairs = state3.num_pairs
    s = fapply(state3, jnp.array([P01 + Relation.LEFT_TO_RIGHT * num_pairs]))
    assert bool(s.amat[0, 0, 1]) is True, "LTR action should set amat[0,1]=1"
    assert bool(s.amat[0, 1, 0]) is False


def test_relation_decoding_rtl(state3):
    num_pairs = state3.num_pairs
    s = fapply(state3, jnp.array([P01 + Relation.RIGHT_TO_LEFT * num_pairs]))
    assert bool(s.amat[0, 1, 0]) is True, "RTL action should set amat[1,0]=1"
    assert bool(s.amat[0, 0, 1]) is False


def test_relation_decoding_bidir(state3):
    num_pairs = state3.num_pairs
    s = fapply(state3, jnp.array([P01 + Relation.BIDIRECTIONAL * num_pairs]))
    assert bool(s.amat[0, 0, 1]) is True
    assert bool(s.amat[0, 1, 0]) is True


def test_relation_decoding_none(state3):
    num_pairs = state3.num_pairs
    s = fapply(state3, jnp.array([P01 + Relation.NONE * num_pairs]))
    assert bool(s.amat[0, 0, 1]) is False
    assert bool(s.amat[0, 1, 0]) is False
    assert bool(s.bmask[0, P01]) is True, "NONE-assigned pair must appear in bmask"


# ---------------------------------------------------------------------------
# Bug 4 regression: transitive closure of directed paths
# ---------------------------------------------------------------------------


def test_transitive_dir_path_chain(state3):
    """Bug 4 regression: 0→1, 1→2 should set has_dir_path[0,2]=1 and block 2→0."""
    s = add_edge(state3, ln=0, rn=1, rel=Relation.LEFT_TO_RIGHT)
    s = add_edge(s, ln=1, rn=2, rel=Relation.LEFT_TO_RIGHT)
    assert float(s.has_dir_path[0, 0, 2]) == 1.0, "Transitive path 0→1→2 should be tracked"
    # RTL on pair (0,2) means action 2→0; should be blocked
    assert fmask_for(s, Relation.RIGHT_TO_LEFT)[P02] == 0, "2→0 should be masked (directed cycle)"


def test_transitive_dir_path_longer(state3):
    """Transitive closure over 3 hops: 0→1→2 and check 2→0 is blocked."""
    s = add_edge(state3, ln=0, rn=1, rel=Relation.LEFT_TO_RIGHT)
    s = add_edge(s, ln=1, rn=2, rel=Relation.LEFT_TO_RIGHT)
    # 0→2 directly should also be blocked (would create cycle via 0→1→2→0)
    assert fmask_for(s, Relation.RIGHT_TO_LEFT)[P02] == 0


# ---------------------------------------------------------------------------
# Ancestral graph validity: rollout produces valid ADMGs
# ---------------------------------------------------------------------------


def _ref_transitive_closure(amat: np.ndarray) -> np.ndarray:
    """Floyd-Warshall transitive closure."""
    n = amat.shape[0]
    tc = amat.copy().astype(bool)
    for k in range(n):
        tc |= tc[:, k : k + 1] & tc[k : k + 1, :]
    return tc


def _is_ancestral(amat_np: np.ndarray) -> bool:
    """True iff amat represents a valid ADMG (no directed cycles, no almost-directed cycles)."""
    n = amat_np.shape[0]
    amat_t = amat_np.T
    dir_part = np.where((amat_np == amat_t) & (amat_np == 1), 0, amat_np).astype(bool)
    bidir_part = np.where((amat_np == amat_t) & (amat_np == 1), amat_np, 0).astype(bool)

    tc = _ref_transitive_closure(dir_part)

    # No directed cycle: tc[i,j] & tc[j,i] for i≠j
    for i in range(n):
        for j in range(i + 1, n):
            if tc[i, j] and tc[j, i]:
                return False

    # No almost-directed cycle: bidir edge (i,j) with directed path between them
    for i in range(n):
        for j in range(n):
            if i != j and bidir_part[i, j] and (tc[i, j] or tc[j, i]):
                return False

    return True


def _random_rollout(num_nodes: int, seed: int) -> np.ndarray:
    """Apply uniform-random valid actions for a full trajectory; return final amat."""
    rng = np.random.default_rng(seed)
    state = create_state(bs=1, num_nodes=num_nodes)
    for _ in range(state.L):
        fmask_np = np.array(state.fmask[0])
        valid = np.where(fmask_np > 0)[0]
        action = int(rng.choice(valid))
        state = fapply(state, jnp.array([action]))
    return np.array(state.amat[0])


@pytest.mark.parametrize("num_nodes", [2, 3, 4, 5])
def test_rollout_produces_ancestral_graph(num_nodes):
    """100 random rollouts should all yield valid ancestral graphs."""
    for seed in range(100):
        amat = _random_rollout(num_nodes, seed)
        assert _is_ancestral(amat), (
            f"num_nodes={num_nodes} seed={seed}: non-ancestral graph produced\n{amat}"
        )


# ---------------------------------------------------------------------------
# All ancestral graphs are reachable (n=2 and n=3)
# ---------------------------------------------------------------------------


def _enumerate_amats(num_nodes: int):
    """Yield all (num_nodes, num_nodes) binary matrices from pair-relation assignments."""
    pairs = list(zip(*np.triu_indices(num_nodes, k=1)))
    for assignment in itertools.product(list(Relation), repeat=len(pairs)):
        amat = np.zeros((num_nodes, num_nodes), dtype=float)
        for (ln, rn), rel in zip(pairs, assignment):
            if rel == Relation.LEFT_TO_RIGHT:
                amat[ln, rn] = 1
            elif rel == Relation.RIGHT_TO_LEFT:
                amat[rn, ln] = 1
            elif rel == Relation.BIDIRECTIONAL:
                amat[ln, rn] = 1
                amat[rn, ln] = 1
        yield amat, assignment


def _try_reach(target_amat: np.ndarray, target_assignment, num_nodes: int) -> bool:
    """Try all orderings of pairs to see if target graph is reachable via fapply."""
    pairs = list(zip(*np.triu_indices(num_nodes, k=1)))
    num_pairs = len(pairs)
    for perm in itertools.permutations(range(num_pairs)):
        state = create_state(bs=1, num_nodes=num_nodes)
        ok = True
        for idx in perm:
            rel = target_assignment[idx]
            action = idx + int(rel) * num_pairs
            if state.fmask[0, action] == 0:
                ok = False
                break
            state = fapply(state, jnp.array([action]))
        if ok:
            return True
    return False


@pytest.mark.parametrize("num_nodes", [2, 3])
def test_all_ancestral_graphs_reachable(num_nodes):
    """Every valid ancestral graph (ADMG) over num_nodes nodes should be reachable."""
    for amat, assignment in _enumerate_amats(num_nodes):
        if not _is_ancestral(amat):
            continue
        assert _try_reach(amat, assignment, num_nodes), (
            f"num_nodes={num_nodes}: ancestral graph is not reachable\n{amat}"
        )
