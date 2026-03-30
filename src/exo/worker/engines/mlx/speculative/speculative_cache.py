#!/usr/bin/env python3
"""SpeculativeArraysCache — wraps ArraysCache for correct GDN rollback.

During speculative verification (S>1), captures:
- all_states: per-step recurrent states from the speculative kernel
- conv_input: full conv_input tensor for conv state rollback

On rejection, rollback(n_accepted) restores both recurrent and conv
state to the correct intermediate position.
"""

import mlx.core as mx


class SpeculativeArraysCache:
    """Wrapper around ArraysCache that supports rollback for speculative decode.

    Delegates all normal cache operations to the underlying ArraysCache.
    Adds all_states/conv_input storage and a rollback() method.
    """

    def __init__(self, base_cache, S, conv_kernel_size=4):
        self.base = base_cache
        self._S = S
        self.n_keep = conv_kernel_size - 1  # typically 3
        self.all_states = None    # [B, T, Hv, Dv, Dk] from speculative kernel
        self.conv_input = None    # [B, n_keep+S, conv_dim] for conv rollback

    # Delegate cache operations
    def __getitem__(self, idx):
        return self.base[idx]

    def __setitem__(self, idx, val):
        self.base[idx] = val

    @property
    def cache(self):
        return self.base.cache

    @cache.setter
    def cache(self, v):
        self.base.cache = v

    @property
    def state(self):
        return self.base.state

    @state.setter
    def state(self, v):
        self.base.state = v

    @property
    def lengths(self):
        return self.base.lengths

    @lengths.setter
    def lengths(self, v):
        self.base.lengths = v

    @property
    def left_padding(self):
        return self.base.left_padding

    @left_padding.setter
    def left_padding(self, v):
        self.base.left_padding = v

    def advance(self, N):
        self.base.advance(N)

    def make_mask(self, N):
        return self.base.make_mask(N)

    def empty(self):
        return self.base.empty()

    @property
    def nbytes(self):
        return self.base.nbytes

    def rollback(self, n_accepted):
        """Roll back to state after processing n_accepted+1 tokens.

        Args:
            n_accepted: number of accepted draft tokens (0 = all rejected,
                        only the first token 'y' was processed correctly)
        """
        # Recurrent state: restore intermediate state at accepted position
        if self.all_states is not None:
            self.base.cache[1] = self.all_states[0, n_accepted]

        # Conv state: slice conv_input to the correct window
        if self.conv_input is not None:
            self.base.cache[0] = self.conv_input[:, n_accepted + 1: n_accepted + 1 + self.n_keep, :]

        # Clear stored states
        self.all_states = None
        self.conv_input = None
