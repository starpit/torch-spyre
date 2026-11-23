# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The spyre_rt build's card-to-card collectives (csrc/distributed_rt).

The eager backend (all_reduce: the parts summed in rank order; all_gather)
and the compiled path's in-graph all-reduce (the exchange, every rank's part
read from this card's window, the graph's own adds) are checked against the
CPU, against each other, and across ranks:

- eager all_reduce and all_gather, exact against the CPU on data whose sums
  are exact in fp16;
- every rank holds the same bits after an all_reduce of random data;
- the compiled all-reduce, bit for bit the same as the compiled path with the
  graph's own sum switched off (allreduce_run: the eager chain), on a plain
  input and on a matmul's output (whose device layout is the graph's own);
- the compiled code reads every rank's part once and calls no allreduce_run.

Skipped unless the build registers spyre::tp_part_run (the spyre_rt build).

Usage:
    torchrun --nproc-per-node 2 -m pytest tests/distributed/test_spyre_rt_collectives.py
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

import torch_spyre  # noqa: F401
import torch_spyre._inductor.lowering as spyre_lowering

if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
    pytest.skip(
        "RANK / WORLD_SIZE not defined, skipping distributed tests",
        allow_module_level=True,
    )
if int(os.environ["WORLD_SIZE"]) < 2:
    pytest.skip("need at least 2 ranks", allow_module_level=True)
if not torch._C._dispatch_has_kernel("spyre::tp_part_run"):
    pytest.skip("not the spyre_rt build", allow_module_level=True)

DEVICE = torch.device("spyre")
C10D_BACKEND = "spyreccl"
_GROUP_NAME = "default"


def _exact(rank: int, shape, seed: int) -> torch.Tensor:
    """Rank `rank`'s input: multiples of 1/8 in [-8, 8), so any sum of a few
    is exact in fp16 whatever the order."""
    g = torch.Generator().manual_seed(1000 * seed + 17 * rank + shape[0])
    return torch.randint(-64, 64, shape, generator=g).to(torch.float16) / 8


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.cpu().view(torch.int16)


def _all_reduce(y: torch.Tensor) -> torch.Tensor:
    y = torch.ops._c10d_functional.all_reduce(y, "sum", _GROUP_NAME)
    return torch.ops._c10d_functional.wait_tensor(y)


def _compiled(fn, in_graph: bool):
    """fn compiled with the graph's own sum on or off. Lowering runs at the
    first call, so every call is made under the setting."""
    compiled = torch.compile(fn, dynamic=False)

    def call(*args):
        real = spyre_lowering._in_graph_all_reduce_world
        if not in_graph:
            spyre_lowering._in_graph_all_reduce_world = lambda *a: None
        try:
            return compiled(*args)
        finally:
            spyre_lowering._in_graph_all_reduce_world = real

    return call


@instantiate_parametrized_tests
class TestSpyreRtCollectives(TestCase):
    @classmethod
    def setUpClass(cls):
        torch.spyre._impl._lazy_init()
        if not dist.is_initialized():
            dist.init_process_group(f"cpu:gloo,spyre:{C10D_BACKEND}")
        c10d._register_process_group(_GROUP_NAME, dist.group.WORLD)
        cls.world = dist.get_world_size()
        cls.rank = dist.get_rank()
        # Two compiles of one function must stay two artifacts: the cache key
        # cannot see the switch _compiled makes.
        torch._inductor.config.force_disable_caches = True

    @classmethod
    def tearDownClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()

    def setUp(self):
        super().setUp()
        torch.compiler.reset()

    def _assert_ranks_agree(self, t: torch.Tensor):
        outs = [torch.empty_like(t) for _ in range(self.world)]
        dist.all_gather(outs, t)
        for r, out in enumerate(outs):
            self.assertTrue(
                torch.equal(_bits(out), _bits(t)),
                f"rank {r}'s bits differ from rank {self.rank}'s",
            )

    @parametrize("shape", [(1, 4096), (32, 4096), (512, 4096), (8, 2048)])
    def test_eager_all_reduce_exact(self, shape):
        for seed in range(4):
            want = _exact(0, shape, seed)
            for r in range(1, self.world):
                want = want + _exact(r, shape, seed)
            dev = _exact(self.rank, shape, seed).to(DEVICE)
            dist.all_reduce(dev)
            self.assertEqual(dev.cpu(), want, atol=0, rtol=0)

    @parametrize("shape", [(1, 4096), (32, 4096), (8, 2048)])
    def test_eager_all_gather_exact(self, shape):
        for seed in range(4):
            dev = _exact(self.rank, shape, seed).to(DEVICE)
            outs = [torch.empty_like(dev) for _ in range(self.world)]
            dist.all_gather(outs, dev)
            for r, out in enumerate(outs):
                self.assertEqual(out.cpu(), _exact(r, shape, seed), atol=0, rtol=0)

    @parametrize("shape", [(1, 4096), (32, 4096)])
    def test_eager_all_reduce_ranks_agree(self, shape):
        g = torch.Generator().manual_seed(31 * self.rank + shape[0])
        dev = (torch.randn(shape, generator=g) * 4).to(torch.float16).to(DEVICE)
        dist.all_reduce(dev)
        self._assert_ranks_agree(dev)

    @parametrize("rows", [1, 8, 129])
    def test_compiled_all_reduce_matches_allreduce_run(self, rows):
        def in_graph(x):
            return _all_reduce(x * 1.0)

        def through_run(x):
            return _all_reduce(x * 1.0)

        f_in, f_run = _compiled(in_graph, True), _compiled(through_run, False)
        for seed in range(3):
            exact = _exact(self.rank, (rows, 4096), seed)
            want = _exact(0, (rows, 4096), seed)
            for r in range(1, self.world):
                want = want + _exact(r, (rows, 4096), seed)
            self.assertEqual(f_in(exact.to(DEVICE)).cpu(), want, atol=0, rtol=0)
            g = torch.Generator().manual_seed(7919 * seed + 31 * self.rank + rows)
            x = (torch.randn(rows, 4096, generator=g) * 4).to(torch.float16)
            got = f_in(x.to(DEVICE))
            self.assertTrue(torch.equal(_bits(got), _bits(f_run(x.to(DEVICE)))))
            self._assert_ranks_agree(got)

    @parametrize("rows", [1, 4, 8, 129])
    def test_compiled_all_reduce_of_a_matmul_matches_allreduce_run(self, rows):
        k = 4096 // self.world

        def in_graph(x, w):
            return _all_reduce(x @ w)

        def through_run(x, w):
            return _all_reduce(x @ w)

        f_in, f_run = _compiled(in_graph, True), _compiled(through_run, False)
        for seed in range(3):
            g = torch.Generator().manual_seed(11 * seed + 13 * self.rank + rows)
            x = (torch.randn(rows, k, generator=g) / 4).to(torch.float16)
            w = (torch.randn(k, 4096, generator=g) / 16).to(torch.float16)
            got = f_in(x.to(DEVICE), w.to(DEVICE))
            want = f_run(x.to(DEVICE), w.to(DEVICE))
            self.assertTrue(torch.equal(_bits(got), _bits(want)))

    def test_compiled_code_reads_every_part(self):
        def fn(x):
            return _all_reduce(x * 1.0)

        x = torch.ones(8, 4096, dtype=torch.float16, device=DEVICE)
        _, sources = run_and_get_code(torch.compile(fn, dynamic=False), x)
        code = "\n".join(sources)
        self.assertEqual(code.count("torch.ops.spyre.tp_exchange_run("), 1)
        self.assertEqual(code.count("torch.ops.spyre.tp_part_run("), self.world)
        self.assertNotIn("torch.ops.spyre.allreduce_run(", code)


if __name__ == "__main__":
    run_tests()
