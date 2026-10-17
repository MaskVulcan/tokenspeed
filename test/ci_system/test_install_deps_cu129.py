# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Hopper package admission distinguishes runtime ABI from DSL tooling."""

from test.ci_system.install_deps_cu129 import _is_cuda13_runtime

import pytest


@pytest.mark.parametrize(
    "name,version",
    [
        ("nvidia-cutlass-dsl-libs-cu13", "4.8.0"),
        ("nvidia_cutlass_dsl_libs_cu13", "4.8.0"),
        ("nvidia-cutlass-dsl-libs-cu12", "4.8.0"),
        ("torch", "2.14.0+cu126"),
        ("cuda-bindings", "12.9.7"),
    ],
)
def test_cuda12_runtime_accepts_transitive_dsl_tooling(name, version):
    assert not _is_cuda13_runtime(name, version)


@pytest.mark.parametrize(
    "name,version",
    [
        ("nvidia-cudnn-cu13", "9.10.2"),
        ("nvidia-cublas", "13.0.0"),
        ("nvidia-cuda-runtime", "13.0.96"),
        ("cuda-bindings", "13.0.0"),
        ("torch", "2.14.0+cu130"),
        ("tokenspeed-cutedsl-kda", "0.0.1"),
    ],
)
def test_cuda13_runtime_still_requires_a_fresh_environment(name, version):
    assert _is_cuda13_runtime(name, version)
