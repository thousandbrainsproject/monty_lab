# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.


from .base_visitor import BaseLibraryVisitor


class ScipyCallVisitor(BaseLibraryVisitor):
    """Visitor for tracking SciPy function calls and operations."""

    def __init__(self):
        super().__init__("scipy")
