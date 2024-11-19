# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from .base import CONFIGS as BASE
from .from_monty import CONFIGS as FROM_MONTY

CONFIGS = dict()
CONFIGS.update(BASE)
CONFIGS.update(FROM_MONTY)
