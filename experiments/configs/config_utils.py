# Copyright 2025 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import importlib.util
import os
import sys


def import_config_from_monty(config_file, config_name):
    full_path = os.path.expanduser(f"~/tbp/tbp.monty/benchmarks/configs/{config_file}")
    sys.path.insert(0, os.path.expanduser("~/tbp/tbp.monty"))
    spec = importlib.util.spec_from_file_location(
        os.path.basename(config_file), full_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, config_name)
