import contextlib
import os
import signal
import time 
from  collections.abc import Callable, Iterable, Iterator
from datetime import timedelta
from typing import Any, cast

import torch
from loguru import logger
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.elastic.multiprocessing.errors import record

