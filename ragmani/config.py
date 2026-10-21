"""The experiment's settings as typed dataclasses, one section per class that reads them, and the presets built from them.

Each class takes its own section and reads it by attribute (RagAct reads c.n_queries from ModelConfig).
A run picks a preset and overrides any field with tyro's flags (`push_pick_cube --train.lr 3e-4`); W&B stores what it ran, and load_run reads it back.
"""
import os
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum, StrEnum
from pathlib import Path
from typing import Literal, get_type_hints

import tyro
from torch import nn
from transformers import SchedulerType
from tyro.conf import EnumChoicesFromValues

DATA_DIR = Path(os.environ.get("DATA_DIR", "../data")).resolve()  # every task's demos and dataset; set in .env, or by queue.yaml in a Job


class Activation(StrEnum):
  """The nonlinearity between the MLPs' layers, by name; module is its nn class."""
  RELU = "relu"
  GELU = "gelu"
  SILU = "silu"
  LEAKY_RELU = "leaky_relu"

  @property
  def module(self): 
    return {Activation.RELU: nn.ReLU, Activation.GELU: nn.GELU, Activation.SILU: nn.SiLU, Activation.LEAKY_RELU: nn.LeakyReLU}[self]


@dataclass
class TaskConfig:
  env_id: str  # the ManiSkill env
  traj: Path  # its demos, under DATA_DIR: the controller, and the frames encode renders
  dataset: Path  # its encoded rows, under DATA_DIR

  def __post_init__(self):  # this allows us to override dataset location | e.g. --tasks.0.dataset /tmp/test_dataset
    self.traj = (DATA_DIR / self.traj).resolve()
    self.dataset = (DATA_DIR / self.dataset).resolve()

  @property
  def name(self): return self.env_id.replace("-", "_")  # W&B metric names allow no hyphens


@dataclass
class EncoderConfig:
  name: Literal["ijepa", "siglip"] = "ijepa"  # key in models/encoders.py ENCODERS, which picks the weights too
  normalize: bool = True
  precision: Literal["float32", "bfloat16"] = "float32"  # must match the dataset's
  camera: Literal["render_camera"] = "render_camera"  # the env's render camera the model sees; must match the dataset's


@dataclass
class EncodeConfig:
  batch_size: int = 32  # frames per encoder forward pass
  max_episodes: int | None = None  # None encodes every episode


@dataclass
class RetrievalConfig:
  index: Literal["ivf", "exact"] = "ivf"  # ivf searches with FAISS, approximate; exact compares against every row
  nlist: int | None = None  # clusters FAISS splits the rows into; None uses the square root of the row count
  nprobe: int = 8  # clusters searched per lookup; more is slower and finds more of the true nearest


@dataclass
class CodebookConfig:
  bins: int = 65
  low: float = -5.0  # bin range, in standardized action units when model.normalize is on, raw ones otherwise; actions outside clamp to the ends
  high: float = 5.0
  symlog: bool = True  # True spaces the bins evenly in symlog space, finer near zero
  scale: float = 0.01  # symlog's linear-to-log crossover: bins are near-uniform below about scale and spread geometrically above it


@dataclass
class ModelConfig:
  d_image: int = 1280  # the image embedding's width; training and simulate set it from the dataset
  image_layers: list[int] = field(default_factory=lambda: [512, 256])  # hidden layer sizes taking the image embedding down to d_image_out
  d_image_out: int = 256

  d_state: int = 9  # qpos: 7 arm joint angles + 2 gripper fingers
  state_layers: list[int] = field(default_factory=lambda: [32, 64])
  d_state_out: int = 64

  d_action: int = 4
  normalize: bool = True  # True rescales state and action to mean 0 and std 1 using the training rows, as ACT does
  act: EnumChoicesFromValues[Activation] = Activation.RELU

  n_queries: int = 3  # searches per step, made to point in different directions; each becomes one input to the ACT head
  d_hidden: int = 128
  n_neighbours: int = 6  # rows each search returns
  n_keep: int = 3  # of those, how many are kept: the best ones
  n_attn_heads: int = 4

  chunk_len: int = 20  # actions predicted per step, the chunk ACT executes or ensembles
  d_model: int = 256
  n_heads: int = 8
  encoder_layers: int = 4
  decoder_layers: int = 4
  feedforward: int = 1024
  dropout: float = 0.1
  action_head: Literal["linear", "codebook"] = "linear"  # codebook predicts each action dimension as logits over fixed bins, trained by cross-entropy (DreamerV3 two-hot)
  codebook: CodebookConfig = field(default_factory=CodebookConfig)  # read only when action_head is codebook

  def __post_init__(self):
    if self.d_hidden % self.n_attn_heads:
      raise ValueError(f"d_hidden {self.d_hidden} must divide by n_attn_heads {self.n_attn_heads}")
    if self.d_model % self.n_heads:
      raise ValueError(f"d_model {self.d_model} must divide by n_heads {self.n_heads}")

  @property
  def activation(self): return self.act.module

  @property
  def d_key(self): return self.d_image_out + self.d_state_out


@dataclass
class SimulateAfterConfig:
  checkpoints: list[str] = field(default_factory=lambda: ["best_success", "best", "final"])  # simulated after training, each on episodes fresh layouts
  episodes: int = 0  # 0 skips it


@dataclass
class TrainConfig:
  epochs: int = 1500
  batch_size: int = 512
  seed: int = 42  # weight init, batch order, dropout, and which rows the split holds out

  lr: float = 3.0e-4
  lr_scheduler: EnumChoicesFromValues[SchedulerType] = SchedulerType.COSINE  # any transformers schedule, by its value, e.g. reduce_lr_on_plateau
  lr_scheduler_kwargs: dict[str, float] = field(default_factory=dict)  # passed to the schedule; for reduce_lr_on_plateau, rag_model.py used factor 0.1 patience 5 threshold 1e-3 min_lr 1e-5
  warmup_ratio: float = 0.02  # share of training spent ramping up to lr
  weight_decay: float = 1.0e-4
  max_grad_norm: float = 1.0  # 0 turns clipping off
  pad_weight: float = 1.0  # the end-of-demo loss, relative to the action loss
  combined_start_at: float = 0.5  # share of training before the head sees combined
  combined_warmup_by: float = 0.75  # share of training by which every row sees it

  split: Literal["frame", "episode"] = "episode"  # frame holds out random rows, episode whole demos
  eval_fraction: float = 0.2
  eval_epochs: int = 10

  save_epochs: int = 200  # best/ and final/ are always saved too
  save_limit: int = 3
  resume: bool = False  # continue from the run folder's latest checkpoint and W&B run, if any
  log_steps: int = 100

  simulate_epochs: int = 5  # 0 turns sim checks off
  simulate_episodes: int = 10  # layouts scored per check, which picks best_success
  simulate_after: SimulateAfterConfig = field(default_factory=SimulateAfterConfig)

  def __post_init__(self):
    if self.combined_warmup_by < self.combined_start_at:
      raise ValueError(f"combined_warmup_by {self.combined_warmup_by} must not come before combined_start_at {self.combined_start_at}")

  @property
  def combined_warmup(self): return self.combined_warmup_by - self.combined_start_at


@dataclass
class WandbConfig:
  entity: str = "ragmani"
  project: str = "ragmani"
  mode: Literal["online", "offline", "disabled"] = "online"  # offline uploads later with `wandb sync`
  group: str | None = None  # the W&B group the run joins, such as test_runs
  alert_grad_norm: float = 1000  # an alert when the gradient norm stays above this for 3 logs; 0 turns it off


@dataclass
class Config:
  name: str = "push_cube"  # the preset the run started from
  run_dir: Path = Path("runs/push_cube")  # checkpoints, videos and wandb_id
  tasks: list[TaskConfig] = field(default_factory=lambda: [PUSH_CUBE])  # each task's env, demos and dataset; override as --tasks.0.dataset
  encoder: EncoderConfig = field(default_factory=EncoderConfig)
  encode: EncodeConfig = field(default_factory=EncodeConfig)
  retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
  model: ModelConfig = field(default_factory=ModelConfig)
  train: TrainConfig = field(default_factory=TrainConfig)
  wandb: WandbConfig = field(default_factory=WandbConfig)


PUSH_CUBE = TaskConfig("PushCube-v1", Path("PushCube-v1/trajectory.state.pd_ee_delta_pos.cpu.h5"), Path("push_cube_jepa"))
PICK_CUBE = TaskConfig("PickCube-v1", Path("PickCube-v1/trajectory.state.pd_ee_delta_pos.cpu.h5"), Path("pick_cube_jepa"))  # the motion-planning demos converted to pd_ee_delta_pos


TRAIN = TrainConfig(
  lr=3.0e-5,
  lr_scheduler=SchedulerType.CONSTANT,
  warmup_ratio=0.0,  # the constant schedule takes no warmup
  save_epochs=50,
  resume=True,
  simulate_epochs=50,
  simulate_episodes=20,
  simulate_after=SimulateAfterConfig(episodes=20),
)


PRESETS = {
  "push_cube": ("PushCube-v1, end-effector delta-position control, I-JEPA image embeddings, constant lr 3e-5", Config(train=TRAIN)),
  "push_pick_cube": ("PushCube-v1 and PickCube-v1 in one model: both tasks' demos fill one memory, and simulate shares its episodes between them", Config(name="push_pick_cube", tasks=[PUSH_CUBE, PICK_CUBE], train=TRAIN, run_dir=Path("runs/push_pick_cube"))),
}


def cli(args=None, description=None):
  """The preset named first in args (sys.argv by default), with any flags after it laid on top: `push_cube --train.lr 3e-4`."""
  return tyro.extras.overridable_config_cli(PRESETS, args=args, description=description, use_underscores=True)


def to_dict(config):
  """A section as nested dicts of plain values, paths and enums as strings: what W&B stores as the run's config."""
  def plain(value):
    if isinstance(value, dict): return {key: plain(item) for key, item in value.items()}
    if isinstance(value, list): return [plain(item) for item in value]
    if isinstance(value, Path): return str(value)
    if isinstance(value, Enum): return value.value
    return value

  return plain(asdict(config))


def load_run(run_id, kind=Config, values=None):
  """Reload a run from its W&B config dict: each section, task, path and enum rebuilt from plain values, skipping keys that are not settings, such as the Trainer's."""
  if values is None:
    import wandb
    values = wandb.Api().run(f"{WandbConfig.entity}/{WandbConfig.project}/{run_id}").config
  hints = get_type_hints(kind)

  def rebuild(key, value):
    hint = hints[key]
    if is_dataclass(hint): return load_run(run_id, hint, value)
    if key == "tasks": return [TaskConfig(**task) for task in value]
    if hint is Path or isinstance(hint, type) and issubclass(hint, Enum): return hint(value)
    return value

  return kind(**{key: rebuild(key, value) for key, value in values.items() if key in hints})
