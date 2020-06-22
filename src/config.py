import torch
from pytorch_lightning.utilities import parsing
from base_config import __C, parse_args_and_set_config, edict, _to_values_only


parse_bool = lambda x: bool(parsing.str_to_bool(x))

if torch.cuda.is_available():
    __C.orig_dir = (
        "/storage1/samenabar/code/CLMAC/clevr-dataset-gen/datasets/CLEVR_v1.2",
        edict(type=str),
    )
    __C.uni_dir = (
        "/storage1/samenabar/code/CLMAC/clevr-dataset-gen/datasets/CLEVR_Uni_v1.2",
        edict(type=str),
    )
else:
    __C.orig_dir = (
        "/Users/sebamenabar/Documents/datasets/tmp/CLEVR_v1.2",
        edict(type=str),
    )
    __C.uni_dir = (
        "/Users/sebamenabar/Documents/datasets/tmp/CLEVR_Uni_v1.2",
        edict(type=str),
    )

__C.train.num_plot_samples = (32, edict(type=int))
__C.train.augment = (False, edict(type=parse_bool))
__C.train.dataset = ("orig", edict(choices=["orig", "uni", "gqa"]))
__C.train.gradient_clip_val = (8.0, edict(type=float))

__C.train.optimizers = edict()
__C.train.optimizers.default = edict()
__C.train.optimizers.default.lr = 1e-4

__C.model = edict()
__C.model.encoder = edict()
__C.model.encoder.type = (
    "resnet50",
    edict(choices=["none", "scratch", "pretrained", "resnet50", "resnet101"]),
)
__C.model.encoder.ckpt_fp = ("", edict(type=str))
__C.model.encoder.out_nc = 512
__C.model.encoder.train = False

__C.model.mac = edict(
    weight_init="xavier_uniform",
    init_mem="random",
    max_step=(12, edict(type=int)),
    separate_syntax_semantics=False,
    use_feats=("spatial", edict(type=str, choices=["spatial", "objects", "pixels"])),
    num_gt_lobs=0,
    common=edict(
        module_dim=(512, edict(type=int)),
        # use_feats='spatial',
    ),
    input_unit=edict(
        in_channels=(512, edict(type=int)),
        wordvec_dim=300,
        rnn_dim=(512, edict(type=int)),
        bidirectional=True,
        separate_syntax_semantics_embeddings=False,
        stem_act="ELU",
    ),
    control_unit=edict(control_feed_prev=True, control_cont_activation="TANH",),
    read_unit=edict(gate=False, num_lobs=0,),
    write_unit=edict(
        rtom=False,
        self_attn=(True, edict(type=parse_bool)),
        gate=(True, edict(type=parse_bool)),
        gate_shared=(False, edict(type=parse_bool)),
    ),
    output_unit=edict(),
)

cfg = _to_values_only(__C, 0)
