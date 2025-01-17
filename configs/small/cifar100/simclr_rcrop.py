# python DDP_simclr_ccrop.py path/to/this/config

# model
dim = 128
model = dict(type='ResNet', depth=18, num_classes=dim, maxpool=False)
loss = dict(type='NT_Xent_dist', temperature=0.5, base_temperature=0.07)

# data
root = './data'
mean = (0.5071, 0.4867, 0.4408)
std = (0.2675, 0.2565, 0.2761)
batch_size = 512
num_workers = 4
data = dict(
    train=dict(
        ds_dict=dict(
            type='CIFAR100_boxes',
            root=root,
            train=True,
        ),
        rcrop_dict=dict(
            type='cifar_train_rcrop',
            mean=mean, std=std
        ),
        ccrop_dict=dict(
            type='cifar_train_ccrop',
            alpha=0.1,
            mean=mean, std=std
        ),
    ),
    eval_train=dict(
        ds_dict=dict(
            type='CIFAR100',
            root=root,
            train=True,
        ),
        trans_dict=dict(
            type='cifar_test',
            mean=mean, std=std
        ),
    ),
)

# boxes
# boxes
warmup_epochs = 800
loc_interval = 100
box_thresh = 0.10

# training optimizer & scheduler
epochs = 800
lr = 0.06
optimizer = dict(type='SGD', lr=lr, momentum=0.9, weight_decay=5e-4)
lr_cfg = dict(  # passed to adjust_learning_rate(cfg=lr_cfg)
    type='Cosine',
    steps=epochs,
    lr=lr,
    decay_rate=0.1,
    # decay_steps=[100, 150]
    warmup_steps=0,
    # warmup_from=0.01
)


# log & save
log_interval = 20
save_interval = 200
work_dir = None  # rewritten by args
resume = None
load = None
port = 10001
