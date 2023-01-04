# python DDP_moco_ccrop.py path/to/this/config

# model
dim = 128
model = dict(type='ResNet', depth=18, num_classes=dim, maxpool=False)
moco = dict(dim=dim, K=65536, m=0.999, T=0.20, mlp=True)
loss = dict(type='CrossEntropyLoss')

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
            type='cifar_train_mcrop',
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
warmup_epochs = 800
loc_interval = 100
box_thresh = 0.10

# training optimizer & scheduler
epochs = 800
lr = 0.06
optimizer = dict(type='SGD', lr=lr, momentum=0.9, weight_decay=5e-4)

# log & save
log_interval = 20
save_interval = 200
work_dir = None  # rewritten by args
resume = None
load = None
port = 10002
