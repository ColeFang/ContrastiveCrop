# python DDP_simsiam_ccrop.py path/to/this/config

# model
dim, pred_dim = 512, 128
model = dict(type='ResNet', depth=18, num_classes=dim, maxpool=False, zero_init_residual=True)
simsiam = dict(dim=dim, pred_dim=pred_dim)
loss = dict(type='CosineSimilarity', dim=1)

# data
root = './data/tiny-imagenet-200/train'
mean = (0.4802, 0.4481, 0.3975)
std = (0.2302, 0.2265, 0.2262)
batch_size = 512
num_workers = 4
data = dict(
    train=dict(
        ds_dict=dict(
            type='Tiny200_boxes',
            root=root
        ),
        rcrop_dict=dict(
            type='tiny200_train_mcrop',
            mean=mean, std=std
        ),
        ccrop_dict=dict(
            type='tiny200_train_ccrop',
            alpha=0.1,
            mean=mean, std=std
        ),
    ),
    eval_train=dict(
        ds_dict=dict(
            type='ImageFolder',
            root=root
        ),
        trans_dict=dict(
            type='tiny200_test',
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
fix_pred_lr = True
optimizer = dict(type='SGD', lr=lr, momentum=0.9, weight_decay=5e-4)


# log & save
log_interval = 20
save_interval = 200
work_dir = None  # rewritten by args
resume = None
load = None
port = 10001
