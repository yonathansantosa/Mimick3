import pprint
import wandb
import os
os.environ["WANDB_MODE"]="offline"
# wandb.login(key='f8dba7836d4d8c528b40ebd197a992eb44f9c29f')
cfg = {
                    "run":int("20"),
                    "char_emb_dim":int("20"),
                    "char_max_len":int("20"),
                    "char_emb_ascii":"20",
                    "random_seed":int("20"),
                    "shuffle_dataset":"20",
                    "neighbor":int("20"),
                    "validation_split":.8,
                    "batch_size":int("20"),
                    "max_epoch":int("20"),
                    "learning_rate":float("20"),
                    "weight_decay":float("20"),
                    "momentum":float("20"),
                    "multiplier":float("20"),
                    "classif":int("20"),
                    "model":"20",
                    "loss_fn":"20",
                    "loss_reduction":"20",
                    "pca_components":"20"
                }
# wandb.init(
#                 # Set the project where this run will be logged
#                 project="Mimick",
#                 # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
#                 name="test",
#                 config=cfg
#             )
pprint.pp(cfg)
# wandb.finish()