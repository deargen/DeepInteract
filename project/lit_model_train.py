import logging
import os
from pathlib import Path

import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.strategies import DDPStrategy

from project.datasets.PICP.picp_dgl_data_module import PICPDGLDataModule
from project.utils.deepinteract_constants import NODE_COUNT_LIMIT, RESIDUE_COUNT_LIMIT
from project.utils.deepinteract_modules import LitGINI
from project.utils.deepinteract_utils import collect_args, process_args, construct_pl_logger

from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import StochasticWeightAveraging

# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for DeepInteract (https://github.com/BioinfoMachineLearning/DeepInteract):
# -------------------------------------------------------------------------------------------------------------------------------------


def main(args):
    # -----------
    # Data
    # -----------
    # Load protein interface contact prediction (PICP) data module
    picp_data_module = PICPDGLDataModule(casp_capri_data_dir=args.casp_capri_data_dir,
                                         db5_data_dir=args.db5_data_dir,
                                         dips_data_dir=args.dips_data_dir,
                                         batch_size=args.batch_size,
                                         num_dataloader_workers=args.num_workers,
                                         knn=args.knn,
                                         self_loops=args.self_loops,
                                         pn_ratio=args.pn_ratio,
                                         casp_capri_percent_to_use=args.casp_capri_percent_to_use,
                                         db5_percent_to_use=args.db5_percent_to_use,
                                         dips_percent_to_use=args.dips_percent_to_use,
                                         training_with_db5=args.training_with_db5,
                                         testing_with_casp_capri=args.testing_with_casp_capri,
                                         process_complexes=args.process_complexes,
                                         input_indep=args.input_indep, split_ver=args.split_ver)
    picp_data_module.setup()

    # ------------
    # Fine-Tuning
    # ------------
    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    ckpt_path_exists = os.path.exists(ckpt_path)
    ckpt_provided = args.ckpt_name != '' and ckpt_path_exists

    # ------------
    # Model
    # ------------
    # Assemble a dictionary of model arguments
    dict_args = vars(args)
    use_wandb_logger = args.logger_name.lower() == 'wandb'  # Determine whether the user requested to use WandB

    # Pick model and supply it with a dictionary of arguments
    # Baseline Model - Geometry-Focused Inter-Graph Node Interaction (GINI)
    model = LitGINI(num_node_input_feats=picp_data_module.dips_test.num_node_features,
                    num_edge_input_feats=picp_data_module.dips_test.num_edge_features,
                    gnn_activ_fn=nn.SiLU(),
                    num_classes=picp_data_module.dips_test.num_classes,
                    max_num_graph_nodes=NODE_COUNT_LIMIT,
                    max_num_residues=RESIDUE_COUNT_LIMIT,
                    testing_with_casp_capri=dict_args['testing_with_casp_capri'],
                    training_with_db5=dict_args['training_with_db5'],
                    pos_prob_threshold=0.5,
                    gnn_layer_type=dict_args['gnn_layer_type'],
                    num_gnn_layers=dict_args['num_gnn_layers'],
                    num_gnn_hidden_channels=dict_args['num_gnn_hidden_channels'],
                    num_gnn_attention_heads=dict_args['num_gnn_attention_heads'],
                    knn=dict_args['knn'],
                    interact_module_type=dict_args['interact_module_type'],
                    num_interact_layers=dict_args['num_interact_layers'],
                    num_interact_hidden_channels=dict_args['num_interact_hidden_channels'],
                    use_interact_attention=dict_args['use_interact_attention'],
                    num_interact_attention_heads=dict_args['num_interact_attention_heads'],
                    disable_geometric_mode=dict_args['disable_geometric_mode'],
                    num_epochs=dict_args['num_epochs'],
                    pn_ratio=dict_args['pn_ratio'],
                    dropout_rate=dict_args['dropout_rate'],
                    metric_to_track=dict_args['metric_to_track'],
                    weight_decay=dict_args['weight_decay'],
                    batch_size=dict_args['batch_size'],
                    lr=dict_args['lr'],
                    pad=dict_args['pad'],
                    viz_every_n_epochs=dict_args['viz_every_n_epochs'],
                    use_wandb_logger=use_wandb_logger,
                    weight_classes=dict_args['weight_classes'],
                    fine_tune=dict_args['fine_tune'],
                    ckpt_path=ckpt_path)
    args.experiment_name = f'LitGINI-b{args.batch_size}-gl{args.num_gnn_layers}' \
                           f'-n{args.num_gnn_hidden_channels}' \
                           f'-e{args.num_gnn_hidden_channels}' \
                           f'-il{args.num_interact_layers}-i{args.num_interact_hidden_channels}' \
        if not args.experiment_name \
        else args.experiment_name
    litgini_template_ckpt_filename_metric_to_track = f'{args.metric_to_track}:.3f'
    template_ckpt_filename = 'LitGINI-{epoch:02d}-{' + litgini_template_ckpt_filename_metric_to_track + '}'

    # ------------
    # Checkpoint
    # ------------
    if not args.fine_tune:
        model = model.load_from_checkpoint(ckpt_path,
                                           use_wandb_logger=use_wandb_logger,
                                           batch_size=args.batch_size,
                                           lr=args.lr,
                                           weight_decay=args.weight_decay,
                                           dropout_rate=args.dropout_rate) if ckpt_provided else model

    # ------------
    # Trainer
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)

    # -------------
    # Learning Rate
    # -------------
    if args.find_lr:
        lr_finder = trainer.tuner.lr_find(model, datamodule=picp_data_module)  # Run learning rate finder
        fig = lr_finder.plot(suggest=True)  # Plot learning rates
        fig.savefig('optimal_lr.pdf')
        fig.show()
        model.hparams.lr = lr_finder.suggestion()  # Save optimal learning rate
        logging.info(f'Optimal learning rate found: {model.hparams.lr}')

    # ------------
    # Logger
    # ------------
    pl_logger = construct_pl_logger(args)  # Log everything to an external logger
    trainer.logger = pl_logger  # Assign specified logger (e.g. TensorBoardLogger) to Trainer instance

    # -----------
    # Callbacks
    # -----------
    # Create and use callbacks
    mode = 'min' if 'ce' in args.metric_to_track else 'max'
    early_stop_callback = pl.callbacks.EarlyStopping(monitor=args.metric_to_track,
                                                     mode=mode,
                                                     min_delta=args.min_delta,
                                                     patience=args.patience)
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        monitor=args.metric_to_track,
        mode=mode,
        verbose=True,
        save_last=True,
        save_top_k=3,
        filename=template_ckpt_filename  # Warning: May cause a race condition if calling trainer.test() with many GPUs
    )
    lr_monitor_callback = pl.callbacks.LearningRateMonitor(logging_interval='step', log_momentum=True)
    callbacks = [early_stop_callback, ckpt_callback]
    if args.fine_tune:
        callbacks.append(lr_monitor_callback)
        
    if args.swa:
        swa_callback = StochasticWeightAveraging(swa_epoch_start=args.swa_epoch_start, swa_lrs=args.lr, annealing_epochs=args.swa_annealing_epochs, annealing_strategy=args.swa_annealing_strategy)
        callbacks.append(swa_callback)
        
    callbacks.append(TQDMProgressBar(refresh_rate=1))
        
    trainer.callbacks = callbacks

    # ------------
    # Restore
    # ------------
    # If using WandB, download checkpoint artifact from their servers if the checkpoint is not already stored locally
    if use_wandb_logger and args.ckpt_name != '' and not os.path.exists(ckpt_path):
        checkpoint_reference = f'{args.entity}/{args.project_name}/model-{args.run_id}:best'
        artifact = trainer.logger.experiment.use_artifact(checkpoint_reference, type='model')
        artifact_dir = artifact.download()
        model = model.load_from_checkpoint(Path(artifact_dir) / 'model.ckpt',
                                           use_wandb_logger=use_wandb_logger,
                                           batch_size=args.batch_size,
                                           lr=args.lr,
                                           weight_decay=args.weight_decay)

    # -------------
    # Training
    # -------------
    # Train with the provided model and DataModule
    trainer.fit(model=model, datamodule=picp_data_module)

    # -------------
    # Testing
    # -------------
    trainer.test()


if __name__ == '__main__':
    # -----------
    # Arguments
    # -----------
    # Collect all arguments
    parser = collect_args()

    # Parse all known and unknown arguments
    args, unparsed_argv = parser.parse_known_args()

    # Let the model add what it wants
    parser = LitGINI.add_model_specific_args(parser)

    # Re-parse all known and unknown arguments after adding those that are model specific
    args, unparsed_argv = parser.parse_known_args()

    # Set Lightning-specific parameter values before constructing Trainer instance
    args.max_time = {'hours': args.max_hours, 'minutes': args.max_minutes}
    args.max_epochs = args.num_epochs
    args.profiler = args.profiler_method
    args.accelerator = 'cuda' if args.num_gpus > 0 else 'cpu'
    args.auto_select_gpus = args.auto_choose_gpus
    if args.gpu_offset is None:
        args.gpus = args.num_gpus
    else:
        args.gpus = [args.gpu_offset + i for i in range(args.num_gpus)]
    args.num_nodes = args.num_compute_nodes
    args.precision = args.gpu_precision
    args.accumulate_grad_batches = args.accum_grad_batches
    args.gradient_clip_val = args.grad_clip_val
    args.gradient_clip_algo = args.grad_clip_algo
    args.deterministic = False  # Make LightningModule's training reproducible
    # cuda kernel of torch.cumsum() has deterministic implementation  

    # Set plugins for Lightning
    args.strategy = DDPStrategy()

    # Finalize all arguments as necessary
    args = process_args(args)

    # Begin execution of model training with given args
    main(args)
