import torch
from torch.cuda.amp import autocast
import transformers


from models.uniir_blip import utils


def train_one_epoch(model, data_loader, optimizer, epoch, gpu_id, scheduler, global_step, scaler, config):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("loss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("inbatch_accuracy", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    header = "Train Epoch: [{}]".format(epoch)
    print_freq = config.trainer_config.print_freq

    accumulation_steps = config.trainer_config.gradient_accumulation_steps
    accumulation_counter = 0
    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(gpu_id, non_blocking=True)
            elif isinstance(value, transformers.tokenization_utils_base.BatchEncoding):
                for k, v in value.items():
                    batch[key][k] = v.to(gpu_id)

        if epoch > 0:
            alpha = config.model.alpha
        else:
            alpha = config.model.alpha * min(1, i / len(data_loader))

        # autocast for mixed precision
        with autocast():
            outputs = model(batch=batch, alpha=alpha)
            loss = outputs["loss"]
            inbatch_accuracy = outputs["accuracy"]

        # Scale the loss by the number of accumulation steps since backward averages the gradients.
        loss = loss / accumulation_steps

        # Use scaler for backward
        scaler.scale(loss).backward()

        accumulation_counter += 1
        if accumulation_counter == accumulation_steps:
            global_step += 1

            # optimizer step with scaler
            scaler.step(optimizer)
            scaler.update()

            model.zero_grad()
            scheduler.step()
            accumulation_counter = 0

        metric_logger.update(loss=loss.item() * accumulation_steps)  # We scale back the loss for logging.
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])  # TODO: might need to loop through all param groups
        metric_logger.update(inbatch_accuracy=inbatch_accuracy.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def eval_engine(model_without_ddp, model, data_loader, gpu_id, config):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("loss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("inbatch_accuracy", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    header = "Test:"
    print_freq = config.evaluator.print_freq

    # Save model states
    saved_state = model_without_ddp.state_dict()

    # Clear model queue states
    model_without_ddp.query_queue.copy_(torch.randn_like(model_without_ddp.query_queue))
    model_without_ddp.cand_queue.copy_(torch.randn_like(model_without_ddp.cand_queue))
    model_without_ddp.idx_queue.copy_(torch.full_like(model_without_ddp.idx_queue, -100))
    model_without_ddp.new_ptr_queue.zero_()
    print("Cleared model queue states.")

    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(gpu_id, non_blocking=True)
            elif isinstance(value, transformers.tokenization_utils_base.BatchEncoding):
                for k, v in value.items():
                    batch[key][k] = v.to(gpu_id)

        alpha = config.model.alpha * min(1, i / len(data_loader))

        # autocast for mixed precision
        with autocast():
            outputs = model(batch=batch, alpha=alpha)
            loss = outputs["loss"]
            inbatch_accuracy = outputs["accuracy"]

        metric_logger.update(loss=loss.item())
        metric_logger.update(inbatch_accuracy=inbatch_accuracy.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())

    # Restore model states from the saved variables
    model_without_ddp.load_state_dict(saved_state)
    print("Restored model queue states and model states from the saved variables.")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
