from transformers import Trainer
import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from peft import PeftModel

def _is_peft_model(model):
    return isinstance(model, PeftModel)

class DistillTrainer(Trainer):
    def __init__(self, model, temperature=None, lambda_param=None,  *args, **kwargs):
        super().__init__(model=model, *args, **kwargs)
        self.loss_function = nn.KLDivLoss(reduction="batchmean")
        self.temperature = temperature
        self.lambda_param = lambda_param
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        
        label_mask = (inputs["labels"] != -100)
        logits = inputs.pop("logits")
        if logits.size(1) != label_mask.size(1):
            missing = label_mask.size(1) - logits.size(1)
            logits = torch.cat([logits, torch.zeros(logits.size(0), missing, logits.size(2), device=logits.device)], dim=1)
            label_mask[:,-missing:] = 0
            if missing > 1:        
                print(label_mask.size(), logits.size())
        outputs = model(**inputs)

        #https://huggingface.co/docs/transformers/tasks/knowledge_distillation_for_image_classification
        soft_teacher = F.softmax(logits[label_mask] / self.temperature, dim=-1)
        soft_student = F.log_softmax(outputs.logits[label_mask] / self.temperature, dim=-1)

        # Compute the loss
        distillation_loss = self.loss_function(soft_student, soft_teacher) * (self.temperature ** 2)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                student_target_loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                student_target_loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            student_target_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        self.state.distill_loss = distillation_loss.detach().cpu().item()
        self.state.CE_loss = student_target_loss.detach().cpu().item()
        # Calculate final loss
        loss = (1. - self.lambda_param) * student_target_loss + self.lambda_param * distillation_loss
        return (loss, outputs) if return_outputs else loss

    def log(self, logs) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen

        #print(self.state)
        if (self.state.global_step % self.state.eval_steps) == 0:
            #print(self.state.global_step, self.state.logging_steps, (self.state.global_step % self.state.logging_steps) == 0)
            prefix = 'eval_'
        else:
            prefix = 'train_'
        logs[f'{prefix}distill_loss'] = self.state.distill_loss
        logs[f'{prefix}CE_loss'] = self.state.CE_loss

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)
