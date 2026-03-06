from accelerate import Accelerator
from transformers import AutoModelForCausalLM
import torch
import gc

accelerator = Accelerator()
rank = accelerator.process_index
world_size = accelerator.num_processes

for r in range(world_size):
    if rank == r:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16
        )

        model = model.to(accelerator.device)

        # Important: free temporary CPU buffers
        gc.collect()
        torch.cuda.empty_cache()

    accelerator.wait_for_everyone()