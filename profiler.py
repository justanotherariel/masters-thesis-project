import torch
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn as nn
import time
from src.modules.training.models.transformer import TransformerCombAction, TransformerSepAction

def create_sample_data(batch_size, obs_shape, action_dim, device):
    """Create sample data for profiling."""
    obs = torch.randn(batch_size, *obs_shape, device=device)
    action = torch.randn(batch_size, action_dim, device=device)
    return (obs, action)

def profile_model(model, input_data, num_warmup=10, num_active=10):
    """Profile a single model's performance."""
    model.eval()  # Set to evaluation mode
    
    # Warmup runs
    print(f"Warming up {model.__class__.__name__}...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_data)
    
    # Profile runs
    print(f"Profiling {model.__class__.__name__}...")
    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True
    ) as prof:
        with record_function("model_inference"):
            with torch.no_grad():
                for _ in range(num_active):
                    _ = model(input_data)
    
    return prof

def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model parameters
    batch_size = 32
    in_obs_shape = (8, 8, 16)  # Example observation shape (height, width, channels)
    out_obs_shape = (8, 8, 16)  # Same as input for this example
    action_dim = 4
    d_model = 256
    n_heads = 8
    n_layers = 3
    d_ff = 1024
    drop_prob = 0.1
    
    # Create models
    models = {
        "TransformerCombAction": TransformerCombAction(
            in_obs_shape=in_obs_shape,
            out_obs_shape=out_obs_shape,
            action_dim=action_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            drop_prob=drop_prob
        ),
        "TransformerSepAction": TransformerSepAction(
            in_obs_shape=in_obs_shape,
            out_obs_shape=out_obs_shape,
            action_dim=action_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            drop_prob=drop_prob
        )
    }
    
    # Move models to device
    for model in models.values():
        model.to(device)
    
    # Create sample data
    input_data = create_sample_data(batch_size, in_obs_shape, action_dim, device)
    
    # Profile each model
    results = {}
    for name, model in models.items():
        print(f"\nProfiling {name}...")
        prof = profile_model(model, input_data)
        
        # Print results
        print(f"\nResults for {name}:")
        print("Overall Stats:")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        
        print("\nTop 10 Memory Users:")
        print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
        
        # Save traces if needed
        prof.export_chrome_trace(f"{name}_trace.json")
        
        results[name] = prof
    
    return results

if __name__ == "__main__":
    main()