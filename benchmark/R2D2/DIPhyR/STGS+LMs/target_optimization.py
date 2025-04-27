"""
Functions for optimizing prompts for individual targets.
"""
import torch
import wandb
import logging
from pathlib import Path
import json

logger = logging.getLogger("target_optimization")


def optimize_for_target(target_info, model, tokenizer, device, config, run_id, output_dir):
    """
    Optimize a prompt for a specific target sentence.
    
    Args:
        target_info: Dictionary containing target sentence information
        model: The language model
        tokenizer: The tokenizer
        device: The device to run optimization on
        config: Dictionary of optimization parameters
        run_id: Unique identifier for the W&B run
        output_dir: Directory to save results locally
        
    Returns:
        result: Dictionary containing optimization results
    """
    from main import optimize_inputs, TokenOverlapMetric
    from evaluation_utils import evaluate_generated_output
    
    target_id = target_info["id"]
    target_text = target_info["text"]
    k_target = target_info["k_target"]
    
    # Get pre-computed perplexity from dataset if available
    target_perplexity = target_info.get("perplexity", None)
    
    logger.info(f"Optimizing prompt for target {target_id}: '{target_text}'")
    
    # Create a run name that includes the target information
    run_name = f"{run_id}_target{target_id}_k{k_target}"
    
    # Initialize W&B for this target
    target_config = config.copy()
    target_config.update({
        "target_id": target_id,
        "target_text": target_text,
        "target_k": k_target,
        "target_avg_rank": target_info.get("avg_rank", 0),
        "target_length": target_info.get("length", len(target_text.split())),
        "target_perplexity": target_perplexity
    })
    
    with wandb.init(
        project=config["wandb_project"],
        entity=config.get("wandb_entity"),
        name=run_name,
        group=run_id,
        job_type="single_target_optimization",
        config=target_config
    ) as target_run:
    
        # Tokenize target for later comparison
        target_tokens = tokenizer(target_text, return_tensors="pt").input_ids[0].cpu().tolist()
        
        # Optimize inputs for this target
        optimized_inputs, losses = optimize_inputs(
            model=model,
            tokenizer=tokenizer,
            device=device,
            target_text=target_text,
            losses=config["losses"],
            bptt=config["bptt"],
            seq_len=config["seq_len"],
            epochs=config["epochs"],
            learning_rate=config["learning_rate"],
            temperature=config["temperature"],
            bptt_temperature=config["bptt_temperature"],
            learnable_temperature=config["learnable_temperature"],
            bptt_learnable_temperature=config["bptt_learnable_temperature"],
            stgs_hard=config["stgs_hard"],
            bptt_stgs_hard=config["bptt_stgs_hard"],
            bptt_hidden_state_conditioning=config["bptt_hidden_state_conditioning"],
            plot_every=config["plot_every"],
            eps=config["eps"],
            bptt_eps=config["bptt_eps"],
            vocab_threshold=config["vocab_threshold"],
            filter_vocab=config["filter_vocab"],
            max_gradient_norm=config["max_gradient_norm"],
            batch_size=config["batch_size"],
            kwargs=config,
        )
        
        # Extract the optimized prompt tokens
        optimized_tokens = torch.argmax(optimized_inputs[0], dim=-1).cpu().tolist()
        optimized_text = tokenizer.decode(optimized_tokens)
        
        # Test the optimized prompt by running inference
        with torch.no_grad():
            # Convert tokens to embeddings
            embedding_layer = model.get_input_embeddings()
            input_embeddings = embedding_layer(torch.tensor([optimized_tokens], device=device))
            
            # Generate completion from the model
            outputs = model.generate(
                inputs_embeds=input_embeddings,
                max_length=len(target_tokens) + len(optimized_tokens),
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
                do_sample=False  # Use greedy decoding for deterministic results
            )
            
            # Extract generated tokens (excluding prompt tokens)
            generated_tokens = outputs[0, len(optimized_tokens):].cpu().tolist()
        
        # Evaluate the generated output
        evaluation_metrics = evaluate_generated_output(generated_tokens, target_tokens, tokenizer)
        
        # Calculate token overlap metrics
        token_overlap_metric = TokenOverlapMetric(
            target_text=target_text,
            tokenizer=tokenizer
        )
        token_metrics = token_overlap_metric.measure(
            prompt_tokens=torch.tensor(optimized_tokens)
        )
        
        # Log evaluation metrics to W&B
        wandb.log({**evaluation_metrics, **token_metrics})
        
        # Save results locally
        target_output_dir = Path(output_dir) / f"target_{target_id}"
        target_output_dir.mkdir(parents=True, exist_ok=True)
        
        result = {
            "target_id": target_id,
            "target_text": target_text,
            "target_k": k_target,
            "target_perplexity": target_perplexity,
            "optimized_tokens": optimized_tokens,
            "optimized_text": optimized_text,
            "generated_tokens": generated_tokens,
            "generated_text": evaluation_metrics["generated_text"],
            "evaluation": evaluation_metrics,
            "token_metrics": token_metrics,
            "final_loss": float(losses[-1]),
            "loss_history": [float(loss) for loss in losses]
        }
        
        with open(target_output_dir / "result.json", "w") as f:
            json.dump(result, f, indent=2)
        
        # Save the tensor for future use
        torch.save(optimized_inputs, target_output_dir / "optimized_inputs.pt")
        
        # Create and log artifact
        artifact = wandb.Artifact(
            name=f"optimized-prompt-{target_id}",
            type="model",
            description=f"Optimized prompt for target '{target_text}'"
        )
        artifact.add_file(target_output_dir / "result.json")
        artifact.add_file(target_output_dir / "optimized_inputs.pt")
        target_run.log_artifact(artifact)
    
    #target_run.finish()

    logger.info(f"Optimization completed for target {target_id}")
    logger.info(f"Optimized prompt: '{optimized_text}'")
    logger.info(f"Final loss: {result['final_loss']}")
    logger.info(f"Exact match: {evaluation_metrics['exact_match']}")
    logger.info(f"Token accuracy: {evaluation_metrics['token_accuracy']:.4f}")
    logger.info(f"LCS ratio: {evaluation_metrics['lcs_ratio']:.4f}")
    logger.info(f"Unigram overlap: {evaluation_metrics['unigram_overlap']:.4f}")
    logger.info(f"Bigram overlap: {evaluation_metrics['bigram_overlap']:.4f}")
    
    return result
