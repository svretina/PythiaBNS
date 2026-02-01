import os
import json
import numpy as np
import matplotlib.pyplot as plt
import corner
import dill
from pathlib import Path
import bilby

def plot_tutorial_results(result_dir):
    result_dir = Path(result_dir)
    # Find JSON result
    json_files = list(result_dir.glob("*_result.json"))
    if not json_files:
        print(f"No result JSON found in {result_dir}")
        return
    
    result_file = json_files[0] # Assuming the first found is the one we want
    # Load result
    with open(result_file, 'r') as f:
        res = json.load(f)
    
    posterior = res['posterior']['content']
    
    # We only want the 9 model parameters for the corner plot
    keys = ['a1', 'f1', 'p1', 'a2', 'f2', 'p2', 'a3', 'f3', 'p3']
    
    # Check which keys are actually in the posterior (some might be missing if not sampled)
    available_keys = [k for k in keys if k in posterior]
    
    print(f"Extracting samples for: {available_keys}")
    samples_list = [posterior[k] for k in available_keys]
    
    # Ensure all chains have same length
    min_len = min(len(s) for s in samples_list)
    samples = np.array([s[:min_len] for s in samples_list]).T
    
    # Corner Plot
    print("Generating corner plot...")
    fig = corner.corner(
        samples, 
        labels=[res['priors'][k]['kwargs'].get('latex_label', k) for k in available_keys], 
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True, 
        title_kwargs={"fontsize": 12}
    )
    plot_path = result_dir / "corner_plot.png"
    fig.savefig(plot_path)
    print(f"Corner plot saved to {plot_path}")

    # Maximum Likelihood Estimation (MLE)
    # Bilby doesn't always store 'log_likelihood' in the same place depending on the sampler
    # But often it's in the posterior
    if 'log_likelihood' in posterior:
        idx_ml = np.argmax(posterior['log_likelihood'])
        mle_params = {k: posterior[k][idx_ml] for k in available_keys}
        print(f"MLE Parameters: {mle_params}")
        
        # Plot MLE Waveform vs Injection
        # Note: For tutorial repo we might skip the full waveform plot if time/setup is complex,
        # but let's try a simple one since we're in the source tree.
        try:
            from pythiabns.models.tutorial_models import three_sines
            time = np.linspace(0, 1, 4096)
            
            inj_params = res.get('injection_parameters', {})
            # Filter inj_params to only include those needed by three_sines
            # (which takes a1, f1, p1, etc.)
            clean_inj = {k: inj_params[k] for k in available_keys if k in inj_params}
            
            y_inj = three_sines(time, **clean_inj)['plus']
            y_mle = three_sines(time, **mle_params)['plus']
            
            plt.figure(figsize=(10, 6))
            plt.plot(time, y_inj, label='Injection', alpha=0.7)
            plt.plot(time, y_mle, '--', label='MLE Recovery', alpha=0.7)
            plt.xlim(0, 0.1) # Zoom in to see sines
            plt.title("MLE Waveform Recovery (First 0.1s)")
            plt.xlabel("Time [s]")
            plt.ylabel("Strain")
            plt.legend()
            wf_plot_path = result_dir / "mle_waveform.png"
            plt.savefig(wf_plot_path)
            print(f"MLE waveform plot saved to {wf_plot_path}")
        except Exception as e:
            print(f"Could not generate waveform plot: {e}")
    # For the tutorial, let's assume we know the model or can reconstruct it
    # Easier: just plot the samples if we have the injection too.
    
    # In a real scenario, we'd use the WaveformGenerator.
    # For this tutorial script, I'll just save the figures.
    
    print("Tutorial plots generated.")
    return str(plot_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str, help="Directory containing terminal results")
    args = parser.parse_args()
    plot_tutorial_results(args.dir)
