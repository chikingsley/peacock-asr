#!/usr/bin/env python3
"""
Script to check if a dataset exists on the Hugging Face Hub and examine its structure
"""

import os
import sys
import argparse
import json
from huggingface_hub import HfApi, hf_hub_download
from datasets import load_dataset, get_dataset_config_names

def main():
    parser = argparse.ArgumentParser(description="Check if a dataset exists on the Hugging Face Hub")
    parser.add_argument("--dataset", type=str, required=True, 
                        help="Dataset name on Hugging Face Hub")
    parser.add_argument("--verbose", action="store_true",
                        help="Show more detailed information")
    
    args = parser.parse_args()
    
    # Initialize the Hugging Face API
    api = HfApi()
    
    # Check if the dataset exists
    print(f"Checking if dataset '{args.dataset}' exists...")
    try:
        dataset_info = api.dataset_info(repo_id=args.dataset)
        print(f"✅ Dataset exists on the Hub!")
        print(f"Last modified: {dataset_info.lastModified}")
        print(f"Private: {dataset_info.private}")
        
        # List all files in the repository
        files = api.list_repo_files(repo_id=args.dataset, repo_type="dataset")
        print(f"\nFiles found in repository ({len(files)}):")
        
        has_dataset_files = False
        has_readme = False
        config_files = []
        
        for file in sorted(files):
            file_info = f"  • {file}"
            
            if file.endswith(".parquet") or file.endswith(".arrow"):
                has_dataset_files = True
                file_info += " (data file)"
            elif file == "README.md":
                has_readme = True
                file_info += " (readme)"
            elif file.endswith("dataset_info.json"):
                config_files.append(file)
                file_info += " (config)"
                
            print(file_info)
        
        if not has_dataset_files:
            print("\n⚠️ Warning: No dataset files (.parquet or .arrow) found!")
        
        if not has_readme:
            print("\n⚠️ Warning: No README.md found!")
            
        # Try to get available configs
        try:
            config_names = get_dataset_config_names(args.dataset)
            if config_names:
                print(f"\nAvailable configurations: {', '.join(config_names)}")
            else:
                print("\n⚠️ Warning: No configurations found!")
        except Exception as e:
            print(f"\n⚠️ Warning: Could not get configurations: {e}")
            
            # Try to manually examine config files
            if config_files:
                for config_file in config_files:
                    try:
                        config_path = hf_hub_download(
                            repo_id=args.dataset,
                            repo_type="dataset",
                            filename=config_file
                        )
                        
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                            print(f"\nConfig details from {config_file}:")
                            print(f"  • Version: {config.get('version', 'N/A')}")
                            splits = config.get('splits', {})
                            if splits:
                                for split_name, split_info in splits.items():
                                    print(f"  • Split '{split_name}': {split_info.get('num_examples', 'N/A')} examples")
                    except Exception as config_e:
                        print(f"  • Could not read {config_file}: {config_e}")
                    
        # Try to load the dataset
        print(f"\nAttempting to load dataset...")
        try:
            # First try all configs
            try:
                config_names = get_dataset_config_names(args.dataset)
            except:
                config_names = ["default"]  # Try default if can't get configs
                
            loaded = False
            for config in config_names:
                try:
                    print(f"Trying to load with config '{config}'...")
                    ds = load_dataset(args.dataset, config, split="train")
                    loaded = True
                    print(f"✅ Success! Dataset loaded with config '{config}'")
                    print(f"Dataset contains {len(ds)} examples")
                    
                    # Print the first example's structure
                    if len(ds) > 0:
                        print("\nFirst example structure:")
                        example = ds[0]
                        for k, v in sorted(example.items()):
                            if isinstance(v, (str, int, float, bool)):
                                value = str(v)
                                if len(value) > 100:
                                    value = value[:97] + "..."
                                print(f"  • {k}: {value}")
                            elif isinstance(v, (list, tuple)):
                                print(f"  • {k}: {type(v).__name__} with {len(v)} items")
                            else:
                                print(f"  • {k}: {type(v).__name__}")
                        
                        if args.verbose:
                            # Print available features
                            print("\nDataset features:")
                            for feature_name, feature in sorted(ds.features.items()):
                                print(f"  • {feature_name}: {feature}")
                    break
                except Exception as e:
                    print(f"Could not load dataset with config '{config}': {e}")
            
            if not loaded:
                print("❌ Failed to load the dataset with any configuration.")
                
        except Exception as e:
            print(f"❌ Dataset exists but couldn't be loaded: {e}")
    
    except Exception as e:
        print(f"❌ Dataset could not be found: {e}")
        print("\nPossible reasons:")
        print("  • The dataset doesn't exist")
        print("  • You don't have access to it (if it's private)")
        print("  • Your Hugging Face token isn't set or has expired")
        
if __name__ == "__main__":
    main() 