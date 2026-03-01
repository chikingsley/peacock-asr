#!/usr/bin/env python3
"""
Cleanup Script for Unused Voice Recordings

This script identifies and removes orphaned voice recordings that were likely 
discarded during the recording process but not properly deleted.

It works by comparing all .wav files in a speaker's directory against the recordings 
listed in the metadata files (dataset_info.json, metadata.txt, transcripts.json).
Any .wav file that is not referenced in any metadata file is considered orphaned
and can be safely deleted.

Usage:
    python cleanup_unused_recordings.py --speaker "YourName" [--dry-run] [--force]

Options:
    --speaker       Name of the speaker whose recordings to clean up
    --dry-run       Run without deleting files, just print what would be deleted
    --force         Skip confirmation before deleting files
"""

import os
import json
import argparse
import glob
from colorama import Fore, Style, init as colorama_init

# Initialize colorama
colorama_init()

def parse_metadata_files(speaker_dir):
    """
    Parse all metadata files to get the list of valid recordings.
    Returns a set of filenames that are referenced in any metadata file.
    """
    referenced_files = set()
    
    # Check dataset_info.json (most comprehensive)
    dataset_info_path = os.path.join(speaker_dir, "dataset_info.json")
    if os.path.exists(dataset_info_path):
        try:
            with open(dataset_info_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'recordings' in data:
                    for recording in data['recordings']:
                        if 'path' in recording:
                            # Extract just the filename from the path
                            filename = os.path.basename(recording['path'])
                            referenced_files.add(filename)
            print(f"{Fore.GREEN}Found {len(referenced_files)} referenced files in dataset_info.json{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error reading dataset_info.json: {e}{Style.RESET_ALL}")
    
    # Check metadata.txt (pipe-delimited format)
    metadata_txt_path = os.path.join(speaker_dir, "metadata.txt")
    if os.path.exists(metadata_txt_path):
        try:
            with open(metadata_txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('#'):  # Skip comment lines
                        continue
                    parts = line.strip().split('|')
                    if len(parts) >= 1:
                        filename = parts[0]
                        referenced_files.add(filename)
            print(f"{Fore.GREEN}Found {len(referenced_files)} referenced files after checking metadata.txt{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error reading metadata.txt: {e}{Style.RESET_ALL}")
    
    # Check transcripts.json
    transcripts_json_path = os.path.join(speaker_dir, "transcripts.json")
    if os.path.exists(transcripts_json_path):
        try:
            with open(transcripts_json_path, 'r', encoding='utf-8') as f:
                transcripts = json.load(f)
                for filename in transcripts.keys():
                    referenced_files.add(filename)
            print(f"{Fore.GREEN}Found {len(referenced_files)} referenced files after checking transcripts.json{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error reading transcripts.json: {e}{Style.RESET_ALL}")
    
    return referenced_files

def find_orphaned_files(speaker_dir, referenced_files):
    """
    Find all .wav files in the directory that are not referenced in metadata.
    Returns a list of orphaned file paths.
    """
    all_wav_files = glob.glob(os.path.join(speaker_dir, "*.wav"))
    all_wav_filenames = set(os.path.basename(path) for path in all_wav_files)
    
    orphaned_filenames = all_wav_filenames - referenced_files
    orphaned_filepaths = [os.path.join(speaker_dir, filename) for filename in orphaned_filenames]
    
    return orphaned_filepaths

def cleanup(speaker_name, dry_run=False, force=False):
    """
    Main cleanup function to identify and delete orphaned recordings.
    """
    base_dir = "voice_dataset"
    speaker_dir = os.path.join(base_dir, speaker_name)
    
    if not os.path.exists(speaker_dir):
        print(f"{Fore.RED}Error: Speaker directory {speaker_dir} does not exist{Style.RESET_ALL}")
        return
    
    print(f"{Fore.CYAN}Analyzing recordings for speaker: {speaker_name}{Style.RESET_ALL}")
    
    # Get list of valid recordings from metadata
    referenced_files = parse_metadata_files(speaker_dir)
    if not referenced_files:
        print(f"{Fore.YELLOW}Warning: No referenced files found in metadata. No files will be considered valid.{Style.RESET_ALL}")
        if not force:
            confirmation = input(f"{Fore.RED}This could result in deleting ALL wav files. Continue? (y/n): {Style.RESET_ALL}").lower()
            if confirmation != 'y':
                print(f"{Fore.CYAN}Operation cancelled.{Style.RESET_ALL}")
                return
    
    # Find orphaned files
    orphaned_files = find_orphaned_files(speaker_dir, referenced_files)
    
    if not orphaned_files:
        print(f"{Fore.GREEN}No orphaned recordings found. Everything is clean!{Style.RESET_ALL}")
        return
    
    # Report findings
    total_size = sum(os.path.getsize(f) for f in orphaned_files if os.path.exists(f))
    total_size_mb = total_size / (1024 * 1024)
    
    print(f"\n{Fore.YELLOW}Found {len(orphaned_files)} orphaned recordings totaling {total_size_mb:.2f} MB:{Style.RESET_ALL}")
    for i, filepath in enumerate(orphaned_files[:10]):  # Show max 10 examples
        size_kb = os.path.getsize(filepath) / 1024
        print(f"  {i+1}. {os.path.basename(filepath)} ({size_kb:.1f} KB)")
    
    if len(orphaned_files) > 10:
        print(f"  ... and {len(orphaned_files) - 10} more files")
    
    # If dry run, stop here
    if dry_run:
        print(f"\n{Fore.CYAN}DRY RUN: No files were deleted. Run without --dry-run to actually delete files.{Style.RESET_ALL}")
        return
    
    # Confirm deletion
    if not force:
        confirmation = input(f"\n{Fore.YELLOW}Do you want to delete these {len(orphaned_files)} orphaned files? (y/n): {Style.RESET_ALL}").lower()
        if confirmation != 'y':
            print(f"{Fore.CYAN}Operation cancelled.{Style.RESET_ALL}")
            return
    
    # Delete orphaned files
    deleted_count = 0
    deleted_size = 0
    
    for filepath in orphaned_files:
        try:
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                os.remove(filepath)
                deleted_count += 1
                deleted_size += size
                print(f"{Fore.GREEN}Deleted: {os.path.basename(filepath)}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error deleting {os.path.basename(filepath)}: {e}{Style.RESET_ALL}")
    
    deleted_size_mb = deleted_size / (1024 * 1024)
    print(f"\n{Fore.GREEN}Cleanup complete! Deleted {deleted_count} orphaned recordings ({deleted_size_mb:.2f} MB){Style.RESET_ALL}")

def main():
    parser = argparse.ArgumentParser(description='Cleanup unused voice recordings')
    parser.add_argument('--speaker', required=True, help='Name of the speaker whose recordings to clean')
    parser.add_argument('--dry-run', action='store_true', help='Run without deleting files')
    parser.add_argument('--force', action='store_true', help='Skip confirmation before deleting')
    
    args = parser.parse_args()
    
    print(f"\n{Fore.CYAN}===== Voice Dataset Cleanup Tool ====={Style.RESET_ALL}")
    cleanup(args.speaker, args.dry_run, args.force)

if __name__ == '__main__':
    main() 