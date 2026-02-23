import os
import sys
import argparse
from tqdm import tqdm
from glob import glob
from lhotse import CutSet
from lhotse.shar.writers import SharWriter
from pathlib import Path
import logging
import sentencepiece as spm
import pandas as pd
from collections import defaultdict
from ast import literal_eval

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
    datefmt="%Y/%b/%d %H:%M:%S",
    stream=sys.stdout)

sp = spm.SentencePieceProcessor()
sp.load("ipa_simplified/unigram_127.model")

def get_results(path):
    results = defaultdict(dict)
    with open(path,'r') as f:
        for line in f.readlines():
            idx, pred = line.split('\t')
            category = 'hyp' if 'hyp=' in pred else 'ref'
            pred = pred.replace('hyp=','').replace('ref=','')
            pred = ''.join(list(literal_eval(pred)))
            results[idx][category] = pred
    return results
    

# Define the argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Process a dataset into SHAR format.")
    
    # Input and output paths
    parser.add_argument('--inpath', type=str, default='/scratch/lingjzhu_root/lingjzhu1/lingjzhu/unsup_train', help="Input path where the dataset is located.")
    parser.add_argument('--outpath', type=str, default='/scratch/lingjzhu_root/lingjzhu1/lingjzhu/unsup_train_filtered', help="Output path to save the filtered data.")
    parser.add_argument('--metrics_path', type=str, default='/scratch/lingjzhu_root/lingjzhu1/lingjzhu/unsup_data_comparison_all_processed',help="Input path for the processed data csv.")
    parser.add_argument("--percentile_between_models",type=float,default=19.3090)
    parser.add_argument("--transcription_path",type=str,default='/scratch/lingjzhu_root/lingjzhu1/lingjzhu/zipformer_exp/zipformer_large_crctc/unsup_data_decoded')
    
    return parser.parse_args()

# Main processing script
def main():
    # Parse command-line arguments
    args = parse_args()

    logging.info("Scanning dataset")
    filelist = [i for i in os.listdir(args.inpath) if 'jsonl.gz' in i]

    datasets = [file.replace('.jsonl.gz','').replace('cuts.','') for file in filelist]
    datasets = list(set(datasets))

    logging.info(f"{len(datasets)} speech data files found!")
    logging.info(f"Here are the datasets: {datasets}")
    
    # Create output path if it doesn't exist
    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)

    logging.info("Beginning processing dataset")

    # Process each dataset
    with SharWriter(args.outpath, fields={"recording": "flac"}, shard_size=20000) as writer:
        for i, dataset in enumerate(datasets):
            logging.info(f"Processing {dataset}")
            
            cuts_path = os.path.join(args.inpath,f"cuts.{dataset}.jsonl.gz")
            recording_path =  os.path.join(args.inpath,f"recording.{dataset}.tar")
            cuts = CutSet.from_shar(
                {
                    "cuts": [cuts_path],
                    "recording": [recording_path]
                }
            )
            logging.info(f"Retrieving data from {cuts_path} and {recording_path}")
            
            df = pd.read_csv(os.path.join(args.metrics_path,f"{dataset}_processed_processed.csv"))
            condition = (df['mean_model_vs_model_pfer'] < args.percentile_between_models)
            final_metrics = df[condition]
            final_samples = set([i.replace(':','') for i in final_metrics['id'].tolist()])
            logging.info(f"{len(final_samples)} samples are selected!")

            transcripts = get_results(os.path.join(args.transcription_path,f"recogs-{dataset}-iter-500000-avg-10-use-averaged-model.txt"))
            #print(transcripts)
            
            for cut in tqdm(cuts):
                if cut.id in final_samples:
                    cut.supervisions[0].text = transcripts[cut.id+':']['hyp']
                    y = sp.encode(cut.supervisions[0].text, out_type=int)
                    if len(y)<=512 and len(y)>=5:
                        frames = cut.duration // 0.01
                        T = (frames - 7) // 2 + 1
                        if 0.85*T >= len(y):
                            writer.write(cut)

            logging.info(f"Processing done! {len(datasets) - i - 1} datasets remaining.")

if __name__ == "__main__":
    main()
