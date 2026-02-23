import os
import sys
from tqdm import tqdm
from ipatok import tokenise
from glob import glob
from lhotse import CutSet
from lhotse.shar.writers import SharWriter
from pathlib import Path
import logging
import sentencepiece as spm

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
    datefmt="%Y/%b/%d %H:%M:%S",
    stream=sys.stdout)


filelist = glob('/scratch/lingjzhu_root/lingjzhu1/lingjzhu/compressed_datasets/**/*.jsonl.gz',recursive=True)



inpath = '/scratch/lingjzhu_root/lingjzhu1/lingjzhu/compressed_datasets'
outpath = '/scratch/lingjzhu_root/lingjzhu1/lingjzhu/train_datasets'

if not os.path.exists(outpath):
    os.mkdir(outpath)

datasets = [file.replace(inpath,'') for file in filelist]
datasets = [file.replace(os.path.basename(file),'') for file in datasets]
datasets = list(set(datasets))
datasets = [file for file in datasets if 'dev' not in file and 'test' not in file and 'doreco' not in file and 'magicdata' not in file]
print(datasets)
logging.info("%s speech train data files found!"%len(datasets))

logging.info("Beginning processing dataset")

sp = spm.SentencePieceProcessor()
sp.load("ipa_simplified/unigram_127.model")


data_dir = Path(outpath)
data_dir.mkdir(parents=True, exist_ok=True)
with SharWriter(data_dir, fields={"recording": "flac"}, shard_size=20000) as writer:
    
    for i,dataset in enumerate(datasets):

        data_path = inpath+dataset
        logging.info("Processing %s"%data_path)

        supervision = sorted(glob(os.path.join(data_path,'cuts*')))
        recording =  sorted(glob(os.path.join(data_path,'recording*')))
        assert len(supervision)==len(recording)

        logging.info("%s shards found"%len(supervision))

        cuts = CutSet.from_shar(
                    {
                        "cuts": supervision,
                        "recording": recording
                    }
                )


        for cut in tqdm(cuts):
            if cut.duration >= 1.0 and cut.duration <= 24.0:
                cut.supervisions[0].text = cut.supervisions[0].text.replace(" ",'').replace("ɡ","g").replace("ɛ̈","ɛ").replace("j̈","j").replace("ü","u").replace("k̈","k").replace("ɔ̈","ɔ").replace("ʏ̈","ʏ").replace("ɑ̈","ɑ").replace("ɥ̈","ɥ").replace("ï","i").replace("ö","o").replace("ɪ̈","ɪ").replace("ÿ","y").replace("ä","a").replace("g̈","g").replace("ə̈","ə").replace("ẅ","w").replace("ø̈","ø").replace("ë","e").replace("j̩̩̩̩","j̩")
                y = sp.encode(cut.supervisions[0].text, out_type=int)
                if len(y)<=512 and len(y)>=5:
                    frames = cut.duration // 0.01
                    T = (frames - 7) // 2 + 1
                    if 0.9*T >= len(y):
                        if cut.recording.load_audio().shape[0]==1:
                            writer.write(cut)

        logging.info("Processing done! %s datasets remaining."%(len(datasets)-i-1))
