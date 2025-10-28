import os
import logging
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk
from tokenizers import models, normalizers, pre_tokenizers, trainers, Tokenizer, processors

from comm import Logger

class TatoebaDataset(Dataset):
    def __init__(self, language_pair, source_lang=None, target_lang=None, split='train'):

        super().__init__()
        self.logger = Logger("dataset.TatoebaDataset", level = logging.DEBUG)

        self.support_language_pair = ["fr-en", "cs-en", "de-en", "hi-en", "ru-en"]
        if language_pair not in self.support_language_pair:
            self.logger.error(f"not support language pair: {language_pair}, support list: {self.support_language_pair}")
            raise ValueError(f"not support language pair: {language_pair}, support list: {self.support_language_pair}")

        self.language_pair = language_pair
        lang_a, lang_b = self.language_pair.split('-')

        if source_lang is None and target_lang is None:
            self.source_lang = lang_a
            self.target_lang = lang_b
        else:
            if not (source_lang == lang_a and target_lang == lang_b) and not (source_lang == lang_b and target_lang == lang_a):
                self.logger.error(f"please check source_lang and target_lang, they should be ({lang_a}, {lang_b}) or ({lang_b}, {lang_a})")
                raise ValueError(f"please check source_lang and target_lang, they should be ({lang_a}, {lang_b}) or ({lang_b}, {lang_a})")
            self.source_lang = source_lang
            self.target_lang = target_lang

        self.seq_length = 10  # hardcode
        self.split = split
        hf_split = 'validation' if split == 'valid' else 'train'

        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        local_dataset_path = os.path.join(self.current_dir, "local", f"tatoeba_dataset_{language_pair}_{split}")
        if os.path.exists(local_dataset_path):
            self.logger.info(f"Loading dataset from local disk: {local_dataset_path}")
            self.dataset = load_from_disk(local_dataset_path)

            self.source_tokenizer = self._get_tokenizer(self.source_lang)
            self.target_tokenizer = self._get_tokenizer(self.target_lang)
        else:
            self.logger.debug("Loading dataset...")
            self.dataset = load_dataset("wmt14", self.language_pair, split=hf_split)
            self.logger.debug(f"Load dataset done, <{split}> samples count(before filter): {len(self.dataset)}") # type: ignore

            self.source_tokenizer = self._get_tokenizer(self.source_lang)
            self.target_tokenizer = self._get_tokenizer(self.target_lang)

            self.dataset = self.dataset.map(self._tokenize_function, batched=True, remove_columns=["translation"])
            self.dataset = self.dataset.filter(lambda x: len(x['input_ids']) > 0 and len(x['labels']) > 0)
            
            self.logger.info(f"Saving processed dataset to {local_dataset_path}")
            os.makedirs(local_dataset_path, exist_ok=True)
            self.dataset.save_to_disk(local_dataset_path) # type: ignore

        self.num_samples = len(self.dataset) # type: ignore
        self.source_vocab_size = self.source_tokenizer.get_vocab_size()
        self.target_vocab_size = self.target_tokenizer.get_vocab_size()
        self.pad_token_id = self.source_tokenizer.token_to_id("<pad>")
        self.logger.info(f"<{language_pair}:{split}> samples count: {self.num_samples}, source vocab size: {self.source_vocab_size}, target vocab size: {self.target_vocab_size}")

    def _tokenize_function(self, examples):
        source_texts = [ex[self.source_lang] for ex in examples["translation"]]
        target_texts = [ex[self.target_lang] for ex in examples["translation"]]
        source_encodings = self.source_tokenizer.encode_batch_fast(source_texts)
        target_encodings = self.target_tokenizer.encode_batch_fast(target_texts)

        return {
            "input_ids": [encoding.ids for encoding in source_encodings],
            "labels": [encoding.ids for encoding in target_encodings],
        }

    def _get_tokenizer(self, lang):
        local_tokenizer_path = os.path.join(self.current_dir, "local", f"tatoeba_tokenizer_{lang}.json")

        if(os.path.exists(local_tokenizer_path)):
            self.logger.debug(f"Load tokenizer from local file: {local_tokenizer_path}")
            return Tokenizer.from_file(local_tokenizer_path)
        
        self.logger.debug(f"Training tokenizer for <{lang}>")
        tokenizer = Tokenizer(models.WordLevel(vocab = None, unk_token = "<unk>"))
        tokenizer.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]) # type: ignore
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace() # type: ignore
        trainer = trainers.WordLevelTrainer(special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"], min_frequency = 5)

        def get_training_corpus():
            batch_size = 10
            for i in range(0, len(self.dataset), batch_size): # type: ignore
                batch = self.dataset[i : i + batch_size] # type: ignore
                texts = [item[lang] for item in batch['translation']]
                yield texts


        tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

        pad_token_id = tokenizer.token_to_id("<pad>")
        cls_token_id = tokenizer.token_to_id("<bos>")
        sep_token_id = tokenizer.token_to_id("<eos>")
        
        tokenizer.enable_padding(pad_id = pad_token_id, pad_token = "<pad>", length = self.seq_length)
        tokenizer.enable_truncation(max_length = self.seq_length)

        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"<bos>:0 $A:0 <eos>:0",
            special_tokens=[("<bos>", cls_token_id), ("<eos>", sep_token_id)],
        ) # type: ignore

        tokenizer.save(local_tokenizer_path)
        
        self.logger.debug(f"Train tokenizer for <{lang}> done, vocab size {tokenizer.get_vocab_size()}, saved to {local_tokenizer_path}")
        return tokenizer

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        example = self.dataset[idx]
        input_ids = torch.tensor(example["input_ids"], dtype=torch.long)
        labels = torch.tensor(example["labels"], dtype=torch.long)
        return input_ids, labels
    
if __name__ == "__main__":

    # datasets = TatoebaDataset("fr-en", "en", "fr")


    local_tokenizer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local", f"tatoeba_tokenizer_fr.json")
    tokenizer = Tokenizer.from_file(local_tokenizer_path)

    result = tokenizer.encode("Je déclare reprise la session du Parlement européen qui avait été interrompue le vendredi 17 décembre dernier et je vous renouvelle tous mes vux en espérant que vous avez passé de bonnes vacances.")
    
    print(f"result.ids: {result.ids}")
    print(f"result.type_ids: {result.type_ids}")
    print(f"result.tokens: {result.tokens}")
    print(f"result.offsets: {result.offsets}")
    print(f"result.attention_mask: {result.attention_mask}")
    print(f"result.special_tokens_mask: {result.special_tokens_mask}")
    print(f"result.overflowing: {result.overflowing}")
    

