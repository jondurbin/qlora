import argparse
import gc
import glob
import random
import multiprocessing
from copy import deepcopy

import datasets
from loguru import logger
from tqdm import tqdm
from transformers import AutoTokenizer, AddedToken

IGNORE_INDEX = -100
MAX_LENGTH = 4094


def tokenize(item):
    """Tokenize an item."""
    tokenizer = tokenize.tokenizer
    max_length = tokenize.max_length
    prompt = tokenizer(
        item["input"],
        max_length=max_length,
        truncation=True,
        add_special_tokens=False,
    )
    response = tokenizer(
        item["output"],
        max_length=max_length,
        truncation=True,
        add_special_tokens=False,
    )
    input_ids = [tokenizer.bos_token_id] + prompt.input_ids
    response_ids = response.input_ids
    if len(input_ids) + len(response_ids) > max_length - 1:
        if len(prompt.input_ids) <= 512:
            input_ids = input_ids[0 : max_length - len(input_ids)]
        elif len(response_ids) <= 512:
            input_ids = input_ids[0 : max_length - len(response_ids)]
        else:
            input_ids = input_ids[0 : max_length // 2]
            response_ids = response_ids[0 : max_length // 2]
    inputs = input_ids + response_ids
    inputs.append(tokenizer.eos_token_id)
    response_ids.append(tokenizer.eos_token_id)
    labels = [IGNORE_INDEX for _ in range(len(input_ids))] + response_ids
    return {
        "input_ids": inputs,
        "attention_mask": [1 if _id != IGNORE_INDEX else 0 for _id in labels],
        "labels": labels,
        "length": len(inputs),
    }


def init_worker(func, model, max_length):
    """Initialize each multiprocessing worker."""
    func.max_length = max_length
    func.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    if not func.tokenizer.pad_token_id:
        if func.tokenizer.unk_token_id:
            func.tokenizer.pad_token_id = func.tokenizer.unk_token_id
        else:
            func.tokenizer.pad_token_id = func.tokenizer.eos_token_id

    # Accomodate chatml's snowflakeness.
    func.tokenizer.add_tokens(
        [
            AddedToken("<|im_start|>", special=True, normalized=False),
            AddedToken("<|im_end|>", special=True, normalized=False),
        ]
    )


def main():
    """Main."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-processes", type=int, default=multiprocessing.cpu_count()
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="mistralai/mistral-7b-v0.1",
        help="name of or path to model containing the tokenizer to use",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="tokenized",
        help="prefix given to the individual batches of tokenized data (intermediate parquet files)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the parquet dataset to pack.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=4094,
        help="Max context size to use.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=100000, help="Batch size, to reduce peak RAM"
    )
    args = parser.parse_args()

    # First, we'll tokenize the dataset.
    with multiprocessing.Pool(
        args.num_processes,
        initializer=init_worker,
        initargs=(tokenize, args.tokenizer, args.max_length),
    ) as pool:
        dataset = datasets.Dataset.from_parquet(args.dataset)
        indices = list(range(len(dataset)))
        batches = [
            indices[i : i + args.batch_size]
            for i in range(0, len(indices), args.batch_size)
        ]
        for idx in range(len(batches)):
            batch = dataset.select(batches[idx])
            logger.info(len(batch))
            logger.info(
                f"Processing batch {idx + 1} of {len(batches)} [{(idx + 1) * args.batch_size} / {len(dataset)}]..."
            )
            datasets.Dataset.from_list(
                list(tqdm(pool.imap(tokenize, batch), total=len(batch)))
            ).to_parquet(f"{args.prefix}-{idx}.parquet")
            batch = None
            gc.collect()

    # Combine the entire tokenized dataset.
    tokenized_data = datasets.concatenate_datasets(
        [
            datasets.Dataset.from_parquet(path)
            for path in glob.glob(f"{args.prefix}-*.parquet")
        ]
    )
    tokenized_data.to_parquet(f"{args.prefix}-combined.parquet")

    # We'll use k-bucket first-fit algo to somewhat efficiently pack samples.
    packed = []
    temp_batches = [
        {"input_ids": [], "labels": [], "attention_mask": [], "length": 0, "index": 0}
        for _ in range(16)
    ]
    total_length = 0

    # Reload the tokenizer on the main process.
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.unk_token_id
        if tokenizer.unk_token_id:
            tokenizer.pad_token_id = tokenizer.unk_token_id
        else:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    # Start packing...
    for result in tqdm(tokenized_data):
        total_length += result["length"]
        if result["length"] >= args.max_length:
            packed.append(result)
            continue
        fit_batch = 0
        largest_batch = -1
        largest_batch_idx = 0
        indices = list(range(len(temp_batches)))
        random.shuffle(indices)
        for idx in indices:
            batch = temp_batches[idx]
            if batch["length"] and batch["length"] > largest_batch:
                largest_batch = batch["length"]
                largest_batch_idx = idx
            if batch["length"] + result["length"] <= args.max_length:
                batch["input_ids"] += result["input_ids"]
                batch["labels"] += result["labels"]
                batch["attention_mask"] += [
                    val * (batch["index"] + 1) for val in result["attention_mask"]
                ]
                batch["length"] += result["length"]
                batch["index"] += 1
                fit_batch = True
                break
        if fit_batch:
            continue
        if largest_batch == -1:
            logger.warning(f"Had to pack early: {result['length']}")
            packed.append(result)
            continue
        packed.append(deepcopy(temp_batches[largest_batch_idx]))
        temp_batches[largest_batch_idx] = {
            "input_ids": result["input_ids"],
            "labels": result["labels"],
            "attention_mask": result["attention_mask"],
            "length": result["length"],
            "index": 0,
        }
    for batch in temp_batches:
        logger.info("Finishing batch...")
        if batch["length"]:
            packed.append(batch)

    # Done!
    total_packed_length = sum([item["length"] for item in packed])
    logger.info(f"Total length: {total_length}")
    logger.info(f"Packd length: {total_packed_length}")
    logger.info(f"Packed item count: {len(packed)}")
    datasets.Dataset.from_list(packed).to_parquet(
        "packed-bagel-input-output-v0.4.parquet"
    )


if __name__ == "__main__":
    main()
