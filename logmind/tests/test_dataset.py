import pytest
import torch
from logmind.dataset.parser import LogParser
from logmind.dataset.tokenizer import LogTokenizer
from logmind.dataset.dataset import LogDataset, collate_fn

def test_log_parser_parsing():
    parser = LogParser()
    
    # Test single line parsing
    raw_line = "081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010"
    parsed = parser.parse_line(raw_line)
    
    assert parsed is not None
    assert parsed["date"] == "081109"
    assert parsed["time"] == "203518"
    assert parsed["thread"] == "143"
    assert parsed["level"] == "INFO"
    assert parsed["component"] == "dfs.DataNode$DataXceiver"
    assert "Receiving block" in parsed["message"]
    
    # Test template extraction (normalization)
    template = parser.get_template(parsed["message"])
    assert "blk_" not in template
    assert "10.250.19.102" not in template
    assert template == "Receiving block <block_id> src: <ip> dest: <ip>"

def test_log_parser_rca_labels():
    parser = LogParser()
    
    # Normal sequences should return class 0
    assert parser.get_rca_label(["Receiving block blk_1"], is_anomaly=False) == 0
    
    # Exception/Write anomalies should map to Class 1 (WriteFailure)
    assert parser.get_rca_label(["writeBlock blk_1 received exception java.io.IOException: Could not read from stream"], is_anomaly=True) == 1
    
    # Timeout anomalies should map to Class 2 (ConnectionTimeout)
    assert parser.get_rca_label(["PacketResponder blk_1 Exception java.net.SocketTimeoutException: timeout"], is_anomaly=True) == 2
    
    # Serving anomalies should map to Class 3 (ServingFailure)
    assert parser.get_rca_label(["Got exception while serving blk_1 to /10.10.10.1"], is_anomaly=True) == 3
    
    # Volume anomalies should map to Class 4 (ReplicaVolumeError)
    assert parser.get_rca_label(["Unexpected error trying to delete block blk_1. BlockInfo not found in volumeMap."], is_anomaly=True) == 4
    
    # Unknown anomalies should map to Class 5 (OtherAnomaly)
    assert parser.get_rca_label(["Some weird unexpected error occurred"], is_anomaly=True) == 5

def test_tokenizer():
    tokenizer = LogTokenizer()
    templates = [
        "Receiving block <block_id> src: <ip> dest: <ip>",
        "BLOCK* NameSystem.allocateBlock: <path><num>. <block_id>",
        "Received block <block_id> of size <num> from /<ip>"
    ]
    
    tokenizer.build_vocab(templates)
    
    # 5 special tokens + 3 custom templates = 8 vocab size
    assert len(tokenizer) == 8
    
    # Test encoding
    seq = ["Receiving block <block_id> src: <ip> dest: <ip>", "Unknown template"]
    encoded = tokenizer.encode(seq, add_special_tokens=True, max_len=10)
    
    # Verify shape and special tokens
    assert len(encoded) == 10
    assert encoded[0] == tokenizer.cls_id
    assert encoded[1] == tokenizer.token2id["Receiving block <block_id> src: <ip> dest: <ip>"]
    assert encoded[2] == tokenizer.unk_id
    assert encoded[3] == tokenizer.sep_id
    assert all(idx == tokenizer.pad_id for idx in encoded[4:])
    
    # Test decoding
    decoded = tokenizer.decode(encoded, skip_special_tokens=False)
    assert decoded[0] == "[CLS]"
    assert decoded[1] == "Receiving block <block_id> src: <ip> dest: <ip>"
    assert decoded[2] == "[UNK]"
    assert decoded[3] == "[SEP]"
    assert decoded[4] == "[PAD]"

def test_dataset_mlm():
    tokenizer = LogTokenizer()
    templates = ["TemplateA", "TemplateB", "TemplateC"]
    tokenizer.build_vocab(templates)
    
    sequences = [["TemplateA", "TemplateB", "TemplateC", "TemplateA"]]
    anomaly_labels = [0]
    rca_labels = [0]
    
    # High probability to guarantee some masking happens
    dataset = LogDataset(
        sequences, anomaly_labels, rca_labels, tokenizer, 
        max_len=8, mlm_probability=0.5, mode="pretrain_mlm"
    )
    
    item = dataset[0]
    
    assert "input_ids" in item
    assert "mlm_labels" in item
    assert "padding_mask" in item
    
    input_ids = item["input_ids"]
    mlm_labels = item["mlm_labels"]
    
    # Ensure [CLS] and [SEP] are NOT masked in inputs, and get -100 in labels
    assert input_ids[0] == tokenizer.cls_id
    assert mlm_labels[0] == -100
    
    # For padding tokens, labels should be -100 and inputs should be pad_id
    assert input_ids[6] == tokenizer.pad_id
    assert mlm_labels[6] == -100
    
    # Check that masked positions have correct label values (original token IDs)
    masked_indices = (mlm_labels != -100)
    assert masked_indices.any()
    
    for idx in torch.nonzero(masked_indices).squeeze(1).tolist():
        original_token_id = mlm_labels[idx].item()
        # Original token ID must be in our vocab (TemplateA/B/C)
        assert original_token_id in [tokenizer.token2id["TemplateA"], tokenizer.token2id["TemplateB"], tokenizer.token2id["TemplateC"]]

def test_dataset_clm():
    tokenizer = LogTokenizer()
    templates = ["TemplateA", "TemplateB", "TemplateC"]
    tokenizer.build_vocab(templates)
    
    sequences = [["TemplateA", "TemplateB", "TemplateC"]]
    anomaly_labels = [0]
    rca_labels = [0]
    
    dataset = LogDataset(
        sequences, anomaly_labels, rca_labels, tokenizer, 
        max_len=8, mode="pretrain_clm"
    )
    
    item = dataset[0]
    input_ids = item["input_ids"]
    clm_labels = item["clm_labels"]
    
    # Sequence encodes as: [CLS], TempA, TempB, TempC, [SEP], [PAD], [PAD], [PAD]
    # Inputs:
    assert input_ids[0] == tokenizer.cls_id
    assert input_ids[1] == tokenizer.token2id["TemplateA"]
    assert input_ids[2] == tokenizer.token2id["TemplateB"]
    assert input_ids[3] == tokenizer.token2id["TemplateC"]
    assert input_ids[4] == tokenizer.sep_id
    assert input_ids[5] == tokenizer.pad_id
    
    # Targets should be shifted by 1:
    # Index 0 target: TempA
    # Index 1 target: TempB
    # Index 2 target: TempC
    # Index 3 target: [SEP]
    # Index 4 target: -100 (since it's SEP itself predicting nothing, or we stop predicting)
    assert clm_labels[0] == tokenizer.token2id["TemplateA"]
    assert clm_labels[1] == tokenizer.token2id["TemplateB"]
    assert clm_labels[2] == tokenizer.token2id["TemplateC"]
    assert clm_labels[3] == tokenizer.sep_id
    assert clm_labels[4] == -100
    assert clm_labels[5] == -100

def test_collate_fn():
    batch = [
        {
            "input_ids": torch.tensor([1, 2, 3]),
            "padding_mask": torch.tensor([True, True, False]),
            "anomaly_label": torch.tensor(0.0),
            "rca_label": torch.tensor(0),
            "mlm_labels": torch.tensor([-100, 2, -100])
        },
        {
            "input_ids": torch.tensor([4, 5, 6]),
            "padding_mask": torch.tensor([True, True, True]),
            "anomaly_label": torch.tensor(1.0),
            "rca_label": torch.tensor(2),
            "mlm_labels": torch.tensor([4, -100, 6])
        }
    ]
    
    collated = collate_fn(batch)
    
    assert collated["input_ids"].shape == (2, 3)
    assert collated["padding_mask"].shape == (2, 3)
    assert collated["anomaly_label"].shape == (2,)
    assert collated["rca_label"].shape == (2,)
    assert collated["mlm_labels"].shape == (2, 3)
    
    assert torch.equal(collated["anomaly_label"], torch.tensor([0.0, 1.0]))
    assert torch.equal(collated["rca_label"], torch.tensor([0, 2]))
