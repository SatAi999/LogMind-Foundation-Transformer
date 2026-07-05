import re
import os
import logging
from typing import Dict, List, Tuple, Optional, Set

logger = logging.getLogger(__name__)

class LogParser:
    """
    Custom log parser for HDFS logs.
    Extracts structured fields (date, time, thread, level, component, message),
    replaces variable components with wildcard placeholders to form event templates,
    groups logs by block ID, and labels the root cause of anomalous sequences.
    """
    def __init__(self):
        # Compiled regexes for field extraction
        self.field_pattern = re.compile(
            r"^(\d{6})\s+(\d{6})\s+(\d+)\s+(INFO|WARN|ERROR|FATAL|DEBUG)\s+([^:]+):\s*(.*)$"
        )
        self.block_id_pattern = re.compile(r"(blk_[-]?\d+)")
        
        # Compiled regexes for parameter normalization (template extraction)
        self.norm_patterns = [
            (re.compile(r"blk_[-]?\d+"), "<block_id>"),
            (re.compile(r"/\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d+"), "<ip>"),
            (re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"), "<ip>"),
            (re.compile(r"\b\d+\b"), "<num>"),
            (re.compile(r"/[a-zA-Z0-9_\-\./]+"), "<path>"),
            (re.compile(r"0x[0-9a-fA-F]+"), "<hex>"),
        ]

        # Root Cause Analysis Categories definition
        # Class 0: Normal
        # Class 1: WriteFailure (writeBlock, Could not read, EOFException)
        # Class 2: ConnectionTimeout (SocketTimeout, InterruptedIO, Connection reset, Broken pipe)
        # Class 3: ServingFailure (Got exception while serving, Failed to transfer)
        # Class 4: ReplicaVolumeError (BlockInfo not found in volumeMap, delete error, invalidSet)
        # Class 5: OtherAnomaly (Unclassified anomalies)
        self.rca_keywords = {
            1: ["writeblock", "could not read from stream", "eofexception"],
            2: ["sockettimeoutexception", "interruptedioexception", "connection reset by peer", "broken pipe"],
            3: ["got exception while serving", "failed to transfer"],
            4: ["blockinfo not found in volumemap", "cannot be written to", "invalidset", "unexpected error trying to delete"]
        }

    def parse_line(self, line: str) -> Optional[Dict[str, str]]:
        """
        Parses a single log line into structural fields.
        """
        match = self.field_pattern.match(line.strip())
        if not match:
            return None
        date, time, thread, level, component, message = match.groups()
        return {
            "date": date,
            "time": time,
            "thread": thread,
            "level": level,
            "component": component,
            "message": message
        }

    def get_template(self, message: str) -> str:
        """
        Replaces parameters in a log message with placeholders to produce a generic template.
        """
        template = message
        for pattern, replacement in self.norm_patterns:
            template = pattern.sub(replacement, template)
        return template.strip()

    def get_rca_label(self, sequence_messages: List[str], is_anomaly: bool) -> int:
        """
        Determines the sequence-level Root Cause Analysis (RCA) category.
        Returns 0 for Normal, and 1-5 for various anomaly classes.
        """
        if not is_anomaly:
            return 0
        
        # Check messages for failure indicators
        for msg in sequence_messages:
            msg_lower = msg.lower()
            # Check keywords in order of priority
            for category_id, keywords in self.rca_keywords.items():
                if any(kw in msg_lower for kw in keywords):
                    return category_id
                    
        return 5  # OtherAnomaly

    def parse_file(
        self, 
        log_path: str, 
        label_path: str, 
        max_lines: Optional[int] = None
    ) -> Tuple[Dict[str, List[str]], Dict[str, int], Dict[str, int]]:
        """
        Parses the log file, maps lines to block IDs, and labels them using anomaly_labels.
        Returns:
            block_sequences: Dict mapping block_id to a list of event templates.
            block_labels: Dict mapping block_id to anomaly label (0: Normal, 1: Anomaly).
            block_rca: Dict mapping block_id to RCA label (0-5).
        """
        logger.info("Loading labels from %s...", label_path)
        import pandas as pd
        df_labels = pd.read_csv(label_path)
        
        # Build maps for labels
        raw_labels = dict(zip(df_labels["BlockId"], df_labels["Label"]))
        anomaly_blocks = {k for k, v in raw_labels.items() if v == "Anomaly"}
        
        block_sequences: Dict[str, List[str]] = {}
        block_raw_messages: Dict[str, List[str]] = {}
        
        logger.info("Parsing logs from %s...", log_path)
        lines_processed = 0
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                lines_processed += 1
                if max_lines and lines_processed > max_lines:
                    break
                
                # Check for block ID in the log line
                blk_match = self.block_id_pattern.search(line)
                if not blk_match:
                    continue
                blk_id = blk_match.group(1)
                
                # If block_id is not in our known label set, skip it to avoid unlabelled data
                if blk_id not in raw_labels:
                    continue
                
                # Parse structured fields
                parsed = self.parse_line(line)
                if not parsed:
                    # If regex fails, fallback to cleaning the line directly
                    msg = line.strip()
                else:
                    msg = parsed["message"]
                
                template = self.get_template(msg)
                
                if blk_id not in block_sequences:
                    block_sequences[blk_id] = []
                    block_raw_messages[blk_id] = []
                    
                block_sequences[blk_id].append(template)
                block_raw_messages[blk_id].append(msg)
                
                if lines_processed % 1000000 == 0:
                    logger.info("Processed %d lines...", lines_processed)
                    
        # Apply labeling
        block_labels: Dict[str, int] = {}
        block_rca: Dict[str, int] = {}
        
        for blk_id, seq in block_sequences.items():
            is_anomaly = blk_id in anomaly_blocks
            block_labels[blk_id] = 1 if is_anomaly else 0
            block_rca[blk_id] = self.get_rca_label(block_raw_messages[blk_id], is_anomaly)
            
        logger.info(
            "Parsed %d sequences. Anomalies: %d. Normals: %d.",
            len(block_sequences),
            sum(block_labels.values()),
            len(block_labels) - sum(block_labels.values())
        )
        
        return block_sequences, block_labels, block_rca
