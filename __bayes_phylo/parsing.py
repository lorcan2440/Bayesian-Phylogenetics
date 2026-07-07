from __future__ import annotations

from dataclasses import dataclass

NUCLEOTIDES = ("A", "C", "G", "T")
NUC_TO_STATE = {base: i for i, base in enumerate(NUCLEOTIDES)}


@dataclass(frozen=True)
class SequenceData:
    taxa: tuple[str, ...]
    sequences: tuple[str, ...]
    encoded: tuple[tuple[int, ...], ...]
    n_taxa: int
    n_sites: int

    def as_dict(self) -> dict[str, str]:
        return {name: seq for name, seq in zip(self.taxa, self.sequences)}


def parse_sequences(raw_text: str) -> SequenceData:
    """Parse aligned nucleotide sequences from FASTA or line records.

    Supported formats:
    - FASTA:
      >taxon_1
      ACGT...
      >taxon_2
      ACGT...
    - One line per taxon:
      taxon_1: ACGT...
      taxon_2 ACGT...
    """
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("No sequence input provided.")

    if any(line.startswith(">") for line in lines):
        records = _parse_fasta_records(lines)
    else:
        records = _parse_line_records(lines)

    _validate_records(records)

    taxa = tuple(records.keys())
    seqs = tuple(records[name] for name in taxa)
    encoded = tuple(tuple(NUC_TO_STATE[ch] for ch in seq) for seq in seqs)
    return SequenceData(
        taxa=taxa,
        sequences=seqs,
        encoded=encoded,
        n_taxa=len(taxa),
        n_sites=len(seqs[0]),
    )


def _parse_fasta_records(lines: list[str]) -> dict[str, str]:
    records: dict[str, str] = {}
    current_name: str | None = None
    seq_chunks: list[str] = []

    for line in lines:
        if line.startswith(">"):
            if current_name is not None:
                if not seq_chunks:
                    raise ValueError(f"Taxon '{current_name}' has an empty sequence.")
                records[current_name] = "".join(seq_chunks)
            current_name = line[1:].strip()
            if not current_name:
                raise ValueError("Encountered FASTA header without a taxon name.")
            if current_name in records:
                raise ValueError(f"Duplicate taxon name: '{current_name}'.")
            seq_chunks = []
        else:
            if current_name is None:
                raise ValueError("Sequence line appears before first FASTA header ('>').")
            seq_chunks.append(line.replace(" ", ""))

    if current_name is None:
        raise ValueError("No FASTA records found.")
    if not seq_chunks:
        raise ValueError(f"Taxon '{current_name}' has an empty sequence.")
    records[current_name] = "".join(seq_chunks)
    return records


def _parse_line_records(lines: list[str]) -> dict[str, str]:
    records: dict[str, str] = {}
    for i, line in enumerate(lines, start=1):
        if ":" in line:
            name, seq = line.split(":", 1)
        else:
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(
                    f"Line {i} is not valid. Use 'name: SEQUENCE' or 'name SEQUENCE'."
                )
            name, seq = parts
        name = name.strip()
        seq = seq.strip().replace(" ", "")
        if not name:
            raise ValueError(f"Line {i} is missing a taxon name.")
        if name in records:
            raise ValueError(f"Duplicate taxon name: '{name}'.")
        records[name] = seq
    return records


def _validate_records(records: dict[str, str]) -> None:
    if len(records) < 3:
        raise ValueError("At least 3 taxa are required for phylogenetic inference.")

    lengths = {len(seq) for seq in records.values()}
    if 0 in lengths:
        raise ValueError("All sequences must be non-empty.")
    if len(lengths) != 1:
        raise ValueError("All sequences must have exactly the same length (aligned input).")

    allowed = set(NUCLEOTIDES)
    for name, seq in list(records.items()):
        seq_upper = seq.upper()
        invalid = sorted(set(seq_upper) - allowed)
        if invalid:
            invalid_str = ", ".join(invalid)
            raise ValueError(
                f"Taxon '{name}' contains invalid nucleotide(s): {invalid_str}. "
                "Allowed symbols are A, C, G, T."
            )
        records[name] = seq_upper

