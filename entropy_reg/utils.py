"""Funções utilitárias para formatação e diagnóstico."""


def fmt_bytes(b):
    """Formata um tamanho em bytes para uma unidade legível (B, KB, MB, GB, TB)."""
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if b < 1024 or u == "TB":
            return f"{b:.0f} {u}"
        b /= 1024
