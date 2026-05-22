#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLUSTERING SÉMANTIQUE DES DOMAINES D'ACTIVITÉ
==============================================

Approche :
  1. Pour chaque field unique, construction d'une phrase enrichie avec
     le contexte des entreprises qui l'ont (Activity, Business Overview,
     Capability) — le modèle comprend ainsi que "C4ISR" et "Command &
     Control" sont proches même si les mots sont différents.
  2. Encodage en vecteurs denses via sentence-transformers (local, ~90MB,
     téléchargé une fois au premier lancement).
  3. DBSCAN sur distance cosinus entre ces vecteurs.
  4. Nom canonique = libellé le plus fréquent dans le cluster.
  5. Les corrections manuelles (page admin) s'appliquent PAR-DESSUS,
     avec priorité absolue — elles ne sont pas gérées ici.

Dépendances :
  pip install sentence-transformers scikit-learn numpy
"""

from collections import Counter
from typing import Dict, List

import numpy as np


# ══════════════════════════════════════════════
#  UTILITAIRES
# ══════════════════════════════════════════════

def split_fields(field_str: str) -> List[str]:
    """Sépare les domaines par / ; , et nettoie les espaces."""
    if not field_str or str(field_str).lower() in ("nan", "none", ""):
        return []
    import re
    parts = re.split(r"[/;,]", str(field_str))
    return [p.strip() for p in parts if p.strip()]


def _build_enriched_text(field_lower: str, df) -> str:
    """
    Construit une phrase enrichie pour un field unique en y ajoutant
    le contexte des entreprises qui l'ont dans la base.

    Exemple de sortie pour "c4isr" :
      "c4isr | Thales, Leonardo, Airbus | develops integrated command and
       control systems for armed forces | C2 systems, ISR, communications"

    Le modèle d'embeddings encode cette phrase en tenant compte du sens
    complet, pas seulement du libellé "c4isr".

    Colonnes utilisées (dans l'ordre de priorité) :
      - Activity          : description courte de l'activité
      - Business Overview : description longue (tronquée à 200 chars)
      - Capability        : capacités techniques déclarées
    """
    parts = [field_lower]  # Libellé du field en tête

    matching_rows = []
    for _, row in df.iterrows():
        raw = str(row.get("Field", ""))
        row_fields = [f.strip().lower() for f in split_fields(raw)]
        if field_lower in row_fields:
            matching_rows.append(row)

    if not matching_rows:
        # Pas d'entreprise associée → on encode le libellé seul
        return field_lower

    # Noms des entreprises (max 5 pour ne pas noyer le signal)
    company_names = [
        str(r.get("Company Name", "")) for r in matching_rows
        if str(r.get("Company Name", "")).strip() not in ("", "nan")
    ]
    if company_names:
        parts.append(", ".join(company_names[:5]))

    # Activités
    activities = [
        str(r.get("Activity", "")).strip() for r in matching_rows
        if str(r.get("Activity", "")).strip() not in ("", "nan")
    ]
    if activities:
        # Dédoublonner et joindre
        unique_activities = list(dict.fromkeys(activities))
        parts.append(" | ".join(unique_activities[:5]))

    # Business Overview (tronqué — les premiers 200 caractères suffisent)
    overviews = [
        str(r.get("Business Overview", "")).strip()[:200] for r in matching_rows
        if str(r.get("Business Overview", "")).strip() not in ("", "nan")
    ]
    if overviews:
        unique_overviews = list(dict.fromkeys(overviews))
        parts.append(" ".join(unique_overviews[:3]))

    # Capabilities
    capabilities = [
        str(r.get("Capability", "")).strip()[:150] for r in matching_rows
        if str(r.get("Capability", "")).strip() not in ("", "nan")
    ]
    if capabilities:
        unique_caps = list(dict.fromkeys(capabilities))
        parts.append(" | ".join(unique_caps[:3]))

    return " | ".join(parts)


# ══════════════════════════════════════════════
#  FONCTION PRINCIPALE
# ══════════════════════════════════════════════

def cluster_fields(all_raw_fields: List[str], df, eps: float = 0.30) -> Dict[str, str]:
    """
    Clustering sémantique des domaines d'activité.

    Args:
        all_raw_fields : liste brute de tous les fields (avec doublons)
                         ex: ["Missile systems", "C4ISR", "Missile systems", ...]
        df             : DataFrame pandas de la feuille DataBase (pour le contexte)
        eps            : seuil de distance cosinus DBSCAN
                         0.20 = strict  |  0.30 = équilibre  |  0.45 = large

    Returns:
        dict {field_title_case → canonical_title_case}
        ex: {"Guided Missiles": "Missile Systems",
             "Guided Weapons":  "Missile Systems",
             "C2 Systems":      "C4Isr"}
    """
    from sklearn.cluster import DBSCAN
    from sklearn.metrics.pairwise import cosine_distances

    # ── 1. Comptage et dédoublonnage ──────────────────────────────────────────
    cleaned = [f.strip() for f in all_raw_fields if f.strip()]
    counts  = Counter([f.lower() for f in cleaned])
    unique_lower = list(set(counts.keys()))

    if len(unique_lower) <= 1:
        return {
            f.strip().title(): unique_lower[0].title() if unique_lower else f.strip().title()
            for f in cleaned
        }

    # ── 2. Construction des phrases enrichies ─────────────────────────────────
    print(f"  🔤 Construction des textes enrichis pour {len(unique_lower)} fields...")
    enriched_texts = []
    for field_lower in unique_lower:
        text = _build_enriched_text(field_lower, df)
        enriched_texts.append(text)

    # ── 3. Encodage sémantique (sentence-transformers) ────────────────────────
    print(f"  🧠 Encodage sémantique via sentence-transformers...")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers n'est pas installé.\n"
            "Exécute : pip install sentence-transformers"
        )

    # all-MiniLM-L6-v2 : bon équilibre vitesse/qualité, 384 dimensions, ~90MB
    # Téléchargé automatiquement au premier lancement, puis mis en cache local.
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(
        enriched_texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # Normalisation L2 → distance cosinus = 1 - dot product
    )

    # ── 4. DBSCAN sur matrice de distances cosinus ────────────────────────────
    # normalize_embeddings=True → cosine_distances = 1 - dot product (plus rapide)
    dist_matrix = cosine_distances(embeddings)

    model_db = DBSCAN(
        eps=eps,
        min_samples=1,        # Chaque point forme au moins son propre cluster
        metric="precomputed",
    )
    labels = model_db.fit_predict(dist_matrix)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = list(labels).count(-1)
    print(f"  📊 DBSCAN (eps={eps}): {len(unique_lower)} domaines "
          f"→ {n_clusters} clusters, {n_noise} outliers")

    # ── 5. Nom canonique = membre le plus fréquent du cluster ─────────────────
    cluster_groups: Dict[int, List[str]] = {}
    for idx, label in enumerate(labels):
        cluster_groups.setdefault(label, []).append(unique_lower[idx])

    canonical_map: Dict[str, str] = {}
    for label, members in cluster_groups.items():
        if label == -1:
            # Outliers (rare avec min_samples=1) → chacun garde son nom
            for m in members:
                canonical_map[m] = m.strip().title()
        else:
            canon = max(members, key=lambda m: counts[m]).strip().title()
            for m in members:
                canonical_map[m] = canon

    # Log des clusters non-triviaux
    for label, members in cluster_groups.items():
        if len(members) > 1:
            canon = canonical_map[members[0]]
            print(f"     → {canon}: {[m.title() for m in members]}")

    # ── 6. Mapping final : field brut title-case → canonique title-case ───────
    result_map: Dict[str, str] = {}
    for f in cleaned:
        key = f.strip().title()
        result_map[key] = canonical_map.get(f.lower().strip(), key)

    return result_map
